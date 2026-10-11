"""In-batch pooling, false-negative exclusions, loss and Trainer integration."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.data.in_batch_sampler import InBatchNegativeSampler
from src.data.negative_sampler import build_training_interaction_keys
from src.models.loss import sampled_softmax_logits, sampled_softmax_loss
from src.training.factory import build_model
from src.training.trainer import Trainer
from src.utils.config import Config


def test_pool_masks_train_seen_and_keeps_future_labels_legal():
    # User 0: train=[1,2], val=4, test=5. Other users supply 4 and 5
    # as their *training* positives; user 0 must be allowed to rank them down.
    flat = np.array([1,2,4,5, 3,4,8,9, 6,5,8,9])
    keys = build_training_interaction_keys(flat, np.array([0,4,8,12]), np.array([2,2,2]), 10)
    sampler = InBatchNegativeSampler(10, keys)
    targets = torch.tensor([[0,2], [0,4], [0,5]])
    batch = sampler.sample(torch.arange(3), targets)
    assert batch.item_ids.tolist() == [2,4,5]
    assert batch.valid_mask.tolist() == [[False,True,True], [True,False,True], [True,True,False]]
    # A known item outside the truncated input window is still excluded.
    extra = InBatchNegativeSampler(10, np.r_[keys, 4])
    assert not extra.sample(torch.arange(3), targets).valid_mask[0,1]


def test_duplicate_positives_are_deduplicated_and_masked_for_every_owner():
    sampler = InBatchNegativeSampler(10, [])
    batch = sampler.sample(torch.tensor([0,1,1]), torch.tensor([[0,2], [3,4], [0,4]]))
    assert batch.item_ids.tolist() == [2,4]
    assert batch.valid_mask.tolist() == [[False,True], [True,False], [True,False]]
    assert len(batch.item_ids) <= 3
    # Include the complete row's positives in the exclusion, not only the last.
    batch = sampler.sample(torch.tensor([0,1]), torch.tensor([[2,4], [0,2]]))
    assert batch.item_ids.tolist() == [4]
    assert batch.valid_mask.tolist() == [[False], [True]]


@pytest.mark.parametrize("target", [torch.tensor([[0,2]]), torch.zeros((3,2),dtype=torch.long),
    torch.full((3,2),2), torch.tensor([[2,0],[0,0]]),
    torch.empty((0,2),dtype=torch.long), torch.empty((2,0),dtype=torch.long)])
def test_empty_singleton_and_all_duplicate_batches(target):
    batch = InBatchNegativeSampler(10, []).sample(torch.arange(len(target)), target)
    assert batch.item_ids.numel() == 0
    assert batch.valid_mask.shape == (len(target),0)


def test_shared_logits_and_loss_match_independent_reference():
    torch.manual_seed(7)
    seq = torch.randn(3,2,4,requires_grad=True)
    pos = torch.randn(3,2,4,requires_grad=True)
    neg = torch.randn(3,2,4,requires_grad=True)
    pool = torch.randn(3,4,requires_grad=True)
    mask = torch.tensor([[False,True,True], [True,False,False], [False,False,False]])
    target = torch.tensor([[0,1], [2,3], [4,0]])
    logits = sampled_softmax_logits(seq,pos,neg,.7,pool,mask)
    assert logits.shape == (3,2,6)
    assert torch.isneginf(logits[0,:,3]).all()
    losses = []
    for b in range(3):
        for t in range(2):
            if target[b,t] == 0:
                continue
            values = [(seq[b,t]*pos[b,t]).sum()/.7]
            values += [(seq[b,t]*v).sum()/.7 for v in neg[b]]
            values += [(seq[b,t]*pool[k]).sum()/.7 for k in range(3) if mask[b,k]]
            losses.append(F.cross_entropy(torch.stack(values)[None,:], torch.zeros(1,dtype=torch.long)))
    actual = sampled_softmax_loss(seq,pos,neg,target,.7,pool,mask)
    expected = torch.stack(losses).mean()
    assert torch.allclose(actual,expected,atol=1e-6)
    actual.backward()
    assert pool.grad is not None and pool.grad.abs().sum() > 0
    assert torch.isfinite(seq.grad).all() and torch.isfinite(pool.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32,torch.bfloat16])
def test_all_masked_negatives_match_baseline_without_nan(dtype):
    seq = torch.randn(2,3,4,dtype=dtype,requires_grad=True)
    pos = torch.randn(2,3,4,dtype=dtype,requires_grad=True)
    neg = torch.randn(2,2,4,dtype=dtype,requires_grad=True)
    pool = torch.randn(2,4,dtype=dtype,requires_grad=True)
    target = torch.tensor([[0,1,2],[0,3,4]])
    base = sampled_softmax_loss(seq,pos,neg,target)
    loss = sampled_softmax_loss(seq,pos,neg,target,in_batch_neg_emb=pool,
        in_batch_neg_mask=torch.zeros((2,2),dtype=torch.bool))
    assert torch.equal(base,loss)
    loss.backward()
    assert torch.isfinite(seq.grad).all() and torch.equal(pool.grad,torch.zeros_like(pool))
    empty = neg[:,:0]
    zero = sampled_softmax_loss(seq,pos,empty,target,in_batch_neg_emb=pool,
        in_batch_neg_mask=torch.zeros((2,2),dtype=torch.bool))
    assert zero.item() == 0 and torch.isfinite(zero)


def test_shared_pool_requires_boolean_row_mask():
    seq = pos = torch.ones(2,2,3)
    neg = torch.ones(2,0,3)
    pool = torch.ones(2,3)
    for mask in (None, torch.ones(2,3,dtype=torch.bool), torch.ones(2,2)):
        with pytest.raises(ValueError):
            sampled_softmax_logits(seq,pos,neg,in_batch_neg_emb=pool,in_batch_neg_mask=mask)


def test_legacy_row_embeddings_still_work():
    seq = pos = torch.ones(2,2,3)
    neg = torch.ones(2,0,3)
    logits = sampled_softmax_logits(seq,pos,neg,in_batch_neg_emb=torch.ones(2,4,3))
    assert logits.shape == (2,2,5)


@pytest.mark.parametrize("users,target", [
    (torch.tensor([0]),torch.tensor([1])),
    (torch.tensor([0,1]),torch.tensor([[1]])),
    (torch.tensor([0]),torch.tensor([[11]])),
    (torch.tensor([0]),torch.tensor([[-1]])),
    (torch.tensor([-1]),torch.tensor([[1]])),
    (torch.tensor([0]),torch.tensor([[1.]])),
])
def test_sampler_rejects_invalid_ids_and_shapes(users,target):
    with pytest.raises(ValueError):
        InBatchNegativeSampler(10,[]).sample(users,target)


@pytest.mark.parametrize("name", ["two_tower","sasrec","mm_sasrec"])
@pytest.mark.parametrize("num_negatives", [0,4])
def test_trainer_uses_safe_pool_and_updates_model(name,num_negatives,synthetic_data,synthetic_dir,tmp_path):
    cfg = Config({
        "model": {"name": name, "hidden_size": 128 if name == "two_tower" else 8,
                  "num_layers": 1, "num_heads": 2, "dropout": 0., "max_seq_len": 8,
                  "max_history": 8, "fusion": "concat", "normalize_embeddings": True,
                  "modalities": {"text": True,"image": True, **({"id": True} if name == "mm_sasrec" else {})}},
        "data": {"processed_dir": str(synthetic_dir),"feature_dir": str(synthetic_dir)},
        "loss": {"type": "sampled_softmax","num_negatives": num_negatives,"in_batch_negatives": True},
        "training": {"batch_size": 16,"epochs": 1,"amp": False,"num_workers": 0},
        "evaluation": {"ks": [5],"item_chunk_size": 128},
    })
    model = build_model(cfg,synthetic_data,device=torch.device("cpu"))
    trainer = Trainer(model,cfg,synthetic_data,tmp_path/name,torch.device("cpu"))
    original = trainer.in_batch_sampler.sample
    calls = []
    def checked(users,target):
        batch = original(users,target)
        for row,user in enumerate(users.tolist()):
            legal = batch.item_ids[batch.valid_mask[row]].tolist()
            assert not set(legal) & set(synthetic_data.train_items(user).tolist())
            assert all(1 <= item <= synthetic_data.num_items for item in legal)
        calls.append(batch.valid_mask.sum().item())
        return batch
    trainer.in_batch_sampler.sample = checked
    before = {name: p.detach().clone() for name,p in model.named_parameters() if p.requires_grad}
    metrics = trainer.train_epoch(0,trainer._train_loader())
    assert calls and sum(calls) > 0 and np.isfinite(metrics["train_loss"])
    assert any(not torch.equal(before[n],p) for n,p in model.named_parameters() if n in before)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
def test_cuda_sampler_matches_cpu():
    users = torch.arange(3)
    target = torch.tensor([[0,2],[0,4],[0,5]])
    keys = np.array([1,2,13,14,25,26])
    cpu = InBatchNegativeSampler(10,keys).sample(users,target)
    gpu = InBatchNegativeSampler(10,keys).sample(users.cuda(),target.cuda())
    assert torch.equal(cpu.item_ids,gpu.item_ids.cpu())
    assert torch.equal(cpu.valid_mask,gpu.valid_mask.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
def test_cuda_amp_shared_loss_backward():
    seq = torch.randn(3,2,8,device="cuda",requires_grad=True)
    pos = torch.randn(3,2,8,device="cuda",requires_grad=True)
    neg = torch.randn(3,4,8,device="cuda",requires_grad=True)
    pool = torch.randn(3,8,device="cuda",requires_grad=True)
    mask = ~torch.eye(3,dtype=torch.bool,device="cuda")
    with torch.autocast("cuda",dtype=torch.float16):
        loss = sampled_softmax_loss(seq,pos,neg,torch.ones(3,2,dtype=torch.long,device="cuda"),
            temperature=.07,in_batch_neg_emb=pool,in_batch_neg_mask=mask)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(seq.grad).all()
    assert torch.isfinite(pool.grad).all() and pool.grad.abs().sum() > 0
