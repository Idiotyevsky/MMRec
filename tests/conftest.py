import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory) -> Path:
    from src.data.synthetic import make_synthetic_dataset

    d = tmp_path_factory.mktemp("synthetic")
    return make_synthetic_dataset(d, num_users=60, num_items=40, seed=0)


@pytest.fixture(scope="session")
def synthetic_cold_dir(tmp_path_factory) -> Path:
    from src.data.synthetic import make_synthetic_dataset

    d = tmp_path_factory.mktemp("synthetic_cold")
    return make_synthetic_dataset(d, num_users=80, num_items=50, cold_ratio=0.15, seed=1)


@pytest.fixture(scope="session")
def synthetic_data(synthetic_dir):
    from src.data.dataset import ProcessedData

    return ProcessedData.load(synthetic_dir)
