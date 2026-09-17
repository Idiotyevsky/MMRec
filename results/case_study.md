# Case study — ID-only SASRec vs multimodal MM-SASRec

* ID-only run: `results/runs/sasrec_20260917-023725_63eba2`
* Multimodal run: `results/runs/mm_gated_iddrop_20260917-023525_19c90f`
* Test users: 100,000
* Recall@20 — ID-only **0.0784**, multimodal **0.0860**

| category | users | share |
|---|---|---|
| both retrieve | 5,220 | 5.2% |
| **rescued by multimodal** | 3,376 | 3.38% |
| **broken by multimodal** | 2,615 | 2.61% |
| both fail | 88,789 | 88.8% |

Item ids are the original MicroLens ids. The 100K subset has no titles, so
`content nbrs` lists the target's nearest neighbours in the raw text+image
feature space — a semantic proxy for what the item is about.

## Rescued by multimodal

### user 6

* history (raw ids, oldest first): `[0, 593, 4334, 4946, 10760, 14643, 17547]`
* ground truth: **17593** (popularity bucket `tail`, train freq 11, cold=False)
* rank: ID-only `88` | multimodal `8`
* ID-only top-5: `[11128, 8482, 3891, 10177, 7520]`
* multimodal top-5: `[13184, 10177, 15167, 3121, 7889]`
* content nbrs of target: `[11786, 13952, 12385, 4422, 18992]`

### user 9

* history (raw ids, oldest first): `[0, 10459, 11347, 11869]`
* ground truth: **12558** (popularity bucket `head`, train freq 44, cold=False)
* rank: ID-only `42` | multimodal `5`
* ID-only top-5: `[12212, 8432, 7911, 9027, 13393]`
* multimodal top-5: `[11544, 10527, 14982, 10662, 12558]`
* content nbrs of target: `[19450, 10458, 15586, 19539, 10973]`

### user 13

* history (raw ids, oldest first): `[1, 2, 5, 1274, 1889, 3522, 4932, 4668]`
* ground truth: **7804** (popularity bucket `head`, train freq 319, cold=False)
* rank: ID-only `127` | multimodal `20`
* ID-only top-5: `[4, 1065, 2437, 5375, 4311]`
* multimodal top-5: `[4, 3598, 888, 1556, 3171]`
* content nbrs of target: `[9186, 4198, 8935, 3128, 16884]`

## Broken by multimodal (counter-examples)

### user 38

* history (raw ids, oldest first): `[1, 1540, 3417, 3598, 4118, 4211, 4201, 3521, 4311]`
* ground truth: **5375** (popularity bucket `middle`, train freq 41, cold=False)
* rank: ID-only `15` | multimodal `37`
* ID-only top-5: `[9659, 4932, 2462, 12032, 2571]`
* multimodal top-5: `[4932, 3890, 272, 3602, 3912]`
* content nbrs of target: `[5768, 13737, 17856, 2381, 4151]`

### user 110

* history (raw ids, oldest first): `[4, 1629, 2751, 477, 7118, 6975, 8611, 5399]`
* ground truth: **14121** (popularity bucket `middle`, train freq 40, cold=False)
* rank: ID-only `15` | multimodal `119`
* ID-only top-5: `[7207, 2768, 8689, 8784, 1937]`
* multimodal top-5: `[16261, 365, 2514, 2495, 84]`
* content nbrs of target: `[19095, 16298, 1636, 18910, 15833]`

### user 119

* history (raw ids, oldest first): `[4, 18, 135, 242, 267, 2522, 2652, 3171, 3912, 4201, 4713, 4785, 4903, 4968, 5029, 5216, 5388, 5554, 6323, 6415, 6324, 7148, 9597, 9659, 9966, 9990, 9792, 9973, 9269, 9088, 11186, 1006, 10199, 11004, 10830, 12024, 12186, 12447, 11193, 12604, 12821, 12805, 12878, 14374, 14472, 14525, 14723, 16236, 16865, 16880, 17239, 17585, 17776, 18301]`
* ground truth: **18974** (popularity bucket `tail`, train freq 11, cold=False)
* rank: ID-only `5` | multimodal `31`
* ID-only top-5: `[15288, 16471, 19109, 8051, 18974]`
* multimodal top-5: `[19121, 12982, 9802, 16169, 10019]`
* content nbrs of target: `[15936, 8524, 13816, 9099, 1472]`

## Hard cases neither model solves

### user 0

* history (raw ids, oldest first): `[0, 5412, 10209, 14805, 16123, 17741]`
* ground truth: **13185** (popularity bucket `middle`, train freq 21, cold=False)
* rank: ID-only `5315` | multimodal `3642`
* ID-only top-5: `[18095, 17333, 17338, 15208, 9561]`
* multimodal top-5: `[7388, 18095, 11684, 17333, 10726]`
* content nbrs of target: `[14003, 15755, 10375, 11391, 1800]`

### user 1

* history (raw ids, oldest first): `[0, 1162, 6196, 2513]`
* ground truth: **9535** (popularity bucket `tail`, train freq 3, cold=False)
* rank: ID-only `10066` | multimodal `2428`
* ID-only top-5: `[10162, 2930, 10358, 4106, 12151]`
* multimodal top-5: `[2990, 11767, 6341, 6822, 5316]`
* content nbrs of target: `[7258, 9048, 10565, 6333, 14615]`

### user 2

* history (raw ids, oldest first): `[0, 12827, 11904, 6905, 14037, 18346]`
* ground truth: **19340** (popularity bucket `tail`, train freq 17, cold=False)
* rank: ID-only `858` | multimodal `341`
* ID-only top-5: `[12666, 12880, 15440, 17134, 18551]`
* multimodal top-5: `[15952, 7582, 16362, 14003, 17104]`
* content nbrs of target: `[16053, 12534, 16638, 15834, 7227]`
