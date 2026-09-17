# Case study — ID-only SASRec vs multimodal MM-SASRec

* ID-only run: `results/runs/sasrec_20260917-060603_47936c`
* Multimodal run: `results/runs/mm_gated_iddrop_20260917-053644_ec427f`
* Test users: 100,000
* Recall@20 — ID-only **0.1253**, multimodal **0.1311**

| category | users | share |
|---|---|---|
| both retrieve | 8,962 | 9.0% |
| **rescued by multimodal** | 4,152 | 4.15% |
| **broken by multimodal** | 3,563 | 3.56% |
| both fail | 83,323 | 83.3% |

Item ids are the original MicroLens ids. The 100K subset has no titles, so
`content nbrs` lists the target's nearest neighbours in the raw text+image
feature space — a semantic proxy for what the item is about.

## Rescued by multimodal

### user 5

* history (raw ids, oldest first): `[0, 6472, 6477, 11823, 14903, 14021, 16261, 15643, 16538, 18596]`
* ground truth: **18036** (popularity bucket `head`, train freq 105, cold=False)
* rank: ID-only `41` | multimodal `17`
* ID-only top-5: `[18892, 18783, 19086, 19147, 18929]`
* multimodal top-5: `[18134, 18929, 19086, 18892, 18452]`
* content nbrs of target: `[9936, 14095, 9589, 16298, 14850]`

### user 12

* history (raw ids, oldest first): `[1, 2, 301, 406, 931, 939]`
* ground truth: **1182** (popularity bucket `head`, train freq 59, cold=False)
* rank: ID-only `206` | multimodal `19`
* ID-only top-5: `[1226, 1005, 1026, 667, 1465]`
* multimodal top-5: `[1026, 1384, 398, 1465, 1408]`
* content nbrs of target: `[2678, 15497, 2998, 15411, 2471]`

### user 26

* history (raw ids, oldest first): `[1, 210, 2699, 3036, 3261, 7804]`
* ground truth: **7875** (popularity bucket `head`, train freq 378, cold=False)
* rank: ID-only `23` | multimodal `12`
* ID-only top-5: `[10225, 8705, 8595, 8163, 7893]`
* multimodal top-5: `[7893, 8163, 8239, 7807, 8587]`
* content nbrs of target: `[4015, 19682, 4567, 12551, 6762]`

### user 77

* history (raw ids, oldest first): `[1, 2, 888, 1549, 1640, 2226]`
* ground truth: **3417** (popularity bucket `head`, train freq 120, cold=False)
* rank: ID-only `128` | multimodal `15`
* ID-only top-5: `[2812, 2385, 2093, 3548, 4499]`
* multimodal top-5: `[1408, 1600, 3548, 2462, 3255]`
* content nbrs of target: `[9958, 3939, 122, 17950, 5020]`

### user 81

* history (raw ids, oldest first): `[1, 888, 272, 1889, 4435, 8160]`
* ground truth: **8556** (popularity bucket `head`, train freq 61, cold=False)
* rank: ID-only `93` | multimodal `17`
* ID-only top-5: `[8595, 9868, 8589, 6554, 12186]`
* multimodal top-5: `[8589, 7151, 8595, 8186, 8722]`
* content nbrs of target: `[14095, 7587, 17904, 9341, 6701]`

## Broken by multimodal (counter-examples)

### user 13

* history (raw ids, oldest first): `[1, 2, 5, 1274, 1889, 3522, 4932, 4668]`
* ground truth: **7804** (popularity bucket `head`, train freq 319, cold=False)
* rank: ID-only `12` | multimodal `314`
* ID-only top-5: `[8595, 6554, 6415, 5999, 9978]`
* multimodal top-5: `[2978, 3548, 6554, 4389, 4986]`
* content nbrs of target: `[9186, 4198, 8935, 3128, 16884]`

### user 57

* history (raw ids, oldest first): `[1, 2, 657, 1706, 3598, 7804, 8876, 11553, 13379, 19512]`
* ground truth: **19529** (popularity bucket `tail`, train freq 4, cold=False)
* rank: ID-only `6` | multimodal `29`
* ID-only top-5: `[17906, 19104, 19015, 17896, 19254]`
* multimodal top-5: `[19015, 18821, 19540, 19519, 19109]`
* content nbrs of target: `[6249, 18070, 6389, 14369, 2347]`

### user 121

* history (raw ids, oldest first): `[4, 3748, 4601, 4569, 6047, 5999, 8199, 9853, 14956, 15127, 17142]`
* ground truth: **19464** (popularity bucket `tail`, train freq 21, cold=False)
* rank: ID-only `19` | multimodal `43`
* ID-only top-5: `[17310, 17672, 18439, 17523, 17674]`
* multimodal top-5: `[17674, 16089, 17310, 18311, 17672]`
* content nbrs of target: `[8082, 5776, 7073, 13450, 6702]`

### user 140

* history (raw ids, oldest first): `[1, 3036, 4311, 4986]`
* ground truth: **7213** (popularity bucket `middle`, train freq 39, cold=False)
* rank: ID-only `15` | multimodal `72`
* ID-only top-5: `[6554, 5999, 9868, 6415, 6318]`
* multimodal top-5: `[6554, 5999, 7381, 4273, 3548]`
* content nbrs of target: `[7922, 8749, 13745, 6522, 1275]`

### user 171

* history (raw ids, oldest first): `[4, 2700, 5117, 2980, 11648, 17380]`
* ground truth: **18036** (popularity bucket `head`, train freq 105, cold=False)
* rank: ID-only `13` | multimodal `28`
* ID-only top-5: `[18783, 18892, 18134, 18371, 18299]`
* multimodal top-5: `[18929, 18134, 18587, 17921, 18596]`
* content nbrs of target: `[9936, 14095, 9589, 16298, 14850]`

## Hard cases neither model solves

### user 0

* history (raw ids, oldest first): `[0, 5412, 10209, 14805, 16123, 17741]`
* ground truth: **13185** (popularity bucket `middle`, train freq 21, cold=False)
* rank: ID-only `2074` | multimodal `1425`
* ID-only top-5: `[18620, 17193, 18418, 19468, 17333]`
* multimodal top-5: `[18420, 18620, 10565, 18596, 17451]`
* content nbrs of target: `[14003, 15755, 10375, 11391, 1800]`

### user 1

* history (raw ids, oldest first): `[0, 1162, 6196, 2513]`
* ground truth: **9535** (popularity bucket `tail`, train freq 3, cold=False)
* rank: ID-only `7068` | multimodal `12666`
* ID-only top-5: `[12151, 10542, 8951, 10806, 11673]`
* multimodal top-5: `[10624, 11673, 13009, 10466, 9596]`
* content nbrs of target: `[7258, 9048, 10565, 6333, 14615]`

### user 2

* history (raw ids, oldest first): `[0, 12827, 11904, 6905, 14037, 18346]`
* ground truth: **19340** (popularity bucket `tail`, train freq 17, cold=False)
* rank: ID-only `30` | multimodal `175`
* ID-only top-5: `[18991, 18700, 18048, 19563, 18600]`
* multimodal top-5: `[18525, 18048, 18950, 18134, 18030]`
* content nbrs of target: `[16053, 12534, 16638, 15834, 7227]`

### user 3

* history (raw ids, oldest first): `[0, 1516, 2521, 8732, 15698, 16062, 16676, 17855, 18701]`
* ground truth: **19618** (popularity bucket `tail`, train freq 1, cold=False)
* rank: ID-only `4317` | multimodal `1689`
* ID-only top-5: `[18929, 19086, 18955, 18950, 18892]`
* multimodal top-5: `[18929, 18134, 18596, 19565, 17921]`
* content nbrs of target: `[8084, 15638, 12958, 15498, 9225]`

### user 4

* history (raw ids, oldest first): `[0, 592, 7090, 6341, 10109]`
* ground truth: **16835** (popularity bucket `middle`, train freq 33, cold=False)
* rank: ID-only `4484` | multimodal `1646`
* ID-only top-5: `[12277, 11843, 12718, 14070, 4880]`
* multimodal top-5: `[14991, 13215, 15558, 15487, 10239]`
* content nbrs of target: `[5413, 19719, 9984, 4871, 15560]`
