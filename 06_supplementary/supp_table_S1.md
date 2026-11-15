Supplementary Table S1 | Datasets of the study (Results 1; Methods). a, The 14 Tahoe-100M cell lines. A condition is one cell line, compound name, dose and plate; treated cells are counted on the plate before any cap. b, The 13 plates used (plate 12 was excluded before analysis). c, The three compound splits of each dataset. On Tahoe-100M each split holds out 75 of the 372 chemical structures (375 compound names); the test file holds at most 500 cells per condition and the training file at most 150. d, Training seeds. 77 conditions of SW 1088 on plate 13 had no vehicle cells on their plate and are not counted (4,536 treated cells).

**a** | Cell lines of the Tahoe-100M subset.

| Cell line | Tissue | Plates | Compounds | Conditions | Treated cells |
|---|---|---|---|---|---|
| A-172 | Brain | 13 | 375 | 1,065 | 1,738,454 |
| H4 | Brain | 13 | 375 | 1,065 | 827,015 |
| SW 1088 | Brain | 11 | 234 | 520 | 94,220 |
| KATO III | Gastric | 13 | 375 | 1,065 | 1,670,000 |
| SNU-1 | Gastric | 13 | 375 | 1,063 | 882,379 |
| COLO 205 | Colorectal | 13 | 375 | 1,065 | 1,126,977 |
| HCT15 | Colorectal | 13 | 375 | 1,065 | 1,206,750 |
| HT-29 | Colorectal | 13 | 375 | 1,065 | 1,636,173 |
| LS 180 | Colorectal | 13 | 375 | 1,065 | 1,212,126 |
| LoVo | Colorectal | 13 | 375 | 1,065 | 2,177,636 |
| RKO | Colorectal | 13 | 375 | 1,065 | 1,775,480 |
| SW1417 | Colorectal | 13 | 375 | 1,065 | 1,292,509 |
| SW48 | Colorectal | 13 | 375 | 1,065 | 749,643 |
| SW480 | Colorectal | 13 | 375 | 1,065 | 4,679,378 |
| All |  | 13 | 375 | 14,363 | 21,068,740 |

**b** | Plates of the Tahoe-100M subset. Cells in the plate file include the vehicle cells.

| Plate | Cells in the plate file | Cell lines | Compounds | Conditions | Treated cells |
|---|---|---|---|---|---|
| 1 | 1,551,891 | 14 | 81 | 1,132 | 1,500,582 |
| 2 | 2,179,994 | 14 | 83 | 1,159 | 2,114,108 |
| 3 | 762,318 | 14 | 42 | 585 | 743,589 |
| 4 | 1,794,817 | 14 | 92 | 1,261 | 1,751,612 |
| 5 | 1,317,326 | 14 | 79 | 1,077 | 1,271,630 |
| 6 | 1,933,341 | 14 | 90 | 1,229 | 1,876,261 |
| 7 | 1,083,874 | 13 | 72 | 936 | 1,049,956 |
| 8 | 2,261,244 | 14 | 90 | 1,192 | 2,173,147 |
| 9 | 1,325,126 | 14 | 75 | 995 | 1,279,031 |
| 10 | 1,966,066 | 14 | 91 | 1,202 | 1,915,901 |
| 11 | 1,647,009 | 14 | 87 | 1,157 | 1,602,513 |
| 13 | 2,067,135 | 13 | 41 | 1,183 | 2,033,125 |
| 14 | 1,784,815 | 14 | 90 | 1,255 | 1,757,285 |

**c** | Compound splits. Split 1 was used for development and Splits 0 and 2 for confirmation.

| Dataset | Quantity | Split 1 (development) | Split 0 | Split 2 |
|---|---|---|---|---|
| Tahoe-100M | Held-out structures | 75 | 75 | 75 |
|  | Held-out compound names | 75 | 75 | 76 |
|  | Test conditions | 2,899 | 3,115 | 2,853 |
|  | Cells in the test file | 1,397,180 | 1,499,142 | 1,373,206 |
|  | Training conditions | 11,464 | 11,248 | 11,510 |
|  | Cells in the training file | 1,713,901 | 1,682,011 | 1,721,029 |
| sci-Plex | Held-out compounds | 37 | 37 | 38 |
|  | Test conditions | 443 | 444 | 455 |

**d** | Training seeds. Every evaluated model had the same seeds on every split.

| Dataset | Seeds per model and split | Seed numbers |
|---|---|---|
| Tahoe-100M | 3 | 0, 1, 2 |
| sci-Plex | 5 | 0, 1, 2, 3, 4 |

Sources: step15_data_record.json (sha256 06c4b547a8a1831a; parts a to c, Tahoe-100M), pathway_vs_fingerprint_v2.json (part c, sci-Plex), step15_model_analysis.json and step14_analysis.json (part d). The tissue of each cell line is the grouping stated in the Methods. Assembled by make_supp_table_s1.py.
