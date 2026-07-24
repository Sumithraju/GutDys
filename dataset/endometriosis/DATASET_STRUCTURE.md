# Endometriosis dataset structure (after preprocessing, before ML)

This is the table that `endometriosis_pipeline.py` `load_data()` reads. Produced by FASTQ → QIIME2/DADA2 → SILVA-138 → genus/phylum relative abundances, with PICRUSt2 predicting `beta_glucuronidase`. Two cohorts are combined; only gut (rectal/stool) samples are kept.

- **Samples (rows):** 211  (by study: {'PRJNA424567': 161, 'PRJNA1145097': 50})
- **Columns:** 27  (5 metadata + 22 features)
- **Classes:** {'Endometriosis': 115, 'Control': 96}

## Column schema

| # | Column | Role | Type | % missing | Description |
|---|--------|------|------|-----------|-------------|
| 1 | `sample_id` | Metadata | object | 0% | Unique run/sample ID |
| 2 | `study` | Metadata | object | 0% | Source study / batch |
| 3 | `diagnosis` | Metadata | object | 0% | **LABEL** — group |
| 4 | `sample_type` | Metadata | object | 0% | Body site |
| 5 | `country` | Metadata | object | 0% | Origin country |
| 6 | `Firmicutes` | Feature | float64 | 0% | Phylum relative abundance (%) |
| 7 | `Bacteroidetes` | Feature | float64 | 0% | Phylum relative abundance (%) |
| 8 | `Proteobacteria` | Feature | float64 | 0% | Phylum relative abundance (%) |
| 9 | `Actinobacteriota` | Feature | float64 | 0% | Phylum relative abundance (%) |
| 10 | `Desulfobacterota` | Feature | float64 | 0% | Phylum relative abundance (%) |
| 11 | `Bacteroides` | Feature | float64 | 0% | Genus relative abundance (%) |
| 12 | `Lactobacillus` | Feature | float64 | 0% | Genus relative abundance (%) |
| 13 | `Faecalibacterium` | Feature | float64 | 0% | Genus relative abundance (%) |
| 14 | `Bifidobacterium` | Feature | float64 | 0% | Genus relative abundance (%) |
| 15 | `Prevotella` | Feature | float64 | 0% | Genus relative abundance (%) |
| 16 | `Ruminococcus` | Feature | float64 | 0% | Genus relative abundance (%) |
| 17 | `Akkermansia` | Feature | float64 | 0% | Genus relative abundance (%) |
| 18 | `Escherichia_coli` | Feature | float64 | 0% | Genus relative abundance (%) — E. coli / Shigella proxy |
| 19 | `Collinsella` | Feature | float64 | 0% | Genus relative abundance (%) |
| 20 | `Roseburia` | Feature | float64 | 0% | Genus relative abundance (%) |
| 21 | `Blautia` | Feature | float64 | 0% | Genus relative abundance (%) |
| 22 | `Alistipes` | Feature | float64 | 0% | Genus relative abundance (%) |
| 23 | `Parabacteroides` | Feature | float64 | 0% | Genus relative abundance (%) |
| 24 | `Desulfovibrio` | Feature | float64 | 0% | Genus relative abundance (%) |
| 25 | `Sutterella` | Feature | float64 | 0% | Genus relative abundance (%) |
| 26 | `beta_glucuronidase` | Feature | float64 | 0% | PICRUSt2-predicted enzyme abundance (EC 3.2.1.31, estrobolome) |
| 27 | `shannon_diversity` | Feature | float64 | 16% | Shannon alpha-diversity index |

> **Note:** metadata columns are dropped before training. `beta_glucuronidase` is PICRUSt2-**predicted** (not measured) and is z-scored within each study to remove batch magnitude. `shannon_diversity` is excluded as a feature in the combined model (it behaved as a study/batch marker). Only gut samples are included (vaginal samples of PRJNA424567 were removed).

## Example — one full sample (transposed)
```
sample_id              SRR30152043
study                 PRJNA1145097
diagnosis                  Control
sample_type                  stool
country                        USA
Firmicutes               93.348261
Bacteroidetes             5.133693
Proteobacteria                 0.0
Actinobacteriota          1.406085
Desulfobacterota           0.11196
Bacteroides               5.081006
Lactobacillus             0.016465
Faecalibacterium          9.411222
Bifidobacterium                0.0
Prevotella                     0.0
Ruminococcus               6.72089
Akkermansia                    0.0
Escherichia_coli               0.0
Collinsella                    0.0
Roseburia                 0.243678
Blautia                   8.159905
Alistipes                      0.0
Parabacteroides                0.0
Desulfovibrio              0.11196
Sutterella                     0.0
beta_glucuronidase          624.55
shannon_diversity         5.384652
```