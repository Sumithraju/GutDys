# PCOS dataset structure (after preprocessing, before ML)

This is the table that `pcos_pipeline.py` `load_data()` reads. It is produced by the bioinformatics stage: FASTQ → QIIME2/DADA2 → SILVA-138 taxonomy → genus/phylum relative abundances, with PICRUSt2 predicting `beta_glucuronidase`.

- **Samples (rows):** 91  (SRP077213 = 43, SRP085887 = 48)
- **Columns:** 27  (5 metadata + 22 features)
- **Classes:** {'PCOS': 57, 'Control': 34}

## Column schema

| # | Column | Role | Type | % missing | Description |
|---|--------|------|------|-----------|-------------|
| 1 | `sample_id` | Metadata | object | 0% | Unique run ID (SRA accession) |
| 2 | `study` | Metadata | object | 0% | Source study / batch |
| 3 | `diagnosis` | Metadata | object | 0% | **LABEL** — Control / PCOS |
| 4 | `sample_type` | Metadata | object | 0% | Body site (stool) |
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
| 26 | `beta_glucuronidase` | Feature | float64 | 100% | PICRUSt2-predicted enzyme abundance (EC 3.2.1.31, estrobolome) |
| 27 | `shannon_diversity` | Feature | float64 | 2% | Shannon alpha-diversity index |

> **Note:** metadata columns are dropped before training. `beta_glucuronidase` is currently 100% missing for PCOS (PICRUSt2 was run only for the endometriosis arm), so it is auto-dropped; the PCOS model uses taxa + Shannon. Relative-abundance features sum to ~100% within a sample.

## Example — one full sample (transposed)

```
sample_id             SRR4457864
study                  SRP077213
diagnosis                Control
sample_type                stool
country                  Unknown
Firmicutes             74.925889
Bacteroidetes          24.140865
Proteobacteria          0.505051
Actinobacteriota        0.296443
Desulfobacterota        0.087835
Bacteroides            12.390206
Lactobacillus           0.139987
Faecalibacterium        8.495279
Bifidobacterium              0.0
Prevotella              0.115283
Ruminococcus            7.899649
Akkermansia             0.019214
Escherichia_coli        0.041173
Collinsella                  0.0
Roseburia                    0.0
Blautia                 8.912495
Alistipes              10.968379
Parabacteroides              0.0
Desulfovibrio           0.087835
Sutterella              0.403491
beta_glucuronidase           NaN
shannon_diversity       7.130699
```