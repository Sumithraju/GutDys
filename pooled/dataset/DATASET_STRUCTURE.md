# Pooled (3-class) dataset structure — SYNTHETIC / DEMO

⚠ **SYNTHETIC / DEMO data — not real patients.** This is the table `pooled_pipeline.py` `load_data()` reads. It demonstrates the multi-class (Control / Endometriosis / PCOS) version of the same pipeline. Feature values are already standardised (z-scored), so they are not percentages.

- **Samples (rows):** 720  (by study: {'Wang_2023_China': 100, 'Niafar_2022_Poland': 80, 'Liu_2022_China': 60, 'Hernandes_2020': 59, 'Jobira_2020_Austria': 54, 'D1_endo_stool': 49, 'Talwar_2025': 49, 'Zhu_2017_China': 48, 'Qi_2019_China': 48, 'Svensson_2021': 45, 'Guo_2022_China': 44, 'Do_2022': 31, 'Ata_2019': 28, 'MacSharry_2024': 25})
- **Columns:** 20  (6 metadata + 14 features)
- **Classes:** {'Control': 346, 'PCOS': 235, 'Endometriosis': 139}

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
| 9 | `Bacteroides` | Feature | float64 | 0% | Genus relative abundance (%) |
| 10 | `Lactobacillus` | Feature | float64 | 0% | Genus relative abundance (%) |
| 11 | `Faecalibacterium` | Feature | float64 | 0% | Genus relative abundance (%) |
| 12 | `Bifidobacterium` | Feature | float64 | 0% | Genus relative abundance (%) |
| 13 | `beta_glucuronidase` | Feature | float64 | 0% | PICRUSt2-predicted enzyme abundance (EC 3.2.1.31, estrobolome) |
| 14 | `shannon_diversity` | Feature | float64 | 0% | Shannon alpha-diversity index |
| 15 | `FB_ratio` | Feature | float64 | 0% | Firmicutes/Bacteroidetes ratio |
| 16 | `BL_ratio` | Feature | float64 | 0% | Bacteroides/Lactobacillus ratio |
| 17 | `Fae_Bact_ratio` | Feature | float64 | 0% | Faecalibacterium/Bacteroides ratio |
| 18 | `BG_Lacto_ratio` | Feature | float64 | 0% | beta-gluc./Lactobacillus ratio |
| 19 | `dysbiosis_index` | Feature | float64 | 0% | Composite dysbiosis index |
| 20 | `label` | Metadata | int64 | 0% | **LABEL** (numeric 0/1/2) |

> **Note:** `label` (0/1/2) is the numeric target; `diagnosis` is its text form. Both are metadata for training. This table also carries pre-computed ratio features (FB_ratio, etc.). Do **not** interpret these results biologically — the data is synthetic.

## Example — one full sample (transposed)
```
sample_id                   ENDO_01
study                 D1_endo_stool
diagnosis             Endometriosis
sample_type                   stool
country                     Unknown
Firmicutes                -2.029276
Bacteroidetes               -0.2464
Proteobacteria             2.152688
Bacteroides               -0.709607
Lactobacillus             -0.142223
Faecalibacterium          -0.825914
Bifidobacterium            0.917481
beta_glucuronidase         0.637462
shannon_diversity         -0.787891
FB_ratio                   1.197993
BL_ratio                    1.83023
Fae_Bact_ratio             0.498083
BG_Lacto_ratio             5.302519
dysbiosis_index            3.731396
label                             1
```