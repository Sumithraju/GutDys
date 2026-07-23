# Literature reference base — PCOS & Endometriosis gut microbiome

Consolidated evidence base backing `LITERATURE_EXPECTATIONS` in
`src/pcos_pipeline.py` and `src/endometriosis_pipeline.py`. Citation keys here
match the `refs` fields in those dicts.

**How to read this file.** Not every taxon gets an expected direction. A
direction is only asserted where the evidence is reasonably consistent across
studies. Where the literature is mixed, that is recorded explicitly as
`direction: NOT ASSERTED` with the reason — those taxa are still reported in
`taxa_group_comparison.csv` for exploration, they just don't count toward the
`[literature-check] N/M taxa match` score. Claiming a direction we can't
support would make that score meaningless.

---

## Source studies (data actually in this pipeline)

### `Lindheim2017` — SRP077213 *(our primary PCOS dataset, n=43)*
Lindheim L, et al. "Alterations in Gut Microbiome Composition and Barrier
Function Are Associated with Reproductive and Metabolic Defects in Women with
PCOS: A Pilot Study." *PLOS ONE* 2017;12(1):e0168390.
<https://doi.org/10.1371/journal.pone.0168390>

- n = 24 PCOS / 19 control, stool 16S. **Matches our loaded data exactly.**
- Alpha diversity: PCOS ~15% **lower** Faith's PD and observed OTUs
  (p = 0.027, 0.030).
- **CRITICAL CAVEAT:** *"No statistically significant differences were observed
  between PCOS patients and controls in bacterial taxa with a relative
  abundance >1% or in the Firmicutes:Bacteroidetes ratio."*
- The only significant taxa were **rare** (<1% abundance), all **lower in PCOS**:
  Tenericutes phylum (p < 0.0001, confirmed by qPCR), an unclassified
  ML615J-28 genus (p = 0.026), and an unclassified Bacteroidetes/S24-7 genus
  (p = 0.039).

> **Implication for this pipeline (important):** almost every feature we
> extract is an *abundant* taxon — exactly the class of taxa this paper
> reports as **not** differing significantly. The strongest real signal in this
> dataset (Tenericutes) is a phylum we do not currently extract at all. So
> modest classifier performance on SRP077213 is the *expected* result, not a
> bug, and the per-taxon direction matches below should be read as weak
> descriptive agreement, not confirmation of significant differences.

### `Zhang2017` — SRP085887 *(our second PCOS dataset, n=48)*
Zhang J, et al. "Dysbiosis of Gut Microbiota Associated with Clinical
Parameters in Polycystic Ovary Syndrome." *Frontiers in Microbiology*
2017;8:324. <https://doi.org/10.3389/fmicb.2017.00324>

- n = 33 PCOS / 15 control (PO=21, PN=12, CO=6, CN=9). **Verified against SRA
  BioSample `host disease` attributes — counts match exactly.**
- Sequencing: V3–V4, Illumina MiSeq **600-cycle kit (2×300 bp reads)**.
  *This is why DADA2 truncation must be ~280/240 for this study, not the
  250/220 used for Lindheim's 2×250 data — see `qiime2_process.sh`.*
- Richness (observed OTUs, Chao1): CN highest → CO/PN → PO lowest.
  **No** difference in Shannon or Simpson.
- Higher in PCOS-obese: *Bacteroides*, *Escherichia/Shigella*, *Streptococcus*,
  *Blautia*, *Parabacteroides* (CAG1).
- Higher in lean controls: *Akkermansia*, *Clostridium* IV, *Lactobacillus*,
  *Oscillibacter*, unclassified Ruminococcaceae.
- Enriched in obese controls specifically: *Collinsella*, *Paraprevotella*,
  *Slackia*.

### `Kommagani2024` — PRJNA1145097 *(our endometriosis dataset, n=50)*
Stool 16S, 31 control / 19 endometriosis. Labels recovered from H/E prefixes
in the `Library Name` field (no disease column in the SraRunTable).

---

## Recent syntheses (2023–2025) — the "last 4 years" evidence base

### `RepSci2024` — largest PCOS meta-analysis of observational studies
"Gut Microbiome Composition in Polycystic Ovary Syndrome Adult Women: A
Systematic Review and Meta-analysis of Observational Studies."
*Reproductive Sciences* 2024.
<https://link.springer.com/article/10.1007/s43032-023-01440-4>

- 14 cohort studies, **513 PCOS / 435 controls**.
- **No significant change in overall gut microbiome biodiversity in PCOS.**
- **Proteobacteria significantly higher in PCOS** (meta-analysis of 3 studies).

> Note this **conflicts** with `Lindheim2017`'s ~15% lower alpha diversity and
> partially with `Zhang2017`'s lower richness. Pooled across many studies the
> diversity effect washes out. We therefore treat `shannon_diversity` as a
> **weak/contested** expectation, not a firm one.

### `eClinMed2024` — individual-participant-level PCOS reanalysis
"Gut microbiota in women with polycystic ovary syndrome: an individual based
analysis of publicly available data." *eClinicalMedicine* 2024.
<https://www.thelancet.com/journals/eclinm/article/PIIS2589-5370(24)00463-2/fulltext>

- Pools raw data across studies (~948 individuals) rather than effect sizes —
  the closest published analogue to what this pipeline is doing.

### `MR2023` — Mendelian randomization (causal, not just correlational)
"The association between gut microbiome and PCOS: evidence from meta-analysis
and two-sample Mendelian randomization." *Front Microbiol* 2023;14:1203902.
<https://doi.org/10.3389/fmicb.2023.1203902>

- Genus-level enrichment in PCOS: **Bacteroides, Enterococcus,
  Escherichia-Shigella**; species-level: *Ruminococcus gnavus* group,
  *Parabacteroides distasonis*, *Bacteroides fragilis*.
- 8 taxa causally associated with PCOS; protective effect of genus
  *Sellimonas* survived Bonferroni correction.

### `PhysGen2025` — multi-body-site PCOS review with GRADE evidence grading
"Dysbiosis in PCOS: a systematic review of microbiome alterations across body
sites with GRADE assessment of evidence quality." *Physiological Genomics* 2025.
<https://journals.physiology.org/doi/full/10.1152/physiolgenomics.00072.2025>

### `HROpen2025` — endometriosis scoping review
"Microbiome dysbiosis and endometriosis: a systematic scoping review of current
literature and knowledge gaps." *Human Reproduction Open* 2025;2025(4):hoaf061.
<https://academic.oup.com/hropen/article/2025/4/hoaf061/8269774>

- Explicitly flags **pronounced heterogeneity** in taxonomic profiles across
  studies at every anatomical site — the reason several endometriosis taxa
  below are left `NOT ASSERTED`.

### `FrontMicro2025Endo` — endometriosis gut microbiota meta-analysis
"Association between endometriosis and gut microbiota: systematic review and
meta-analysis." *Front Microbiol* 2025;16:1552134.
<https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2025.1552134/full>

- 11 studies, 433 endometriosis / 1,294 controls (alpha & beta diversity).

### `Endobiota2019` — foundational endometriosis multi-site study
"The Endobiota Study: Comparison of Vaginal, Cervical and Gut Microbiota
Between Women with Stage 3/4 Endometriosis and Healthy Controls."
*Scientific Reports* 2019;9:2204. <https://doi.org/10.1038/s41598-019-39700-6>

- More stage 3/4 endometriosis women had **Shigella/Escherichia-dominant**
  stool microbiome.
- ***Prevotella* and *Dialister* decreased** in endometriosis.
- *Bacteroides* and *Prevotella* decreased further in more advanced disease.

---

## Consolidated expected directions

### PCOS

| Taxon | Expected higher in | Key refs | Confidence |
|---|---|---|---|
| Proteobacteria | PCOS | RepSci2024, Zhang2017 | **strong** (meta-analysis) |
| Escherichia_coli (Esch-Shigella) | PCOS | MR2023, Zhang2017 | **strong** |
| Bacteroides | PCOS | MR2023, Zhang2017 | moderate |
| Parabacteroides | PCOS | MR2023, Zhang2017 | moderate |
| Blautia | PCOS | Zhang2017 | weak (single study) |
| Collinsella | PCOS | Zhang2017 (obese controls), IR literature | weak/contested |
| Desulfovibrio | PCOS | general dysbiosis literature | weak |
| Lactobacillus | Control | Zhang2017 | moderate |
| Akkermansia | Control | Zhang2017 | moderate |
| Faecalibacterium | Control | SCFA/anti-inflammatory literature | moderate |
| Bifidobacterium | Control | general | weak |
| Roseburia | Control | butyrate-producer literature | weak |
| shannon_diversity | Control | Lindheim2017, Zhang2017 | **contested** (RepSci2024 finds none) |
| Firmicutes | — | — | **NOT ASSERTED** — Lindheim2017 explicitly found no F:B difference |
| Bacteroidetes | — | — | **NOT ASSERTED** — same |
| Prevotella | — | — | **NOT ASSERTED** — inconsistent across studies |
| Ruminococcus | — | — | **NOT ASSERTED** — our column merges true Ruminococcaceae with SILVA's reclassified `[Ruminococcus]_gnavus_group` (Lachnospiraceae); MR2023 links *gnavus* specifically to PCOS but our column can't isolate it |
| Alistipes, Sutterella | — | — | **NOT ASSERTED** — thin/mixed |

### Endometriosis

| Taxon | Expected higher in | Key refs | Confidence |
|---|---|---|---|
| Escherichia_coli (Esch-Shigella) | Endometriosis | Endobiota2019, HROpen2025 | **strong** |
| Proteobacteria | Endometriosis | HROpen2025 | moderate |
| Prevotella | Control | Endobiota2019 | moderate *(newly assertable — was previously excluded)* |
| Lactobacillus | Control | Endobiota2019, HROpen2025 | moderate |
| Faecalibacterium | Control | SCFA literature | moderate |
| Roseburia | Control | butyrate-producer literature | weak |
| Bifidobacterium | Control | general | weak |
| shannon_diversity | Control | FrontMicro2025Endo | weak/contested |
| Collinsella | Endometriosis | inflammation literature | weak |
| Desulfovibrio | Endometriosis | H2S/barrier-disruption literature | weak |
| Bacteroides | — | — | **NOT ASSERTED** — decreases in *advanced* disease but not consistently overall |
| Firmicutes, Bacteroidetes, Ruminococcus, Blautia, Alistipes, Parabacteroides, Sutterella, Akkermansia | — | — | **NOT ASSERTED** — HROpen2025 flags pronounced cross-study heterogeneity |

---

## Known gaps in our feature set

1. **Tenericutes is not extracted** — yet it is `Lindheim2017`'s single
   strongest finding (p < 0.0001, qPCR-confirmed) in the very dataset we use
   as primary. Worth adding to `qiime2_to_pipeline.py` (verify it appears in
   `table-L2.tsv` first, per the project rule of never hardcoding a taxon
   blind).
2. **Species-level resolution is impossible from 16S** — `MR2023`'s findings
   for *R. gnavus*, *P. distasonis*, *B. fragilis* cannot be tested here.
   `Escherichia_coli` in our schema is genus-level *Escherichia-Shigella*, a
   proxy only.
3. **Rare taxa (<1%) are where PCOS signal actually lives** per `Lindheim2017`,
   but relative-abundance features plus `SelectKBest` bias us toward abundant
   taxa.
4. **Obesity/BMI is a major confounder** — `Zhang2017` shows obese *controls*
   cluster with PCOS rather than lean controls. Our pipeline has no BMI
   feature, so some "PCOS signal" may be obesity signal. BMI *is* available in
   the SRA BioSample records for SRP085887 (`host body mass index`) if we want
   to model or stratify on it.
