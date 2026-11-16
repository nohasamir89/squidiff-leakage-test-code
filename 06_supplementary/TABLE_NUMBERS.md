# Published numbers of the supplementary tables and figures, and the files that build them

The supplementary tables and figures are numbered in the paper in the order in which the paper first cites them.
The scripts and their output files were written before that order was fixed, and they keep the numbers they were
built with. They were not renamed: their names and checksums are recorded in `MANIFEST.csv`
and in the "Sources" line under each table of the supplementary file.

**To check a table of the paper, find its published number in the first column: the row names the script that
built it and its two output files** (a formatted table, and a long-format file with every value at full
precision). The output files in this package are the ones the scripts wrote; their checksums are in
`MANIFEST.csv`.

Each script reads only committed records and checks each one against the sha256 value written in the script. It
also checks that the record is tracked and unmodified in git, and it refuses to overwrite an existing output. The
scripts are written for the layout of the research repository, where the records are under `leakage_results/`
and `tahoe_results/` and the scripts are run from the top folder; in this package the same records are in
`07_records/`. To rebuild a table, run its script in a checkout of the research repository.

## Tables

| Published number | Title | Script | Output files |
|---|---|---|---|
| Table S1 | Datasets of the study | `make_supp_table_s1.py` | `supp_table_S1.md`, `supp_table_S1.csv` |
| Table S2 | Gene identity and population distance of the models and the references | `make_supp_table_s5.py` | `supp_table_S5.md`, `supp_table_S5.csv` |
| Table S3 | What the population distance of the original model is made of, on Tahoe-100M | `make_supp_table_s4_s7.py` | `supp_table_S4.md`, `supp_table_S4.csv` |
| Table S4 | Training objective on training and on held-out cells | `make_supp_table_s2.py` | `supp_table_S2.md`, `supp_table_S2.csv` |
| Table S5 | Glioblastoma: gene-panel sweep and reproduction of the published analysis | `make_supp_table_s3.py` | `supp_table_S3.md`, `supp_table_S3.csv` |
| Table S6 | Response correlation on sci-Plex | `make_supp_table_s6.py` | `supp_table_S6.md`, `supp_table_S6.csv` |
| Table S7 | What the population distance of the rewired model is made of, on Tahoe-100M | `make_supp_table_s4_s7.py` | `supp_table_S7.md`, `supp_table_S7.csv` |
| Table S8 | Settings of the experimental references, sampling steps and entries set to zero | `make_supp_table_s9.py` | `supp_table_S9.md`, `supp_table_S9.csv` |
| Table S9 | Interventions on the compound input of the rewired model on sci-Plex | `make_supp_table_s10.py` | `supp_table_S10.md`, `supp_table_S10.csv` |
| Table S10 | Sensitivity of the comparisons with the cross-validated ridge to the fold assignment | `make_supp_table_s20.py` | `supp_table_S20.md`, `supp_table_S20.csv` |
| Table S11 | Separation of the two changes of the rewired model on sci-Plex | none: the values were entered from the record named under the table | none |
| Table S12 | The rewired model against the original model on Tahoe-100M | `make_supp_table_s12.py` | `supp_table_S12.md`, `supp_table_S12.csv` |
| Table S13 | Classifier-free guidance | `make_supp_table_s13.py` | `supp_table_S13.md`, `supp_table_S13.csv` |
| Table S14 | The discrimination score on Tahoe-100M | `make_supp_table_s15.py` | `supp_table_S15.md`, `supp_table_S15.csv` |
| Table S15 | Response-magnitude agreement | `make_supp_table_s16.py` | `supp_table_S16.md`, `supp_table_S16.csv` |
| Table S16 | Spread agreement on Tahoe-100M and its decomposition | `make_supp_table_s17.py` | `supp_table_S17.md`, `supp_table_S17.csv` |
| Table S17 | Recognition by RNA fingerprinting on Tahoe-100M | `make_supp_table_s18.py` | `supp_table_S18.md`, `supp_table_S18.csv` |

Two scripts need a remark. `make_supp_table_s4_s7.py` builds two tables in one run (the published Tables S3
and S7; files `supp_table_S4.*` and `supp_table_S7.*`). `make_supp_table_s20.py` builds the published Table S10.

All scripts import `supp_table_common.py` (shared checks and formatting) and use the Python standard library only.
In the research repository, from its top folder:

    python3 make_supp_table_s5.py        # builds the published Table S2

## Figures

The three supplementary figures are drawn by `plot_supp_figs.R` (folder `05_figures` of this package; R with
ggplot2 and jsonlite). Before it draws, the script checks its values against panels b, c and d of
`fig5_values.csv`, the file the tables of the same measures were checked against. Like the table scripts, it is
written for the layout of the research repository.

| Published number | Title | Image written by the script | Panel of `fig5_values.csv` |
|---|---|---|---|
| Fig. S1 | Response-magnitude agreement | `supp_fig_S2_magnitude.png`, `supp_fig_S2_magnitude.pdf` | b |
| Fig. S2 | Spread agreement on Tahoe-100M | `supp_fig_S1_spread.png`, `supp_fig_S1_spread.pdf` | c |
| Fig. S3 | RNA-fingerprinting recognition on Tahoe-100M | `supp_fig_S3_fingerprinting.png`, `supp_fig_S3_fingerprinting.pdf` | d |

The image files are not part of this package; the script draws them from the records.

## Notes

The supplementary notes have no script of their own. Each names its record in the text or in a "Source" line, and
the tables it rests on are listed above.
