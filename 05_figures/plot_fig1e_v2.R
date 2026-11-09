# plot_fig1e_v2.R - Fig. 1e: reproduction of the published glioblastoma analysis at 200 genes, and its failure
#                   at 2,000 genes.
#                   v2: reads the two re-evaluations on the correct scale (24 Oct 2026; the decoded predictions
#                   are no longer passed through log1p a second time):
#                     200 genes   eval_gbm_deltaz_200_v2.py, run_eval_200_v2.sh, SLURM log eval_200_v2_18927698.out
#                     2,000 genes eval_gbm_deltaz_v2.py,     run_eval_gbm_v2.sh,  SLURM log eval_v2_18927697.out
#                   Compared with plot_fig1e.R (md5 dee87eecfd6b6694c541995d580783ec) only five things change:
#                     1. the two log paths (the corrected logs);
#                     2. the output names carry "_v2", so the old figure and values file are not overwritten;
#                     3. the line under the plot names the measure correctly ("mean expression", manuscript
#                        label fix of 9 Nov 2026), which adds one line, so the figure is 0.1 in taller;
#                     4. two checks were added before drawing (see "checks" below);
#                     5. the y axis keeps 3 % of headroom above 1, because the corrected 200-gene reconstruction
#                        values (0.990 and 0.996) sit at the top edge and the points were cut in half there.
#                        The axis still starts at 0 and the tick marks are unchanged.
#
# Purpose: show, per cell type, the Pearson correlation between the predicted and measured mean expression of the
#          treated cells (across all genes of the panel) for each of the five held-out compounds, at 200 genes and at
#          2,000 genes, together with the vehicle reconstruction at each panel size (Pearson correlation between
#          the mean expression of reconstructed and of measured vehicle cells).
# Logic:   the evaluation scripts write no table, so the two evaluation logs are the record. They are parsed here:
#            per-compound rows   "  tumor           panobinostat     0.831   0.676    5114"   (R is the first number)
#            reconstruction rows "  tumor           : recon R = 0.990  r2 = 0.973  (n_enc=300)"
#            summary rows        "  tumor           : R = 0.844  (paper ~0.91; n_drugs=5)"  (used only as a check)
#          Published values are the two numbers reported in He et al. (Squidiff, Nat. Methods 2026) for this
#          analysis: 0.912 (tumour cells) and 0.915 (oligodendrocytes); they are the only typed-in numbers.
#          Checks before drawing (the script stops if either fails):
#            a. every parsed correlation lies in [0, 1], because the y axis is fixed to [0, 1] and ggplot would
#               otherwise drop a point outside it with only a warning;
#            b. the mean of the five parsed per-compound values equals the mean the log itself printed, within
#               0.001 (the per-compound values are printed to three decimals, so their mean can differ from the
#               printed mean by rounding only).
#          The parsed values are written to figures/fig1e_values_v2.csv (feeds Supplementary Table S3).
# Output:  figures/fig1e_reproduction_R_v2.png (600 dpi) and .pdf (vector), and figures/fig1e_values_v2.csv,
#          all relative to the directory the script is run from (the Squidiff_ms worktree root).
# Run:     conda activate /scratch/ni2071/conda_envs/R_env ; Rscript plot_fig1e_v2.R

suppressPackageStartupMessages(library(ggplot2))

# ---- inputs: the two corrected evaluation logs (outside the repository, copied into the package) ----
LOGS <- c("200"  = "/scratch/ni2071/Diffusion/gbm_run_200/logs/eval_200_v2_18927698.out",
          "2000" = "/scratch/ni2071/Diffusion/gbm_run/logs/eval_v2_18927697.out")
for (f in LOGS) if (!file.exists(f)) stop(sprintf("missing log: %s", f))
PUBLISHED <- c("Tumour cells" = 0.912, "Oligodendrocytes" = 0.915)   # He et al. 2026, glioblastoma analysis

# ---- colours and labels (identical to plot_fig1e.R) ----
INK <- "#0b0b0b"; INK2 <- "#52514e"; MUTED <- "#898781"; GRID <- "#e1e0d9"; AXIS <- "#c3c2b7"; SURFACE <- "#fcfcfb"
type_lab <- c(tumor = "Tumour cells", oligodendrocyte = "Oligodendrocytes")
type_col <- c("Tumour cells" = INK, "Oligodendrocytes" = MUTED)

# ---- parse the logs (identical parser to plot_fig1e.R) ----
# per-compound rows: cell type, compound name, then R as the first number with three decimals
# reconstruction rows: cell type, a colon, then "recon R = <value>"
parse_log <- function(path, n_genes) {
  x <- readLines(path)
  per <- regmatches(x, regexec("^\\s*(tumor|oligodendrocyte)\\s+(\\S+)\\s+(-?[0-9]\\.[0-9]{3})", x))
  per <- do.call(rbind, lapply(per[lengths(per) == 4], function(m)
    data.frame(n_genes = n_genes, cell_type = type_lab[m[2]], compound = m[3], r = as.numeric(m[4]))))
  rec <- regmatches(x, regexec("^\\s*(tumor|oligodendrocyte)\\s*:\\s*recon R = (-?[0-9.]+)", x))
  rec <- do.call(rbind, lapply(rec[lengths(rec) == 3], function(m)
    data.frame(n_genes = n_genes, cell_type = type_lab[m[2]], compound = "vehicle reconstruction", r = as.numeric(m[3]))))
  stopifnot(nrow(per) == 10, nrow(rec) == 2)      # five compounds x two cell types, one reconstruction per type
  rbind(per, rec)
}

# ---- the log's own summary means, used only for check b ----
# summary rows: cell type, a colon, then "R = <value>" (the "recon R" rows do not match: "recon " precedes "R")
parse_summary <- function(path, n_genes) {
  x <- readLines(path)
  s <- regmatches(x, regexec("^\\s*(tumor|oligodendrocyte)\\s*:\\s*R = (-?[0-9.]+)", x))
  s <- do.call(rbind, lapply(s[lengths(s) == 3], function(m)
    data.frame(n_genes = n_genes, cell_type = unname(type_lab[m[2]]), r_log = as.numeric(m[3]))))
  if (is.null(s) || nrow(s) != 2) stop(sprintf("expected two summary rows (one per cell type) in %s", path))
  s
}

d <- rbind(parse_log(LOGS["200"], 200), parse_log(LOGS["2000"], 2000))

# ---- check a: every value inside the fixed y axis, so no point is dropped silently ----
if (any(d$r < 0 | d$r > 1)) {
  print(d[d$r < 0 | d$r > 1, ], row.names = FALSE)
  stop("a parsed correlation lies outside [0, 1]; the fixed y axis would drop it")
}

# ---- check b: mean of the five parsed compounds == the mean printed in the log (rounding only) ----
pm <- aggregate(r ~ n_genes + cell_type, data = d[d$compound != "vehicle reconstruction", ], FUN = mean)
sm <- rbind(parse_summary(LOGS["200"], 200), parse_summary(LOGS["2000"], 2000))
chk <- merge(pm, sm, by = c("n_genes", "cell_type"))
stopifnot(nrow(chk) == 4)                         # two sizes x two cell types
chk$diff <- abs(chk$r - chk$r_log)
print(chk[order(chk$cell_type, chk$n_genes), ], row.names = FALSE)
if (any(chk$diff > 0.001)) stop("mean of the parsed compounds differs from the log's own mean by more than rounding")
cat("check b passed: parsed per-compound means equal the logs' own means within 0.001\n")

# ---- factors and panels for drawing (identical to plot_fig1e.R) ----
d$cell_type <- factor(d$cell_type, levels = names(type_col))
d$size_lab  <- factor(ifelse(d$n_genes == 200, "200 genes", "2,000 genes"), levels = c("200 genes", "2,000 genes"))
d$panel     <- factor(ifelse(d$compound == "vehicle reconstruction", "Vehicle reconstruction",
                             "Prediction of held-out compounds"),
                      levels = c("Prediction of held-out compounds", "Vehicle reconstruction"))

dir.create("figures", showWarnings = FALSE)
write.csv(d[, c("panel", "n_genes", "cell_type", "compound", "r")], "figures/fig1e_values_v2.csv", row.names = FALSE)

# ---- positions: cell type on the x axis, panel size as a small offset, compounds spread within it ----
d$x0 <- as.numeric(d$cell_type) + ifelse(d$n_genes == 200, -0.2, 0.2)
pred <- d[d$panel == "Prediction of held-out compounds", ]
pred$x <- pred$x0 + ave(seq_len(nrow(pred)), pred$cell_type, pred$n_genes,
                        FUN = function(i) seq(-0.12, 0.12, length.out = length(i)))
rec  <- d[d$panel == "Vehicle reconstruction", ]; rec$x <- rec$x0
means <- aggregate(r ~ panel + cell_type + n_genes + x0 + size_lab, data = pred, FUN = mean)
pub   <- data.frame(panel = factor("Prediction of held-out compounds", levels = levels(d$panel)),
                    cell_type = factor(names(PUBLISHED), levels = names(type_col)),
                    x = as.numeric(factor(names(PUBLISHED), levels = names(type_col))), r = unname(PUBLISHED))
pts   <- rbind(pred[, c("panel", "cell_type", "size_lab", "x", "r")], rec[, c("panel", "cell_type", "size_lab", "x", "r")])
pts$fill <- ifelse(pts$size_lab == "200 genes", type_col[as.character(pts$cell_type)], SURFACE)

# ---- the figure (identical to plot_fig1e.R except the wording of the line under the plot) ----
p <- ggplot() +
  # published value: a short dashed bar across the cell type's two groups
  geom_segment(data = pub, aes(x = x - 0.38, xend = x + 0.38, y = r, yend = r),
               linetype = "22", colour = INK2, linewidth = 0.45) +
  # mean over the five compounds: a solid bar under the points, wider than the point cloud
  geom_segment(data = means, aes(x = x0 - 0.2, xend = x0 + 0.2, y = r, yend = r, colour = cell_type),
               linewidth = 1.0) +
  geom_point(data = pts, aes(x = x, y = r, colour = cell_type, shape = cell_type, fill = fill),
             size = 2.3, stroke = 0.6) +
  facet_wrap(~ panel, nrow = 1) +
  scale_colour_manual(values = type_col, guide = "none") +
  scale_shape_manual(values = c("Tumour cells" = 21, "Oligodendrocytes" = 24), guide = "none") +
  scale_fill_identity() +
  scale_x_continuous(breaks = 1:2, labels = names(type_col), limits = c(0.45, 2.55), expand = c(0, 0)) +
  # limits stay [0, 1] (check a guarantees no value is dropped); 3 % headroom above 1 so that points near 1
  # are drawn whole (change 5); nothing is added below 0, so the axis still starts at 0
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25), expand = expansion(mult = c(0, 0.03))) +
  labs(x = NULL, y = "Pearson r",
       caption = sprintf("Filled points: 200-gene panel; hollow points: 2,000-gene panel. Left: one point per held-out\ncompound (five), solid bar = their mean, dashed bar = published value (%.2f tumour cells,\n%.2f oligodendrocytes). Right: correlation between the mean expression of reconstructed\nand of measured vehicle cells.",
                         PUBLISHED["Tumour cells"], PUBLISHED["Oligodendrocytes"])) +
  theme_minimal(base_size = 9) +
  theme(
    plot.background    = element_rect(fill = SURFACE, colour = NA),
    panel.background   = element_rect(fill = SURFACE, colour = NA),
    panel.grid.minor   = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(colour = GRID, linewidth = 0.3),
    axis.line.x        = element_line(colour = AXIS, linewidth = 0.4),
    axis.text.x        = element_text(colour = INK, size = 7.5),
    axis.text.y        = element_text(colour = INK2, size = 7),
    axis.title.y       = element_text(colour = INK2, size = 8),
    strip.text         = element_text(hjust = 0, colour = INK, size = 8.5, margin = margin(b = 3)),
    plot.caption       = element_text(hjust = 0, size = 6.5, colour = INK2, lineheight = 1.05, margin = margin(t = 6)),
    panel.spacing.x    = unit(16, "pt"),
    plot.margin        = margin(6, 10, 4, 6)
  )

# the caption gained one line, so the figure is 0.1 in taller than plot_fig1e.R's; the panels keep their size
ggsave("figures/fig1e_reproduction_R_v2.png", p, width = 5.2, height = 3.1, units = "in", dpi = 600, bg = SURFACE)
ggsave("figures/fig1e_reproduction_R_v2.pdf", p, width = 5.2, height = 3.1, units = "in", bg = SURFACE)

# ---- values for checking against Section 1 (manuscript: 0.84 tumour cells, 0.93 oligodendrocytes at 200 genes;
#      vehicle reconstruction 0.43-0.51 at 2,000 genes) ----
print(means[order(means$cell_type, means$n_genes), c("cell_type", "n_genes", "r")], row.names = FALSE)
print(rec[order(rec$cell_type, rec$n_genes), c("cell_type", "n_genes", "r")], row.names = FALSE)
cat("written: figures/fig1e_reproduction_R_v2.png, .pdf and figures/fig1e_values_v2.csv\n")
