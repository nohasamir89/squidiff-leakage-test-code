# plot_fig1e.R - Fig. 1e: reproduction of the published glioblastoma analysis at 200 genes, and its failure at 2,000.
#
# Purpose: show, per cell type, the Pearson correlation between predicted and measured response for each of the
#          five held-out compounds, at 200 genes (reproduces the published values) and at 2,000 genes (does not),
#          together with the vehicle reconstruction at each panel size.
# Logic:   the evaluation scripts wrote no table, so the two evaluation logs are the record. They are parsed here:
#            per-compound rows   "  tumor           panobinostat     0.800   0.574    5114"   (R is the first number)
#            reconstruction rows "  tumor           : recon R = 0.982  r2 = 0.940  (n_enc=300)"
#          Published values are the two numbers reported in He et al. (Squidiff, Nat. Methods 2026) for this
#          analysis: 0.912 (tumour cells) and 0.915 (oligodendrocytes); they are the only typed-in numbers.
#          The parsed values are written to figures/fig1e_values.csv (feeds Supplementary Table S3).
# Output:  figures/fig1e_reproduction_R.png (600 dpi) and .pdf (vector), in the Squidiff repo.

suppressPackageStartupMessages(library(ggplot2))

# Release fix (21 Oct 2026): the two SLURM logs ship inside the package; the cluster paths are the fallback.
LOGS <- c("200"  = "08_external_pipelines/gbm_run_200/logs/eval_200_16378137.out",
          "2000" = "08_external_pipelines/gbm_run/logs/eval_16377417.out")
if (!all(file.exists(LOGS))) LOGS <- c("200"  = "/scratch/ni2071/Diffusion/gbm_run_200/logs/eval_200_16378137.out",
                                      "2000" = "/scratch/ni2071/Diffusion/gbm_run/logs/eval_16377417.out")
PUBLISHED <- c("Tumour cells" = 0.912, "Oligodendrocytes" = 0.915)   # He et al. 2026, glioblastoma analysis

INK <- "#0b0b0b"; INK2 <- "#52514e"; MUTED <- "#898781"; GRID <- "#e1e0d9"; AXIS <- "#c3c2b7"; SURFACE <- "#fcfcfb"
type_lab <- c(tumor = "Tumour cells", oligodendrocyte = "Oligodendrocytes")
type_col <- c("Tumour cells" = INK, "Oligodendrocytes" = MUTED)

# ---- parse the logs ----
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
d <- rbind(parse_log(LOGS["200"], 200), parse_log(LOGS["2000"], 2000))
d$cell_type <- factor(d$cell_type, levels = names(type_col))
d$size_lab  <- factor(ifelse(d$n_genes == 200, "200 genes", "2,000 genes"), levels = c("200 genes", "2,000 genes"))
d$panel     <- factor(ifelse(d$compound == "vehicle reconstruction", "Vehicle reconstruction",
                             "Prediction of held-out compounds"),
                      levels = c("Prediction of held-out compounds", "Vehicle reconstruction"))

dir.create("figures", showWarnings = FALSE)
write.csv(d[, c("panel", "n_genes", "cell_type", "compound", "r")], "figures/fig1e_values.csv", row.names = FALSE)

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
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25), expand = c(0, 0)) +
  labs(x = NULL, y = "Pearson r",
       caption = sprintf("Filled points: 200-gene panel; hollow points: 2,000-gene panel. Left: one point per held-out\ncompound (five), solid bar = their mean, dashed bar = published value (%.2f tumour cells,\n%.2f oligodendrocytes). Right: correlation between reconstructed and measured vehicle cells.",
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

ggsave("figures/fig1e_reproduction_R.png", p, width = 5.2, height = 3.0, units = "in", dpi = 600, bg = SURFACE)
ggsave("figures/fig1e_reproduction_R.pdf", p, width = 5.2, height = 3.0, units = "in", bg = SURFACE)

# ---- values for checking against Section 1 ----
print(means[order(means$cell_type, means$n_genes), c("cell_type", "n_genes", "r")], row.names = FALSE)
print(rec[order(rec$cell_type, rec$n_genes), c("cell_type", "n_genes", "r")], row.names = FALSE)
cat("written: figures/fig1e_reproduction_R.png, .pdf and figures/fig1e_values.csv\n")
