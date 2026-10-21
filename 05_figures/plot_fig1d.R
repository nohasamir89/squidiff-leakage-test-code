# plot_fig1d.R - Fig. 1d: gene-panel sweep of the original model on the glioblastoma screen.
#
# Purpose: draw, for ten panel sizes (100 to 1,000 genes), (top) how well the model reconstructs vehicle
#          cells and (bottom) its margin over the vehicle-mean reference on the responsive genes.
# Logic:   reads gene_sensitivity_SEEDED.csv, the seeded re-evaluation of the ten checkpoints (decoder seed 42,
#          run_gbm_seeded_reeval.sh, 2 July 2026); one row per size and cell type. Columns used:
#            n_hvg            panel size (number of highly variable genes)
#            cell_type        tumor / oligodendrocyte
#            recon_R_vehicle  Pearson r between reconstructed and measured vehicle cells
#            recon_ok         1 if recon_R_vehicle >= 0.70 (the pre-set floor) in BOTH cell types at that size,
#                             else 0 (a per-size flag written to both rows; see eval_gbm_sensitivity.py recon_ok)
#            BminusC_R_deg    Pearson r of the prediction (B) minus that of the vehicle mean (C), responsive genes
#            frac_zero        fraction of zero entries in the panel (printed, not drawn)
#          A size is 'interpreted' only if recon_ok == 1 in BOTH cell types; such points are filled, others
#          hollow. No text is drawn inside the panels, so nothing can overlap the data; the dashed lines and
#          the hollow points are explained in the caption line under the plot.
#          Nothing is typed in; the per-size table is printed for checking against Section 1.
# Output:  figures/fig1d_gene_panel_sweep_R.png (600 dpi) and .pdf (vector), in the Squidiff repo.

suppressPackageStartupMessages(library(ggplot2))

# Release fix (21 Oct 2026): the record ships inside the package; the original cluster path is the fallback.
CSV <- "08_external_pipelines/gbm_squidiff_robustness_test/gene_sensitivity_SEEDED.csv"
if (!file.exists(CSV)) CSV <- "/scratch/ni2071/Diffusion/gbm_squidiff_robustness_test/gene_sensitivity_SEEDED.csv"
d <- read.csv(CSV, stringsAsFactors = FALSE)
d <- d[order(d$cell_type, d$n_hvg), ]
stopifnot(all(c("n_hvg", "cell_type", "recon_R_vehicle", "recon_ok", "BminusC_R_deg", "frac_zero") %in% names(d)))
stopifnot(!any(duplicated(d[, c("n_hvg", "cell_type")])))   # one evaluation per size and cell type

INK <- "#0b0b0b"; INK2 <- "#52514e"; MUTED <- "#898781"; GRID <- "#e1e0d9"; AXIS <- "#c3c2b7"; SURFACE <- "#fcfcfb"
FLOOR <- 0.70                                            # the reconstruction floor used in the study

# a size counts only if both cell types pass the floor
ok_by_size <- tapply(d$recon_ok == 1, d$n_hvg, all)
d$interpreted <- as.logical(ok_by_size[as.character(d$n_hvg)])
d$cell_type   <- factor(d$cell_type, levels = c("tumor", "oligodendrocyte"),
                        labels = c("Tumour cells", "Oligodendrocytes"))
type_col      <- c("Tumour cells" = INK, "Oligodendrocytes" = MUTED)
d$fill        <- ifelse(d$interpreted, type_col[as.character(d$cell_type)], SURFACE)

P1 <- "Vehicle reconstruction (Pearson r)"
P2 <- "Margin over the vehicle mean,\nresponsive genes (Pearson r)"
long <- rbind(
  data.frame(panel = P1, n_hvg = d$n_hvg, cell_type = d$cell_type, value = d$recon_R_vehicle, fill = d$fill),
  data.frame(panel = P2, n_hvg = d$n_hvg, cell_type = d$cell_type, value = d$BminusC_R_deg,   fill = d$fill)
)
long$panel <- factor(long$panel, levels = c(P1, P2))
hlines <- data.frame(panel = factor(c(P1, P2), levels = c(P1, P2)), y = c(FLOOR, 0))
# the line passes through the mean at each size; every evaluation is still drawn as its own point
means <- aggregate(value ~ panel + n_hvg + cell_type, data = long, FUN = mean)

p <- ggplot(long, aes(x = n_hvg, y = value, colour = cell_type, shape = cell_type)) +
  geom_hline(data = hlines, aes(yintercept = y), linetype = "22", colour = INK2, linewidth = 0.4) +
  geom_line(data = means, aes(x = n_hvg, y = value, colour = cell_type), inherit.aes = FALSE,
            linewidth = 0.45, alpha = 0.9) +
  geom_point(aes(fill = fill), size = 2.4, stroke = 0.6) +
  facet_wrap(~ panel, ncol = 1, scales = "free_y") +
  scale_colour_manual(values = type_col, name = NULL) +
  scale_shape_manual(values = c("Tumour cells" = 21, "Oligodendrocytes" = 24), name = NULL) +
  scale_fill_identity() +
  scale_x_continuous(breaks = seq(100, 1000, 100),
                     labels = function(x) ifelse(x %% 200 == 0 | x == 100, format(x, big.mark = ","), ""),
                     expand = expansion(mult = 0.04)) +
  scale_y_continuous(expand = expansion(mult = 0.12)) +
  labs(x = "Genes in the panel", y = NULL,
       caption = sprintf("Dashed lines: the reconstruction floor (%.2f) and zero margin.\nHollow points: sizes at which reconstruction fell below the floor\nin at least one cell type.", FLOOR)) +
  theme_minimal(base_size = 9) +
  theme(
    plot.background    = element_rect(fill = SURFACE, colour = NA),
    panel.background   = element_rect(fill = SURFACE, colour = NA),
    panel.grid.minor   = element_blank(),
    panel.grid.major   = element_line(colour = GRID, linewidth = 0.3),
    axis.line.x        = element_line(colour = AXIS, linewidth = 0.4),
    axis.ticks.x       = element_line(colour = AXIS, linewidth = 0.3),
    axis.text          = element_text(colour = INK2, size = 7),
    axis.title.x       = element_text(colour = INK2, size = 8, margin = margin(t = 4)),
    strip.text         = element_text(hjust = 0, colour = INK, size = 8.5, lineheight = 0.95, margin = margin(b = 3)),
    legend.position    = "top",
    legend.justification = "left",
    legend.text        = element_text(size = 7.5, colour = INK),
    legend.key.size    = unit(9, "pt"),
    legend.margin      = margin(0, 0, 2, 0),
    plot.caption       = element_text(hjust = 0, size = 6.5, colour = INK2, lineheight = 1.05,
                                      margin = margin(t = 6)),
    panel.spacing.y    = unit(12, "pt"),
    plot.margin        = margin(6, 10, 4, 6)
  ) +
  guides(colour = guide_legend(override.aes = list(fill = unname(type_col))))

dir.create("figures", showWarnings = FALSE)
ggsave("figures/fig1d_gene_panel_sweep_R.png", p, width = 4.2, height = 4.6, units = "in", dpi = 600, bg = SURFACE)
ggsave("figures/fig1d_gene_panel_sweep_R.pdf", p, width = 4.2, height = 4.6, units = "in", bg = SURFACE)

# ---- per-size table for checking against Section 1 ----
tum <- d[d$cell_type == "Tumour cells",     c("n_hvg", "interpreted", "recon_R_vehicle", "BminusC_R_deg", "frac_zero")]
oli <- d[d$cell_type == "Oligodendrocytes", c("n_hvg", "recon_R_vehicle", "BminusC_R_deg")]
wide <- merge(tum, oli, by = "n_hvg", suffixes = c("_tumour", "_oligo"))
wide <- wide[order(wide$n_hvg), c("n_hvg", "interpreted", "recon_R_vehicle_tumour", "recon_R_vehicle_oligo",
                                  "BminusC_R_deg_tumour", "BminusC_R_deg_oligo", "frac_zero")]
print(format(wide, digits = 3), row.names = FALSE)
cat(sprintf("interpreted sizes: %s\n", paste(wide$n_hvg[wide$interpreted], collapse = ", ")))
cat(sprintf("tumour margin over the vehicle mean at interpreted sizes: %.2f to %.2f\n",
            min(wide$BminusC_R_deg_tumour[wide$interpreted]), max(wide$BminusC_R_deg_tumour[wide$interpreted])))
cat("written: figures/fig1d_gene_panel_sweep_R.png and .pdf\n")
