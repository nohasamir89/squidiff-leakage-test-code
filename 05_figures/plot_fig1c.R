# plot_fig1c.R - Fig. 1c: why the original model's populations are far from the measured cells.
#
# Purpose: draw the per-condition breakdown of the population distance for the original model on one split
#          (Split 1, seed 0), from tahoe_results/step15_diag_cells_drug_split_1_seed_0_per_condition.csv.
# Logic:   each row of the CSV is one held-out condition; the 'base' rows are the original model. For every
#          condition the diagnostic compared the generated population with the measured treated cells:
#            xx / yy               mean pairwise distance within the generated / measured population (spread)
#            lib_pred / lib_true   mean total expression per cell (library size, log-normalised units)
#            zero_pred / zero_true fraction of zero entries
#            d_true_ctrl           distance from the measured treated mean to the vehicle mean (the response)
#            d_mean_pred_true      distance from the generated mean to the measured treated mean
#            d_mean_pred_ctrl      distance from the generated mean to the vehicle mean
#          Each panel shows the distribution over conditions (violin) with the quartiles (box) and a ratio of
#          means in the corner, all computed from the CSV. Nothing is typed in.
# Output:  figures/fig1c_population_breakdown_R.png (300 dpi) and .pdf (vector).

suppressPackageStartupMessages(library(ggplot2))

CSV <- "tahoe_results/step15_diag_cells_drug_split_1_seed_0_per_condition.csv"
d <- read.csv(CSV, stringsAsFactors = FALSE)
d <- d[d$arm == "base", ]                       # the original model only
stopifnot(nrow(d) > 0)

INK <- "#0b0b0b"; INK2 <- "#52514e"; MUTED <- "#898781"; GRID <- "#e1e0d9"; AXIS <- "#c3c2b7"; SURFACE <- "#fcfcfb"
BLUE <- "#2a78d6"                                 # Split 1 colour: this panel is Split 1 only
GREY <- "#b8b6ae"                                 # measured cells

# ---- panels and groups (labels broken over two lines so nothing overlaps) ----
P1 <- "Spread within\npopulation"; P2 <- "Expression\nper cell"; P3 <- "Zero entries\n(fraction)"; P4 <- "Distance\nbetween means"
G_M <- "Measured"; G_G <- "Generated"
G_TV <- "Treated to\nvehicle"; G_GT <- "Generated\nto treated"; G_GV <- "Generated\nto vehicle"

mk <- function(panel, group, values) data.frame(panel = panel, group = group, value = values)
long <- rbind(
  mk(P1, G_M, d$yy),        mk(P1, G_G, d$xx),
  mk(P2, G_M, d$lib_true),  mk(P2, G_G, d$lib_pred),
  mk(P3, G_M, d$zero_true), mk(P3, G_G, d$zero_pred),
  mk(P4, G_TV, d$d_true_ctrl), mk(P4, G_GT, d$d_mean_pred_true), mk(P4, G_GV, d$d_mean_pred_ctrl)
)
long$panel <- factor(long$panel, levels = c(P1, P2, P3, P4))
long$group <- factor(long$group, levels = c(G_M, G_G, G_TV, G_GT, G_GV))
fills <- setNames(c(GREY, BLUE, GREY, BLUE, BLUE), c(G_M, G_G, G_TV, G_GT, G_GV))

# ---- ratios of means, one label per panel, computed from the data ----
m <- colMeans(d[, c("xx", "yy", "lib_pred", "lib_true", "zero_pred", "zero_true",
                    "d_mean_pred_true", "d_mean_pred_ctrl", "d_true_ctrl")])
labs_df <- data.frame(
  panel = factor(c(P1, P2, P3, P4), levels = levels(long$panel)),
  text  = c(sprintf("generated / measured\n= %.1f×", m["xx"] / m["yy"]),
            sprintf("generated / measured\n= %.1f×", m["lib_pred"] / m["lib_true"]),
            sprintf("%.0f%% vs %.0f%% zeros", 100 * m["zero_pred"], 100 * m["zero_true"]),
            sprintf("generated mean displaced\n%.1f× the measured response", m["d_mean_pred_true"] / m["d_true_ctrl"]))
)

p <- ggplot(long, aes(x = group, y = value, fill = group)) +
  geom_violin(colour = NA, width = 0.85, alpha = 0.95, scale = "width", trim = TRUE) +
  geom_boxplot(width = 0.14, outlier.shape = NA, fill = SURFACE, colour = INK2, linewidth = 0.3, fatten = 1.5) +
  geom_text(data = labs_df, aes(x = -Inf, y = Inf, label = text), inherit.aes = FALSE,
            hjust = -0.05, vjust = 1.3, size = 2.4, colour = INK2, lineheight = 0.9) +
  facet_wrap(~ panel, scales = "free", nrow = 1) +
  scale_fill_manual(values = fills, guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0.03, 0.30)),
                     labels = function(x) trimws(format(x, drop0trailing = TRUE))) +
  labs(x = NULL, y = NULL) +
  theme_minimal(base_size = 9) +
  theme(
    plot.background    = element_rect(fill = SURFACE, colour = NA),
    panel.background   = element_rect(fill = SURFACE, colour = NA),
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_line(colour = GRID, linewidth = 0.3),
    axis.line.x        = element_line(colour = AXIS, linewidth = 0.4),
    axis.text.x        = element_text(colour = INK, size = 6.4, lineheight = 0.9),
    axis.text.y        = element_text(colour = INK2, size = 7),
    strip.text         = element_text(hjust = 0, colour = INK, size = 8.5, lineheight = 0.95,
                                      margin = margin(b = 4)),
    panel.spacing.x    = unit(14, "pt"),
    plot.margin        = margin(6, 10, 4, 6)
  )

dir.create("figures", showWarnings = FALSE)
# PNG at 600 dpi for viewing and the doc; the PDF is vector and is the file for assembly and submission
ggsave("figures/fig1c_population_breakdown_R.png", p, width = 8.0, height = 3.0, units = "in", dpi = 600, bg = SURFACE)
ggsave("figures/fig1c_population_breakdown_R.pdf", p, width = 8.0, height = 3.0, units = "in", bg = SURFACE)

# ---- values for checking against the text ----
cat(sprintf("conditions: %d (arm = base, %s)\n", nrow(d), CSV))
cat(sprintf("spread  generated %.3f  measured %.3f  ratio %.2f\n", m["xx"], m["yy"], m["xx"] / m["yy"]))
cat(sprintf("library generated %.1f  measured %.1f  ratio %.2f\n", m["lib_pred"], m["lib_true"], m["lib_pred"] / m["lib_true"]))
cat(sprintf("zeros   generated %.3f  measured %.3f\n", m["zero_pred"], m["zero_true"]))
cat(sprintf("distance of generated mean: to treated mean %.3f, to vehicle mean %.3f; measured response %.3f; ratio %.2f\n",
            m["d_mean_pred_true"], m["d_mean_pred_ctrl"], m["d_true_ctrl"], m["d_mean_pred_true"] / m["d_true_ctrl"]))
cat("written: figures/fig1c_population_breakdown_R.png and .pdf\n")
