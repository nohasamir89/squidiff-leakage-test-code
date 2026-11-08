# plot_fig4.R - Fig. 4b-e: the rewired model and optimal-transport pairing on Tahoe-100M (ggplot2).
#
# Purpose: (b) original model against rewired model on the three measures, per split;
#          (c) exact optimal-transport pairing against random pairing (step 18): population distance, gene
#              identity, the discrimination score and cell tracking, with the post hoc comparison against the
#              unaltered vehicle cells;
#          (d) the matching-strength sweep on the development split (step 20): each setting against random pairing,
#              as a function of how far the pairs sit between random and exact matching;
#          (e) minibatch matching (m = 32) against exact matching on the confirmation splits (step 21), with the
#              non-inferiority margin on population distance.
# Logic:   every value is read from the committed records; nothing is typed in.
#            tahoe_results/step15_model_long.csv            (b) pooled values, line == "ALL": base and recipe
#            tahoe_results/step15_model_analysis.json       (b) contrasts recipe minus base, printed for checking
#            tahoe_results/step18_analysis.json             (c) contrasts ot_pair minus recipe, all three splits
#            tahoe_results/posthoc_ot_energy_vs_references.json  (c) ot_pair minus control_paste, post hoc
#            tahoe_results/step20_analysis_development.json (d) contrasts per setting (mb4, mb8, mb32) minus recipe,
#                                                               and the cosine fraction f of each setting
#            tahoe_results/step21_analysis.json             (e) contrasts mb32 minus ot_pair with the margin delta,
#                                                               confirmation splits; development split from its
#                                                               'development_descriptive' entry
#          Filled point: interval excludes zero; hollow: includes zero. Intervals are the records' 97.5 %
#          structure-clustered percentile bootstrap intervals (post hoc comparison: same construction).
# Where:   lives in <repo>/figures/squidiff_publication_figures/ and writes there:
#          fig4b_rewired_vs_original_R, fig4c_ot_pairing_R, fig4d_matching_sweep_R, fig4e_mb32_confirmation_R
#          (.png 600 dpi and .pdf), and fig4_values.csv.   Run: Rscript figures/squidiff_publication_figures/plot_fig4.R
# Needs:   ggplot2 and jsonlite.

suppressPackageStartupMessages(library(ggplot2))
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("package 'jsonlite' is needed")

script_arg <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
REPO <- if (length(script_arg) == 1) normalizePath(file.path(dirname(script_arg), "..", "..")) else getwd()
OUT  <- file.path(REPO, "figures", "squidiff_publication_figures")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
TR <- function(name) file.path(REPO, "tahoe_results", name)
for (f in c("step15_model_long.csv", "step15_model_analysis.json", "step18_analysis.json",
            "posthoc_ot_energy_vs_references.json", "step20_analysis_development.json", "step21_analysis.json"))
  if (!file.exists(TR(f))) stop("input file not found: ", TR(f))

read_json_py <- function(path) {
  txt <- paste(readLines(path, warn = FALSE), collapse = "\n")
  txt <- gsub("(?<![A-Za-z0-9_\"])-?(NaN|Infinity)(?![A-Za-z0-9_\"])", "null", txt, perl = TRUE)
  jsonlite::fromJSON(txt, simplifyVector = FALSE)
}
INK <- "#0b0b0b"; INK2 <- "#52514e"; MUTED <- "#898781"; GRID <- "#e1e0d9"; AXIS <- "#c3c2b7"; SURFACE <- "#fcfcfb"
cols      <- c("Split 1" = "#2a78d6", "Split 0" = "#eb6834", "Split 2" = "#1baf7a")
split_map <- c(drug_split_1 = "Split 1", drug_split_0 = "Split 0", drug_split_2 = "Split 2")
SPLITS    <- c("drug_split_1", "drug_split_0", "drug_split_2")
fmt_tick  <- function(x) sub("\\.?0+$", "", sprintf("%.3f", x))

base_theme <- theme_minimal(base_size = 9) +
  theme(
    plot.background    = element_rect(fill = SURFACE, colour = NA),
    panel.background   = element_rect(fill = SURFACE, colour = NA),
    panel.grid.minor   = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(colour = GRID, linewidth = 0.3),
    axis.line.x        = element_line(colour = AXIS, linewidth = 0.4),
    axis.ticks.x       = element_line(colour = AXIS, linewidth = 0.3),
    axis.text          = element_text(colour = INK2, size = 7),
    axis.text.x        = element_text(colour = INK, size = 7.5),
    axis.title         = element_text(colour = INK2, size = 8),
    strip.text         = element_text(hjust = 0, colour = INK, size = 8, lineheight = 0.95, margin = margin(b = 3)),
    legend.position    = "bottom",
    legend.text        = element_text(size = 7, colour = INK),
    legend.key.size    = unit(8, "pt"),
    plot.caption       = element_text(hjust = 0, size = 6.5, colour = INK2, lineheight = 1.05, margin = margin(t = 6)),
    plot.caption.position = "plot",
    panel.spacing.x    = unit(14, "pt"),
    plot.margin        = margin(6, 10, 4, 6)
  )

# a contrast entry {diff, ci} -> one row
rows <- list()
add <- function(panel, facet, x, split, point, lo, hi, extra = NA, source = "") {
  stopifnot(is.finite(point), is.finite(lo), is.finite(hi), lo <= point, point <= hi)
  rows[[length(rows) + 1]] <<- data.frame(panel = panel, facet = facet, x = x, split = split, point = point,
                                          lo = lo, hi = hi, extra = extra, source = source, stringsAsFactors = FALSE)
}
ctr <- function(lst, key) { x <- lst[[key]]; if (is.null(x)) stop("missing contrast ", key); x }

# ================= Panel b: original vs rewired, three measures =================
long <- read.csv(TR("step15_model_long.csv"), stringsAsFactors = FALSE)
long <- long[long$line == "ALL" & long$submission %in% c("base", "recipe") &
             long$metric %in% c("de_overlap", "dpc_mean", "energy"), ]
stopifnot(nrow(long) == 2 * 3 * 3)
B_FACETS <- c(de_overlap = "Gene identity\n(Jaccard, top-50 genes)", dpc_mean = "Response correlation\n(Pearson r, top-50 genes)",
              energy = "Population distance\n(energy; lower is better)")
b <- data.frame(facet = factor(B_FACETS[long$metric], levels = B_FACETS),
                model = factor(ifelse(long$submission == "base", "Original\nmodel", "Rewired\nmodel"),
                               levels = c("Original\nmodel", "Rewired\nmodel")),
                split = factor(split_map[long$split], levels = names(cols)), value = long$value)
for (i in seq_len(nrow(b))) add("b", as.character(b$facet[i]), as.character(b$model[i]), as.character(b$split[i]),
                                 b$value[i], b$value[i], b$value[i], source = "step15_model_long.csv")
pb <- ggplot(b, aes(x = model, y = value, colour = split, group = split)) +
  geom_line(linewidth = 0.5, alpha = 0.9) +
  geom_point(size = 2.3) +
  facet_wrap(~ facet, scales = "free_y", nrow = 1) +
  scale_colour_manual(values = cols, name = NULL) +
  scale_x_discrete(expand = expansion(add = 0.6)) +
  scale_y_continuous(labels = fmt_tick, expand = expansion(mult = 0.15)) +
  labs(x = NULL, y = NULL) +
  base_theme + theme(axis.text.x = element_text(colour = INK, size = 7.5, lineheight = 0.9))
# the matching contrasts (recipe minus base) with intervals, printed for checking against the text
s15 <- read_json_py(TR("step15_model_analysis.json"))
cat("panel b: recipe minus base (97.5 % intervals)\n")
for (sp in SPLITS) {
  cc <- s15$per_split[[sp]]$contrasts
  for (k in c("Q1_primary_de_recipe_minus_base", "secondary_dpc_recipe_minus_base", "secondary_en_recipe_minus_base")) {
    x <- ctr(cc, k); cat(sprintf("  %-8s %-36s %+.4f [%+.4f, %+.4f] %s\n", split_map[[sp]], k, x$diff, x$ci[[1]], x$ci[[2]], x$verdict))
  }
}

# ================= Panel c: exact optimal-transport pairing against random pairing (step 18) =================
s18 <- read_json_py(TR("step18_analysis.json"))
ph  <- read_json_py(TR("posthoc_ot_energy_vs_references.json"))
C_FACETS <- c(en = "Population distance\nvs random pairing", en_cp = "Population distance\nvs vehicle cells",
              de = "Gene identity", pds = "Discrimination\nscore", tr = "Cell tracking")
for (sp in SPLITS) {
  cc <- s18$per_split[[sp]]$contrasts
  x <- ctr(cc, "Q1_primary_energy_ot_pair_minus_recipe"); add("c", C_FACETS[["en"]],  "OT", split_map[[sp]], x$diff, x$ci[[1]], x$ci[[2]], source = "step18_analysis.json")
  x <- ctr(cc, "Q2_gate_de_ot_pair_minus_recipe");        add("c", C_FACETS[["de"]],  "OT", split_map[[sp]], x$diff, x$ci[[1]], x$ci[[2]], source = "step18_analysis.json")
  x <- ctr(cc, "Q3_gate_pds_ot_pair_minus_recipe");       add("c", C_FACETS[["pds"]], "OT", split_map[[sp]], x$diff, x$ci[[1]], x$ci[[2]], source = "step18_analysis.json")
  x <- ctr(cc, "exploratory_tracking_ot_pair_minus_recipe"); add("c", C_FACETS[["tr"]], "OT", split_map[[sp]], x$diff, x$ci[[1]], x$ci[[2]], source = "step18_analysis.json")
}
for (r in ph$results) if (r$contrast == "ot_pair - control_paste")
  add("c", C_FACETS[["en_cp"]], "OT", split_map[[r$split]], r$point, r$lo, r$hi, source = "posthoc_ot_energy_vs_references.json")

# ================= Panel d: matching-strength sweep on the development split (step 20) =================
s20 <- read_json_py(TR("step20_analysis_development.json"))
d20 <- s20$per_split$drug_split_1
D_FACETS <- c(en = "Population distance", de = "Gene identity", pds = "Discrimination\nscore", tr = "Cell tracking")
settings <- c("mb4", "mb8", "mb32")
set_lab  <- c(mb4 = "m = 4", mb8 = "m = 8", mb32 = "m = 32", ot_pair = "exact")
for (s in settings) {
  cc <- d20$contrasts[[s]]; f <- d20$f_cos[[s]]; stopifnot(is.finite(f))
  x <- ctr(cc, "Q1_energy_minus_recipe");       add("d", D_FACETS[["en"]],  set_lab[[s]], "Split 1", x$diff, x$ci[[1]], x$ci[[2]], extra = f, source = "step20_analysis_development.json")
  x <- ctr(cc, "Q2_de_minus_recipe");           add("d", D_FACETS[["de"]],  set_lab[[s]], "Split 1", x$diff, x$ci[[1]], x$ci[[2]], extra = f, source = "step20_analysis_development.json")
  x <- ctr(cc, "Q3_pds_minus_recipe");          add("d", D_FACETS[["pds"]], set_lab[[s]], "Split 1", x$diff, x$ci[[1]], x$ci[[2]], extra = f, source = "step20_analysis_development.json")
  x <- ctr(cc, "secondary_tracking_minus_recipe"); add("d", D_FACETS[["tr"]], set_lab[[s]], "Split 1", x$diff, x$ci[[1]], x$ci[[2]], extra = f, source = "step20_analysis_development.json")
}
# exact matching (f = 1): from the step 20 record if it carries an 'ot_pair' entry, else from step 18 (split 1)
if (!is.null(d20$contrasts$ot_pair)) {
  cc <- d20$contrasts$ot_pair; src <- "step20_analysis_development.json"
  k <- c(en = "Q1_energy_minus_recipe", de = "Q2_de_minus_recipe", pds = "Q3_pds_minus_recipe", tr = "secondary_tracking_minus_recipe")
} else {
  cc <- s18$per_split$drug_split_1$contrasts; src <- "step18_analysis.json"
  k <- c(en = "Q1_primary_energy_ot_pair_minus_recipe", de = "Q2_gate_de_ot_pair_minus_recipe",
         pds = "Q3_gate_pds_ot_pair_minus_recipe", tr = "exploratory_tracking_ot_pair_minus_recipe")
}
for (m in names(k)) { x <- ctr(cc, k[[m]]); add("d", D_FACETS[[m]], "exact", "Split 1", x$diff, x$ci[[1]], x$ci[[2]], extra = 1, source = src) }
cat("panel d: exact-matching point taken from", src, "\n")

# ================= Panel e: m = 32 against exact matching on the confirmation splits (step 21) =================
s21 <- read_json_py(TR("step21_analysis.json"))
E_FACETS <- c(en = "Population distance\n(dashed: margin)", pds = "Discrimination score", de = "Gene identity")
dev <- s21$development_descriptive
x <- ctr(dev$contrasts, "Q1_energy_mb32_minus_ot_pair");   add("e", E_FACETS[["en"]],  "mb32", "Split 1", x$diff, x$ci[[1]], x$ci[[2]], extra = dev$delta, source = "step21_analysis.json (development_descriptive)")
x <- ctr(dev$contrasts, "Q2_pds_mb32_minus_ot_pair");      add("e", E_FACETS[["pds"]], "mb32", "Split 1", x$diff, x$ci[[1]], x$ci[[2]], source = "step21_analysis.json (development_descriptive)")
x <- ctr(dev$contrasts, "secondary_de_mb32_minus_ot_pair"); add("e", E_FACETS[["de"]],  "mb32", "Split 1", x$diff, x$ci[[1]], x$ci[[2]], source = "step21_analysis.json (development_descriptive)")
for (sp in c("drug_split_0", "drug_split_2")) {
  ps <- s21$per_split[[sp]]; cc <- ps$contrasts
  x <- ctr(cc, "Q1_energy_mb32_minus_ot_pair");   add("e", E_FACETS[["en"]],  "mb32", split_map[[sp]], x$diff, x$ci[[1]], x$ci[[2]], extra = ps$delta, source = "step21_analysis.json")
  x <- ctr(cc, "Q2_pds_mb32_minus_ot_pair");      add("e", E_FACETS[["pds"]], "mb32", split_map[[sp]], x$diff, x$ci[[1]], x$ci[[2]], source = "step21_analysis.json")
  x <- ctr(cc, "secondary_de_mb32_minus_ot_pair"); add("e", E_FACETS[["de"]],  "mb32", split_map[[sp]], x$diff, x$ci[[1]], x$ci[[2]], source = "step21_analysis.json")
}

d <- do.call(rbind, rows)
d$excludes_zero <- d$lo > 0 | d$hi < 0
write.csv(d, file.path(OUT, "fig4_values.csv"), row.names = FALSE)

# ---- panel c plot ----
cc <- d[d$panel == "c", ]
cc$facet <- factor(cc$facet, levels = C_FACETS); cc$split <- factor(cc$split, levels = names(cols))
cc$fill  <- ifelse(cc$excludes_zero, cols[as.character(cc$split)], SURFACE)
pc <- ggplot(cc, aes(x = split, y = point, colour = split)) +
  geom_hline(yintercept = 0, linetype = "22", colour = INK2, linewidth = 0.4) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.18, linewidth = 0.5) +
  geom_point(aes(fill = fill), shape = 21, size = 2.3, stroke = 0.6) +
  scale_colour_manual(values = cols, guide = "none") + scale_fill_identity() +
  scale_x_discrete(expand = expansion(add = 0.7)) +
  facet_wrap(~ facet, scales = "free_y", nrow = 1) +
  scale_y_continuous(labels = fmt_tick) +
  labs(x = NULL, y = "Optimal-transport pairing\nminus comparator",
       caption = "Comparator: the rewired model trained with random pairing, except the second panel (unaltered vehicle cells, post hoc comparison).\nNegative population distance = closer to the measured cells. Cell tracking: cosine similarity between a generated cell and its vehicle cell.\nFilled: interval excludes zero; hollow: includes zero.") +
  base_theme + theme(legend.position = "none", axis.text.x = element_text(colour = INK, size = 7))

# ---- panel d plot ----
dd <- d[d$panel == "d", ]
dd$facet <- factor(dd$facet, levels = D_FACETS)
dd$fill  <- ifelse(dd$excludes_zero, cols[["Split 1"]], SURFACE)
# straight line from random pairing (0, 0) to exact matching (1, value): the proportional reference
ref <- dd[dd$x == "exact", c("facet", "point")]; ref$f0 <- 0; ref$y0 <- 0; ref$f1 <- 1
xlab_df <- unique(dd[, c("x", "extra")]); xlab_df <- xlab_df[order(xlab_df$extra), ]
pd <- ggplot(dd, aes(x = extra, y = point)) +
  geom_hline(yintercept = 0, linetype = "22", colour = INK2, linewidth = 0.4) +
  geom_segment(data = ref, aes(x = f0, y = y0, xend = f1, yend = point), colour = MUTED, linetype = "33", linewidth = 0.4) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.04, colour = cols[["Split 1"]], linewidth = 0.5) +
  geom_point(aes(fill = fill), shape = 21, colour = cols[["Split 1"]], size = 2.3, stroke = 0.6) +
  scale_fill_identity() +
  facet_wrap(~ facet, scales = "free_y", nrow = 1) +
  scale_x_continuous(breaks = c(0, 0.25, 0.5, 0.75, 1), labels = c("0", "0.25", "0.5", "0.75", "1"), limits = c(-0.05, 1.08)) +
  scale_y_continuous(labels = fmt_tick) +
  labs(x = "Matching strength f (random pairing = 0, exact matching = 1)",
       y = "Setting minus random pairing",
       caption = sprintf("Development split. Minibatch sizes m = 4, 8 and 32 sit at f = %s; the last point is exact matching (f = 1).\nDotted line: proportional change from random to exact matching. Filled: interval excludes zero; hollow: includes zero.",
                         paste(sprintf("%.2f", xlab_df$extra[xlab_df$x != "exact"]), collapse = ", "))) +
  base_theme + theme(axis.text.x = element_text(colour = INK, size = 7),
                     panel.grid.major.x = element_line(colour = GRID, linewidth = 0.3))

# ---- panel e plot ----
ee <- d[d$panel == "e", ]
ee$facet <- factor(ee$facet, levels = E_FACETS); ee$split <- factor(ee$split, levels = names(cols))
ee$fill  <- ifelse(ee$excludes_zero, cols[as.character(ee$split)], SURFACE)
marg <- ee[!is.na(ee$extra), ]                                       # the energy margin, one per split
pe <- ggplot(ee, aes(x = split, y = point, colour = split)) +
  geom_hline(yintercept = 0, linetype = "22", colour = INK2, linewidth = 0.4) +
  geom_segment(data = marg, aes(x = as.numeric(split) - 0.3, xend = as.numeric(split) + 0.3, y = extra, yend = extra),
               colour = INK2, linetype = "33", linewidth = 0.45, inherit.aes = FALSE) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.18, linewidth = 0.5) +
  geom_point(aes(fill = fill), shape = 21, size = 2.3, stroke = 0.6) +
  scale_colour_manual(values = cols, guide = "none") + scale_fill_identity() +
  scale_x_discrete(limits = names(cols)) +          # declared first: the margin bar uses numeric positions on this axis
  facet_wrap(~ facet, scales = "free_y", nrow = 1) +
  scale_y_continuous(labels = fmt_tick) +
  labs(x = NULL, y = "m = 32 minus exact matching",
       caption = "Split 1 is the development split on which m = 32 was chosen; Splits 0 and 2 are the confirmation splits. Population distance is\nnon-inferior when the whole interval lies below the margin (25 % of exact matching's gain over random pairing, fixed in advance).\nFilled: interval excludes zero; hollow: includes zero.") +
  base_theme + theme(legend.position = "none")

ggsave(file.path(OUT, "fig4b_rewired_vs_original_R.png"), pb, width = 7.0, height = 2.9, units = "in", dpi = 600, bg = SURFACE)
ggsave(file.path(OUT, "fig4b_rewired_vs_original_R.pdf"), pb, width = 7.0, height = 2.9, units = "in", bg = SURFACE)
ggsave(file.path(OUT, "fig4c_ot_pairing_R.png"),          pc, width = 9.0, height = 3.0, units = "in", dpi = 600, bg = SURFACE)
ggsave(file.path(OUT, "fig4c_ot_pairing_R.pdf"),          pc, width = 9.0, height = 3.0, units = "in", bg = SURFACE)
ggsave(file.path(OUT, "fig4d_matching_sweep_R.png"),      pd, width = 8.0, height = 2.9, units = "in", dpi = 600, bg = SURFACE)
ggsave(file.path(OUT, "fig4d_matching_sweep_R.pdf"),      pd, width = 8.0, height = 2.9, units = "in", bg = SURFACE)
ggsave(file.path(OUT, "fig4e_mb32_confirmation_R.png"),   pe, width = 6.6, height = 2.9, units = "in", dpi = 600, bg = SURFACE)
ggsave(file.path(OUT, "fig4e_mb32_confirmation_R.pdf"),   pe, width = 6.6, height = 2.9, units = "in", bg = SURFACE)

# ---- printed values ----
cat("panel b values\n"); print(b[order(b$facet, b$split, b$model), ], row.names = FALSE)
fmt <- function(x) sprintf("%+.4f [%+.4f, %+.4f] %s", x$point, x$lo, x$hi, ifelse(x$excludes_zero, "excludes zero", "includes zero"))
for (p in c("c", "d", "e")) {
  cat("panel", p, "\n"); z <- d[d$panel == p, ]
  for (i in seq_len(nrow(z))) cat(sprintf("  %-46s %-8s %-8s %s%s\n", gsub("\n", " ", z$facet[i]), z$x[i], z$split[i], fmt(z[i, ]),
                                          ifelse(is.na(z$extra[i]), "", sprintf("  (f or margin %.4f)", z$extra[i]))))
}
cat("written to", OUT, ": fig4b/4c/4d/4e PNG+PDF and fig4_values.csv\n")
