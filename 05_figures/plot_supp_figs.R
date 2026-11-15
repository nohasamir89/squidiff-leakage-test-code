# plot_supp_figs.R - Supplementary Figs S1-S3 (ggplot2): spread agreement, response-magnitude agreement and
#                    RNA-fingerprinting recognition.
#
# Purpose: draw the three supplementary figures as clean images: no text printed inside the picture (the caption
#          is in the supplementary file) and file names that carry the supplementary figure number.
#            Supplementary Fig. S1  spread agreement on Tahoe-100M                        (step 19; Suppl. Note 5)
#            Supplementary Fig. S2  response-magnitude agreement, Tahoe-100M and sci-Plex (steps 16, 17; Note 4)
#            Supplementary Fig. S3  RNA-fingerprinting recognition on Tahoe-100M          (step 22; Note 6)
#          These were panels c, b and d of the former Fig. 5 (plot_fig5.R). Panel a of that figure is withdrawn
#          from the paper and is not drawn here. plot_fig5.R is left untouched.
# Logic:   1. Every value is read from the committed records; nothing is typed in.
#               tahoe_results/step19_heterogeneity.json (S1) metrics[sub].ha, contrasts '..ha_<sub>_minus_0',
#                                                            replicate.ha with '..ha_ceiling_minus_0'
#               tahoe_results/step16_analysis.json      (S2) metrics[sub].mra, contrasts '..mra_<sub>_minus_0',
#                                                            replicate.mra (14 contexts with a plate copy)
#               leakage_results/step17_mra.json         (S2) metrics[sub].mra, contrasts '..mra_<sub>_minus_0',
#                                                            anchor.mra
#               tahoe_results/step22_analysis.json      (S3) rates.group[set]: hit, chance, unassigned (fractions)
#          2. Guard: the values are compared, row by row, with panels c, b and d of the committed
#             fig5_values.csv, the file the supplementary tables S16-S18 were checked against. The script stops,
#             and writes nothing, if a row is missing or a value differs by more than 1e-12.
#          3. The reading code, the colours, the theme and the sizes are those of plot_fig5.R, so the pictures
#             are the former panels without their printed text.
#          Filled point: interval excludes zero; hollow: includes it. Diamonds: references without a test.
#          Intervals are the records' 97.5 % cluster-bootstrap percentile intervals. The rewired model is
#          'recipe' on Tahoe-100M and 'fm_50' on sci-Plex.
# Where:   lives in <repo>/figures/squidiff_publication_figures/ and writes there (.png 600 dpi and .pdf):
#            supp_fig_S1_spread, supp_fig_S2_magnitude, supp_fig_S3_fingerprinting
#          Run from the repository:
#            Rscript figures/squidiff_publication_figures/plot_supp_figs.R
# Needs:   ggplot2 and jsonlite.

suppressPackageStartupMessages(library(ggplot2))
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("package 'jsonlite' is needed")

script_arg <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
REPO <- if (length(script_arg) == 1) normalizePath(file.path(dirname(script_arg), "..", "..")) else getwd()
OUT  <- file.path(REPO, "figures", "squidiff_publication_figures")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
P16  <- file.path(REPO, "tahoe_results/step16_analysis.json")
P17  <- file.path(REPO, "leakage_results/step17_mra.json")
P19  <- file.path(REPO, "tahoe_results/step19_heterogeneity.json")
P22  <- file.path(REPO, "tahoe_results/step22_analysis.json")
OLD  <- file.path(OUT, "fig5_values.csv")                       # committed values of the former Fig. 5 (guard)
for (f in c(P16, P17, P19, P22, OLD)) if (!file.exists(f)) stop("input file not found: ", f)

read_json_py <- function(path) {
  txt <- paste(readLines(path, warn = FALSE), collapse = "\n")
  txt <- gsub("(?<![A-Za-z0-9_\"])-?(NaN|Infinity)(?![A-Za-z0-9_\"])", "null", txt, perl = TRUE)
  jsonlite::fromJSON(txt, simplifyVector = FALSE)
}
find_contrast <- function(contrasts, pattern) {       # the one contrast whose key matches the regex
  k <- grep(pattern, names(contrasts), value = TRUE)
  if (length(k) != 1) stop("expected one contrast matching '", pattern, "', found: ", paste(k, collapse = " | "))
  contrasts[[k]]
}
INK <- "#0b0b0b"; INK2 <- "#52514e"; MUTED <- "#898781"; GRID <- "#e1e0d9"; AXIS <- "#c3c2b7"; SURFACE <- "#fcfcfb"
cols      <- c("Split 1" = "#2a78d6", "Split 0" = "#eb6834", "Split 2" = "#1baf7a")
offset    <- c("Split 1" = 0.22, "Split 0" = 0, "Split 2" = -0.22)
split_map <- c(drug_split_1 = "Split 1", drug_split_0 = "Split 0", drug_split_2 = "Split 2")
SPLITS    <- c("drug_split_1", "drug_split_0", "drug_split_2")
fmt_tick  <- function(x) sub("\\.?0+$", "", sprintf("%.3f", x))

rows <- list()
add <- function(panel, facet, row, split, point, lo = NA, hi = NA, chance = NA, source = "") {
  stopifnot(is.finite(point))
  if (is.finite(lo)) stopifnot(lo <= point, point <= hi)
  rows[[length(rows) + 1]] <<- data.frame(panel = panel, facet = facet, row = row, split = split, point = point,
                                          lo = lo, hi = hi, chance = chance, source = source, stringsAsFactors = FALSE)
}


# ================= (b) response-magnitude agreement, Tahoe and sci-Plex =================
B_ROWS <- c(base = "Original model", rewired = "Rewired model", ridge_train = "Training-set ridge", ridge_S7 = "Cross-validated ridge")
s16 <- read_json_py(P16)
for (sp in SPLITS) {
  ps <- s16$per_split[[sp]]
  for (s in names(B_ROWS)) {
    key <- if (s == "rewired") "recipe" else s
    v <- ps$metrics[[key]]$mra
    x <- find_contrast(ps$contrasts, paste0("mra_", key, "_minus_0$"))
    add("b", "Tahoe-100M", B_ROWS[[s]], split_map[[sp]], v, x$ci[[1]], x$ci[[2]], chance = 0, source = "step16_analysis.json")
  }
  add("b", "Tahoe-100M", "Plate replicate (14 contexts)", split_map[[sp]], ps$replicate$mra, chance = 0, source = "step16_analysis.json (replicate)")
}
s17 <- read_json_py(P17)
for (sp in SPLITS) {
  ps <- s17$per_split[[sp]]
  for (s in names(B_ROWS)) {
    key <- if (s == "rewired") "fm_50" else s
    v <- ps$metrics[[key]]$mra
    x <- find_contrast(ps$contrasts, paste0("^(Q1_primary|secondary)_mra_", key, "_minus_0$"))
    add("b", "sci-Plex", B_ROWS[[s]], split_map[[sp]], v, x$ci[[1]], x$ci[[2]], chance = 0, source = "step17_mra.json")
  }
  add("b", "sci-Plex", "Split-half anchor", split_map[[sp]], ps$anchor$mra, chance = 0, source = "step17_mra.json (anchor)")
}

# ================= (c) spread agreement, Tahoe =================
s19 <- read_json_py(P19)
C_ROWS <- c(recipe = "Rewired model", ot_pair = "Rewired model, OT pairing", mean_response = "Compound-blind average",
            ridge_train = "Training-set ridge", ridge_S7 = "Cross-validated ridge")
for (sp in SPLITS) {
  ps <- s19$per_split[[sp]]
  for (s in names(C_ROWS)) {
    v <- ps$metrics[[s]]$ha
    x <- find_contrast(ps$contrasts, paste0("ha_", s, "_minus_0$"))
    add("c", "Spread agreement", C_ROWS[[s]], split_map[[sp]], v, x$ci[[1]], x$ci[[2]], chance = 0, source = "step19_heterogeneity.json")
  }
  x <- find_contrast(ps$contrasts, "ha_ceiling_minus_0$")
  add("c", "Spread agreement", "Plate replicate (14 contexts)", split_map[[sp]], ps$replicate$ha, x$ci[[1]], x$ci[[2]], chance = 0, source = "step19_heterogeneity.json (replicate)")
}

# ================= (d) RNA-fingerprinting recognition, Tahoe (group level) =================
s22 <- read_json_py(P22)
D_ROWS <- c(real_half = "Measured cells (other half)", recipe = "Rewired model", ot_pair = "Rewired model, OT pairing",
            mb32 = "Rewired model, OT pairing (m = 32)", mean_response = "Compound-blind average",
            control_paste = "Unaltered vehicle cells", ridge_train = "Training-set ridge", ridge_S7 = "Cross-validated ridge")
D1 <- "Recognised (% of query populations)"; D2 <- "No compound assigned (%)"
for (sp in SPLITS) {
  g <- s22$per_split[[sp]]$rates$group
  for (s in names(D_ROWS)) {
    r <- g[[s]]; stopifnot(!is.null(r))
    add("d", D1, D_ROWS[[s]], split_map[[sp]], 100 * r$hit, chance = 100 * r$chance, source = "step22_analysis.json (rates.group)")
    add("d", D2, D_ROWS[[s]], split_map[[sp]], 100 * r$unassigned, source = "step22_analysis.json (rates.group)")
  }
}

# ---- all rows in one table; 'excludes' = the interval does not contain zero ----
d <- do.call(rbind, rows)
d$excludes <- ifelse(is.finite(d$lo), d$lo > d$chance | d$hi < d$chance, NA)

# ---- guard: the same values as panels b, c and d of the committed fig5_values.csv ----
# Each row is identified by panel, facet, row label and split. Missing values (no interval, no chance level)
# must be missing on both sides.
old <- read.csv(OLD, stringsAsFactors = FALSE, check.names = FALSE)
old <- old[old$panel %in% c("b", "c", "d"), ]
key <- function(z) paste(z$panel, z$facet, z$row, z$split, sep = " | ")
if (nrow(old) != nrow(d) || !setequal(key(old), key(d)) || anyDuplicated(key(d)))
  stop("guard: the rows differ from panels b, c, d of fig5_values.csv (", nrow(d), " here, ", nrow(old), " there)")
old <- old[match(key(d), key(old)), ]
for (col in c("point", "lo", "hi", "chance")) {
  x <- d[[col]]; y <- as.numeric(old[[col]])
  bad <- xor(is.na(x), is.na(y)) | (!is.na(x) & !is.na(y) & abs(x - y) > 1e-12)
  if (any(bad)) stop("guard: column '", col, "' differs from fig5_values.csv at: ", paste(key(d)[bad], collapse = " ; "))
}
if (!identical(as.logical(d$excludes), as.logical(old$excludes))) stop("guard: column 'excludes' differs from fig5_values.csv")
cat("guard passed:", nrow(d), "values equal panels b, c, d of fig5_values.csv (tolerance 1e-12)\n")

base_theme <- theme_minimal(base_size = 9) +
  theme(
    plot.background    = element_rect(fill = SURFACE, colour = NA),
    panel.background   = element_rect(fill = SURFACE, colour = NA),
    panel.grid.major.y = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.x = element_line(colour = GRID, linewidth = 0.3),
    axis.line.x        = element_line(colour = AXIS, linewidth = 0.4),
    axis.ticks.x       = element_line(colour = AXIS, linewidth = 0.3),
    axis.text.x        = element_text(colour = INK2, size = 7),
    axis.text.y        = element_text(colour = INK, size = 7.5),
    axis.title.x       = element_text(colour = INK2, size = 8),
    strip.text         = element_text(hjust = 0, colour = INK, size = 8.5),
    legend.position    = "bottom",
    legend.text        = element_text(size = 7, colour = INK),
    legend.key.size    = unit(8, "pt"),
    plot.caption       = element_text(hjust = 0, size = 6.5, colour = INK2, lineheight = 1.05, margin = margin(t = 6)),
    plot.caption.position = "plot",
    panel.spacing.x    = unit(16, "pt"),
    plot.margin        = margin(6, 10, 4, 6)
  )

# dot plot with per-split offsets, optional intervals, and a dashed line at zero; no text inside the picture
dotplot <- function(z, row_order, xlab, chance_line, ref_rows = character(0)) {
  z$row_i <- match(z$row, row_order); stopifnot(!any(is.na(z$row_i)))
  z$split <- factor(z$split, levels = names(cols))
  z$y     <- z$row_i + offset[as.character(z$split)]
  z$fill  <- ifelse(is.na(z$excludes) | z$excludes, cols[as.character(z$split)], SURFACE)
  z$shape <- ifelse(z$row %in% ref_rows, 23, 21)                   # references without a test: diamonds
  p <- ggplot(z, aes(x = point, y = y, colour = split)) +
    geom_vline(xintercept = chance_line, linetype = "22", colour = INK2, linewidth = 0.4)
  if (any(is.finite(z$lo)))
    p <- p + geom_errorbar(data = z[is.finite(z$lo), ], aes(xmin = lo, xmax = hi), width = 0.16, orientation = "y", linewidth = 0.45)
  p + geom_point(aes(fill = fill, shape = shape), size = 2.2, stroke = 0.6) +
    scale_colour_manual(values = cols, name = NULL) + scale_fill_identity() + scale_shape_identity() +
    scale_y_continuous(breaks = seq_along(row_order), labels = row_order,
                       limits = c(0.4, length(row_order) + 0.6), expand = c(0, 0)) +
    scale_x_continuous(labels = fmt_tick) +
    labs(x = xlab, y = NULL) +
    base_theme
}

# ---- Supplementary Fig. S1: spread agreement, Tahoe-100M (former panel c) ----
cc <- d[d$panel == "c", ]
order_c <- c("Rewired model", "Rewired model, OT pairing", "Compound-blind average", "Training-set ridge",
             "Cross-validated ridge", "Plate replicate (14 contexts)")                 # bottom to top
p1 <- dotplot(cc, order_c, "Spread agreement (rank correlation within a context; 0 = none)", 0)

# ---- Supplementary Fig. S2: response-magnitude agreement, Tahoe-100M and sci-Plex (former panel b) ----
b <- d[d$panel == "b", ]
order_b <- c("Original model", "Rewired model", "Training-set ridge", "Cross-validated ridge",
             "Plate replicate (14 contexts)", "Split-half anchor")
b$facet <- factor(b$facet, levels = c("Tahoe-100M", "sci-Plex"))
p2 <- dotplot(b, order_b, "Response-magnitude agreement (rank correlation within a context; 0 = none)", 0,
              ref_rows = c("Plate replicate (14 contexts)", "Split-half anchor")) + facet_wrap(~ facet, scales = "free_x")

# ---- Supplementary Fig. S3: RNA-fingerprinting recognition, Tahoe-100M (former panel d) ----
# Left: share recognised, with the chance level as a grey bar. Right: share with no compound assigned.
dd <- d[d$panel == "d", ]
order_d <- rev(unname(D_ROWS))                                     # measured cells at the top
dd$facet <- factor(dd$facet, levels = c(D1, D2))
dd$row_i <- match(dd$row, order_d); dd$split <- factor(dd$split, levels = names(cols))
dd$y <- dd$row_i + offset[as.character(dd$split)]
chance_pts <- dd[dd$facet == D1 & is.finite(dd$chance), ]
p3 <- ggplot(dd, aes(x = point, y = y, colour = split)) +
  geom_point(data = chance_pts, aes(x = chance, y = y), colour = MUTED, shape = 124, size = 2.2, inherit.aes = FALSE) +
  geom_point(size = 2.2) +
  facet_wrap(~ facet, scales = "free_x") +
  scale_colour_manual(values = cols, name = NULL) +
  scale_y_continuous(breaks = seq_along(order_d), labels = order_d, limits = c(0.4, length(order_d) + 0.6), expand = c(0, 0)) +
  scale_x_continuous(labels = fmt_tick, limits = function(l) c(0, l[2]), expand = expansion(mult = c(0, 0.08))) +
  labs(x = NULL, y = NULL) +
  base_theme

# ---- write the images (heights are those of plot_fig5.R less the room of the removed text) ----
save2 <- function(name, p, w, h) {
  ggsave(file.path(OUT, paste0(name, ".png")), p, width = w, height = h, units = "in", dpi = 600, bg = SURFACE)
  ggsave(file.path(OUT, paste0(name, ".pdf")), p, width = w, height = h, units = "in", bg = SURFACE)
}
save2("supp_fig_S1_spread",         p1, 5.6, 2.9)
save2("supp_fig_S2_magnitude",      p2, 7.4, 3.1)
save2("supp_fig_S3_fingerprinting", p3, 7.4, 3.2)

# ---- printed values, for checking against Supplementary Tables S17, S16 and S18 ----
for (p in c("c", "b", "d")) {
  cat(c(c = "Supplementary Fig. S1", b = "Supplementary Fig. S2", d = "Supplementary Fig. S3")[[p]], "\n"); z <- d[d$panel == p, ]
  for (i in seq_len(nrow(z))) cat(sprintf("  %-36s %-36s %-8s %8.4f%s%s\n", z$facet[i], z$row[i], z$split[i], z$point[i],
                                          ifelse(is.finite(z$lo[i]), sprintf("  [%.4f, %.4f] %s", z$lo[i], z$hi[i], ifelse(z$excludes[i], "excludes zero", "includes zero")), ""),
                                          ifelse(is.finite(z$chance[i]) & p == "d", sprintf("  chance %.2f", z$chance[i]), "")))
}
cat("written to", OUT, ": supp_fig_S1_spread, supp_fig_S2_magnitude, supp_fig_S3_fingerprinting (.png, .pdf)\n")
