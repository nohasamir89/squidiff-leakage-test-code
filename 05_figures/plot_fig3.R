# plot_fig3.R - Fig. 3b and 3c: interventions on the compound input of the rewired model (sci-Plex).
#
# Purpose: (b) the five interventions that gave the generative model more or different compound information,
#              each as its pre-specified contrast against the rewired model of the time on the development split;
#          (c) the cross-attention variant on all three splits: gene identity against the rewired model and
#              against the parameter-matched control, and population distance against the rewired model.
# Logic:   every value is read from the committed analysis records (nothing typed in):
#            leakage_results/step9_analysis.json    mechanism-of-action labels (contrast "both minus fingerprint",
#                                                   paired median difference in response correlation)
#            leakage_results/step10_analysis.json   learned chemical embedding alone ("chemvae minus fingerprint")
#                                                   and with the fingerprint ("fp_chemvae minus fingerprint"),
#                                                   paired median difference in response correlation
#            leakage_results/step13_analysis.json   fingerprint self-attention, attention over training compounds,
#                                                   cross-attention: mean difference in gene identity (de_overlap)
#                                                   against the reference and the control, and in population
#                                                   distance (energy) against the reference; development split, 95 %
#            leakage_results/step13b_analysis.json  the same contrasts for cross-attention on the two confirmation
#                                                   splits, 97.5 %
#          A point is filled when its interval excludes zero and hollow when it includes zero.
#          The JSON files may contain NaN tokens (written by Python); they are turned into null before parsing.
# Where:   this script lives in <repo>/figures/squidiff_publication_figures/ and writes there:
#          fig3b_interventions_R.{png,pdf}, fig3c_cross_attention_R.{png,pdf}, fig3_values.csv.
#            Rscript figures/squidiff_publication_figures/plot_fig3.R
# Needs:   ggplot2 and jsonlite.

suppressPackageStartupMessages(library(ggplot2))
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("package 'jsonlite' is needed")

script_arg <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
REPO <- if (length(script_arg) == 1) normalizePath(file.path(dirname(script_arg), "..", "..")) else getwd()
OUT  <- file.path(REPO, "figures", "squidiff_publication_figures")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
J <- function(name) file.path(REPO, "leakage_results", name)
for (f in c("step9_analysis.json", "step10_analysis.json", "step13_analysis.json", "step13b_analysis.json"))
  if (!file.exists(J(f))) stop("input file not found: ", J(f))

read_json_py <- function(path) {
  txt <- paste(readLines(path, warn = FALSE), collapse = "\n")
  txt <- gsub("(?<![A-Za-z0-9_\"])-?(NaN|Infinity)(?![A-Za-z0-9_\"])", "null", txt, perl = TRUE)
  jsonlite::fromJSON(txt, simplifyVector = FALSE)
}
pick <- function(lst, pattern) {           # the one list entry whose name matches the pattern
  k <- grep(pattern, names(lst), value = TRUE, fixed = TRUE)
  if (length(k) != 1) stop("expected one key matching '", pattern, "', found: ", paste(k, collapse = " | "))
  lst[[k]]
}

INK <- "#0b0b0b"; INK2 <- "#52514e"; MUTED <- "#898781"; GRID <- "#e1e0d9"; AXIS <- "#c3c2b7"; SURFACE <- "#fcfcfb"
cols <- c("Split 1" = "#2a78d6", "Split 0" = "#eb6834", "Split 2" = "#1baf7a")
fmt_tick <- function(x) sub("\\.?0+$", "", sprintf("%.3f", x))      # up to 3 decimals, trailing zeros dropped

# ---------------- collect the contrasts into one table ----------------
rows <- list()
add <- function(panel, intervention, quantity, split, level, point, lo, hi, source) {
  stopifnot(is.finite(point), is.finite(lo), is.finite(hi), lo <= point, point <= hi)
  rows[[length(rows) + 1]] <<- data.frame(panel = panel, intervention = intervention, quantity = quantity,
                                          split = split, level = level, point = point, lo = lo, hi = hi,
                                          source = source, stringsAsFactors = FALSE)
}
RC <- "Response correlation\n(paired median difference)"
GI <- "Gene identity\n(mean difference)"

s9  <- read_json_py(J("step9_analysis.json"))
c9  <- pick(s9$contrasts, "both minus fingerprint")
add("b", "Mechanism-of-action labels", RC, "Split 1", "95 %", c9$paired_median, c9$ci[[1]], c9$ci[[2]], "step9_analysis.json")

s10 <- read_json_py(J("step10_analysis.json"))
c10b <- pick(s10$contrasts, "fp_chemvae minus fingerprint")
# 'chemvae minus fingerprint' is contained in both keys; the embedding-alone key is the one without 'fp_'
k10 <- names(s10$contrasts)
k_alone <- k10[grepl("chemvae minus fingerprint", k10, fixed = TRUE) & !grepl("fp_chemvae", k10, fixed = TRUE)]
stopifnot(length(k_alone) == 1)
c10a <- s10$contrasts[[k_alone]]
add("b", "Learned chemical embedding", RC, "Split 1", "95 %", c10a$paired_median, c10a$ci[[1]], c10a$ci[[2]], "step10_analysis.json")
add("b", "Chemical embedding + fingerprint", RC, "Split 1", "95 %", c10b$paired_median, c10b$ci[[1]], c10b$ci[[2]], "step10_analysis.json")

s13 <- read_json_py(J("step13_analysis.json"))
arms13 <- s13$results$arms
lab13 <- c(attn_fp = "Fingerprint self-attention", attn_mem = "Attention over similar training compounds",
           attn_cross = "Cross-attention, compound × cell state")
for (arm in names(lab13)) {
  x <- arms13[[arm]]$de_overlap_minus_reference
  add("b", lab13[[arm]], GI, "Split 1", "95 %", x$diff, x$ci[[1]], x$ci[[2]], "step13_analysis.json")
}

# panel c: cross-attention on the three splits
Q1 <- "Gene identity\nvs rewired model"; Q2 <- "Gene identity\nvs matched control"; Q3 <- "Population distance\nvs rewired model"
x <- arms13$attn_cross
add("c", "Cross-attention", Q1, "Split 1", "95 %", x$de_overlap_minus_reference$diff, x$de_overlap_minus_reference$ci[[1]], x$de_overlap_minus_reference$ci[[2]], "step13_analysis.json")
add("c", "Cross-attention", Q2, "Split 1", "95 %", x$de_overlap_minus_control$diff,   x$de_overlap_minus_control$ci[[1]],   x$de_overlap_minus_control$ci[[2]],   "step13_analysis.json")
add("c", "Cross-attention", Q3, "Split 1", "95 %", x$energy_minus_reference$diff,     x$energy_minus_reference$ci[[1]],     x$energy_minus_reference$ci[[2]],     "step13_analysis.json")
s13b <- read_json_py(J("step13b_analysis.json"))
for (sp in c("drug_split_0", "drug_split_2")) {
  p <- s13b$per_split[[sp]]$primary
  lab <- c(drug_split_0 = "Split 0", drug_split_2 = "Split 2")[[sp]]
  add("c", "Cross-attention", Q1, lab, "97.5 %", p$de_overlap_minus_reference$diff, p$de_overlap_minus_reference$ci[[1]], p$de_overlap_minus_reference$ci[[2]], "step13b_analysis.json")
  add("c", "Cross-attention", Q2, lab, "97.5 %", p$de_overlap_minus_control$diff,   p$de_overlap_minus_control$ci[[1]],   p$de_overlap_minus_control$ci[[2]],   "step13b_analysis.json")
  add("c", "Cross-attention", Q3, lab, "97.5 %", p$energy_minus_reference$diff,     p$energy_minus_reference$ci[[1]],     p$energy_minus_reference$ci[[2]],     "step13b_analysis.json")
}
d <- do.call(rbind, rows)
d$excludes_zero <- d$lo > 0 | d$hi < 0
write.csv(d, file.path(OUT, "fig3_values.csv"), row.names = FALSE)

base_theme <- theme_minimal(base_size = 9) +
  theme(
    plot.background    = element_rect(fill = SURFACE, colour = NA),
    panel.background   = element_rect(fill = SURFACE, colour = NA),
    panel.grid.minor   = element_blank(),
    panel.grid.major   = element_line(colour = GRID, linewidth = 0.3),
    axis.line.x        = element_line(colour = AXIS, linewidth = 0.4),
    axis.ticks.x       = element_line(colour = AXIS, linewidth = 0.3),
    axis.text          = element_text(colour = INK2, size = 7),
    axis.title.x       = element_text(colour = INK2, size = 8, margin = margin(t = 4)),
    strip.text         = element_text(hjust = 0, colour = INK, size = 8, lineheight = 0.95, margin = margin(b = 3)),
    plot.caption       = element_text(hjust = 0, size = 6.5, colour = INK2, lineheight = 1.05, margin = margin(t = 6)),
    plot.caption.position = "plot",                       # caption spans the whole figure, not only the panel area
    plot.margin        = margin(6, 10, 4, 6)
  )

# ---------------- Panel b ----------------
b <- d[d$panel == "b", ]
order_b <- c("Mechanism-of-action labels", "Learned chemical embedding", "Chemical embedding + fingerprint",
             "Fingerprint self-attention", "Attention over similar training compounds", "Cross-attention, compound × cell state")
stopifnot(setequal(order_b, b$intervention))
b$intervention <- factor(b$intervention, levels = rev(order_b))          # first intervention at the top
b$quantity     <- factor(b$quantity, levels = c(RC, GI))
b$fill         <- ifelse(b$excludes_zero, cols[["Split 1"]], SURFACE)

pb <- ggplot(b, aes(x = point, y = intervention)) +
  geom_vline(xintercept = 0, linetype = "22", colour = INK2, linewidth = 0.4) +
  geom_errorbar(aes(xmin = lo, xmax = hi), width = 0.18, orientation = "y", colour = cols[["Split 1"]], linewidth = 0.5) +
  geom_point(aes(fill = fill), shape = 21, colour = cols[["Split 1"]], size = 2.4, stroke = 0.6) +
  scale_fill_identity() +
  scale_x_continuous(breaks = scales::breaks_extended(n = 5), labels = fmt_tick) +
  facet_wrap(~ quantity, scales = "free", ncol = 1) +
  labs(x = "Difference from the rewired model (development split)", y = NULL,
       caption = "Filled: interval excludes zero; hollow: includes zero.\nCross-attention is the only effect that cleared zero; it was not confirmed on the other two splits (panel c).") +
  base_theme +
  theme(panel.grid.major.y = element_blank(), axis.text.y = element_text(colour = INK, size = 7.5),
        panel.spacing.y = unit(14, "pt"))

# ---------------- Panel c ----------------
cc <- d[d$panel == "c", ]
cc$split    <- factor(cc$split, levels = names(cols))
cc$quantity <- factor(cc$quantity, levels = c(Q1, Q2, Q3))
cc$fill     <- ifelse(cc$excludes_zero, cols[as.character(cc$split)], SURFACE)

pc <- ggplot(cc, aes(x = split, y = point, colour = split)) +
  geom_hline(yintercept = 0, linetype = "22", colour = INK2, linewidth = 0.4) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.18, linewidth = 0.5) +
  geom_point(aes(fill = fill), shape = 21, size = 2.4, stroke = 0.6) +
  scale_colour_manual(values = cols, guide = "none") +
  scale_fill_identity() +
  facet_wrap(~ quantity, scales = "free_y", nrow = 1) +
  scale_y_continuous(labels = fmt_tick) +
  labs(x = NULL, y = "Cross-attention minus comparator",
       caption = "Split 1 is the development split (95 % interval); Splits 0 and 2 are the confirmation splits (97.5 %).\nFilled: interval excludes zero; hollow: includes zero.") +
  base_theme +
  theme(panel.grid.major.x = element_blank(), axis.text.x = element_text(colour = INK, size = 7.5),
        axis.title.y = element_text(colour = INK2, size = 8), panel.spacing.x = unit(14, "pt"))

ggsave(file.path(OUT, "fig3b_interventions_R.png"),   pb, width = 5.6, height = 4.3, units = "in", dpi = 600, bg = SURFACE)
ggsave(file.path(OUT, "fig3b_interventions_R.pdf"),   pb, width = 5.6, height = 4.3, units = "in", bg = SURFACE)
ggsave(file.path(OUT, "fig3c_cross_attention_R.png"), pc, width = 6.4, height = 3.0, units = "in", dpi = 600, bg = SURFACE)
ggsave(file.path(OUT, "fig3c_cross_attention_R.pdf"), pc, width = 6.4, height = 3.0, units = "in", bg = SURFACE)

# ---------------- printed values for checking against the text ----------------
fmt <- function(x) sprintf("%+.4f [%+.4f, %+.4f] %s", x$point, x$lo, x$hi, ifelse(x$excludes_zero, "excludes zero", "includes zero"))
cat("panel b (development split)\n")
for (i in seq_len(nrow(b))) cat(sprintf("  %-42s %-24s %s\n", as.character(b$intervention[i]), gsub("\n", " ", as.character(b$quantity[i])), fmt(b[i, ])))
cat("panel c (cross-attention)\n")
for (i in seq_len(nrow(cc))) cat(sprintf("  %-32s %-8s %s\n", gsub("\n", " ", as.character(cc$quantity[i])), as.character(cc$split[i]), fmt(cc[i, ])))
cat("written to", OUT, ": fig3b_interventions_R.{png,pdf}, fig3c_cross_attention_R.{png,pdf}, fig3_values.csv\n")
