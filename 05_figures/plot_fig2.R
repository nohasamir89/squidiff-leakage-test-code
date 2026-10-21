# plot_fig2.R - Fig. 2: the rewired model against the original model and the references (ggplot2).
#
# Purpose: draw Fig. 2a (gene identity, Tahoe-100M and sci-Plex) and Fig. 2b (population distance, Tahoe-100M)
#          directly from the committed result files, and write the plotted values to fig2_values.csv
#          (feeds the supplementary table).
# Logic:   Tahoe pooled values   = tahoe_results/step15_model_long.csv  (line == "ALL"; metrics de_overlap, energy)
#          Tahoe plate replicate = tahoe_results/step15_model_analysis.json  (per_split[[split]]$replicate)
#          sci-Plex values       = leakage_results/step14_analysis.json     (per_split[[split]]$thesis_metrics)
#          The rewired model is "recipe" in the Tahoe files and "fm_50" in the sci-Plex file (same configuration:
#          adaptive layer normalisation, flow matching, 50 sampling steps); both are drawn as "Rewired model".
#          The chance level of gene identity is the exact expectation of the Jaccard index of two random sets of
#          50 genes out of 200 (hypergeometric), computed here. Rows are placed by key, and the axis text comes
#          from row_labels, so a change of wording never empties a row. Panel b uses a log10 x axis because the
#          original model's distance is about ten times every other entry; the axis says so. Lower is better in b.
#          Nothing is typed in. Each panel is saved as its own file (PNG 600 dpi, PDF vector) for assembly.
# Where:   this script lives in <repo>/figures/squidiff_publication_figures/ and writes there:
#          fig2a_gene_identity_R.{png,pdf}, fig2b_population_distance_R.{png,pdf}, fig2_values.csv.
#          The repository root is found from the script's own location, so it runs from any directory:
#            Rscript figures/squidiff_publication_figures/plot_fig2.R
#          The values are printed for checking against the text.
# Needs:   ggplot2 and jsonlite. If jsonlite is missing:  conda install -p /scratch/ni2071/conda_envs/R_env -c conda-forge r-jsonlite

suppressPackageStartupMessages(library(ggplot2))
if (!requireNamespace("jsonlite", quietly = TRUE))
  stop("package 'jsonlite' is needed to read the JSON result files: ",
       "conda install -p /scratch/ni2071/conda_envs/R_env -c conda-forge r-jsonlite")

# ---- locate the repository root (two levels above this script) and the output folder ----
script_arg <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
REPO <- if (length(script_arg) == 1) normalizePath(file.path(dirname(script_arg), "..", "..")) else getwd()
OUT  <- file.path(REPO, "figures", "squidiff_publication_figures")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

TAHOE_LONG   <- file.path(REPO, "tahoe_results/step15_model_long.csv")
TAHOE_JSON   <- file.path(REPO, "tahoe_results/step15_model_analysis.json")
SCIPLEX_JSON <- file.path(REPO, "leakage_results/step14_analysis.json")
for (f in c(TAHOE_LONG, TAHOE_JSON, SCIPLEX_JSON)) if (!file.exists(f)) stop("input file not found: ", f)
SPLITS       <- c("drug_split_1", "drug_split_0", "drug_split_2")

# ---- labels and colours, as in the paper ----
split_map  <- c(drug_split_1 = "Split 1", drug_split_0 = "Split 0", drug_split_2 = "Split 2")
cols       <- c("Split 1" = "#2a78d6", "Split 0" = "#eb6834", "Split 2" = "#1baf7a")
offset     <- c("Split 1" = 0.2, "Split 0" = 0, "Split 2" = -0.2)        # Split 1 on top, as in the legend
row_labels <- c(base = "Original model", rewired = "Rewired model", ridge_train = "Training-set ridge",
                ridge_S7 = "Cross-validated ridge", mean_response = "Compound-blind average",
                control_paste = "Unaltered vehicle cells", replicate = "Plate replicate")
tahoe_keys   <- c(base = "base", rewired = "recipe", ridge_train = "ridge_train", ridge_S7 = "ridge_S7",
                  mean_response = "mean_response", control_paste = "control_paste")   # ours -> key in the files
sciplex_keys <- c(base = "base", rewired = "fm_50", ridge_train = "ridge_train", ridge_S7 = "ridge_S7",
                  mean_response = "mean_response")
INK <- "#0b0b0b"; INK2 <- "#52514e"; MUTED <- "#898781"; GRID <- "#e1e0d9"; AXIS <- "#c3c2b7"; SURFACE <- "#fcfcfb"

# ---- JSON written by Python may contain NaN / Infinity tokens, which are not valid JSON: read as text, replace
#      them by null, then parse. With simplifyVector = FALSE a null becomes NULL, and add() stops on any NULL
#      value the figure needs. ----
read_json_py <- function(path) {
  txt <- paste(readLines(path, warn = FALSE), collapse = "\n")
  txt <- gsub("(?<![A-Za-z0-9_\"])-?(NaN|Infinity)(?![A-Za-z0-9_\"])", "null", txt, perl = TRUE)
  jsonlite::fromJSON(txt, simplifyVector = FALSE)
}

# ---- chance level: overlap I ~ Hypergeom(50 'true' genes, 150 others, 50 drawn); Jaccard = I / (100 - I) ----
i <- 0:50
chance <- sum(dhyper(i, 50, 150, 50) * i / (100 - i))

# ---- read the values into one long table: panel, dataset, split, submission, source_key, label, metric, value ----
rows <- list()
add <- function(panel, dataset, split, key, source_key, metric, value) {
  stopifnot(length(value) == 1, is.finite(value))
  rows[[length(rows) + 1]] <<- data.frame(panel = panel, dataset = dataset, split = split, submission = key,
                                          source_key = source_key, label = unname(row_labels[key]),
                                          metric = metric, value = as.numeric(value), stringsAsFactors = FALSE)
}

long <- read.csv(TAHOE_LONG, stringsAsFactors = FALSE)
long <- long[long$line == "ALL", ]
for (split in SPLITS) {
  for (key in names(tahoe_keys)) {
    src <- tahoe_keys[[key]]
    for (metric in c("de_overlap", "energy")) {
      if (key == "control_paste" && metric == "de_overlap") next   # undefined: the vehicle cells predict no change
      v <- long$value[long$split == split & long$submission == src & long$metric == metric]
      stopifnot(length(v) == 1)                                    # stop if the file is not as expected
      add(if (metric == "de_overlap") "a" else "b", "Tahoe-100M", split, key, src, metric, v)
    }
  }
}
aj <- read_json_py(TAHOE_JSON)
for (split in SPLITS) {
  rep <- aj$per_split[[split]]$replicate
  add("a", "Tahoe-100M", split, "replicate", "replicate", "de_overlap", rep$de_overlap)
  add("b", "Tahoe-100M", split, "replicate", "replicate", "energy",     rep$energy)
}
bj <- read_json_py(SCIPLEX_JSON)
for (split in SPLITS) {
  tm <- bj$per_split[[split]]$thesis_metrics
  for (key in names(sciplex_keys)) {
    src <- sciplex_keys[[key]]
    add("a", "sci-Plex", split, key, src, "de_overlap", tm[[src]]$de_overlap)
  }
}
d <- do.call(rbind, rows)
d$split_label <- factor(split_map[d$split], levels = names(cols))
stopifnot(!any(is.na(d$split_label)))

values_out <- rbind(d[, c("panel", "dataset", "split", "submission", "source_key", "label", "metric", "value")],
                    data.frame(panel = "a", dataset = "both", split = "all", submission = "chance",
                               source_key = "computed", label = "Chance (50 of 200)", metric = "de_overlap",
                               value = chance))
write.csv(values_out, file.path(OUT, "fig2_values.csv"), row.names = FALSE)

# ---- shared look ----
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
    axis.text.y        = element_text(colour = INK, size = 8),
    axis.title.x       = element_text(colour = INK2, size = 8),
    strip.text         = element_text(hjust = 0, colour = INK, size = 8.5, face = "plain"),
    legend.position    = "bottom",
    legend.text        = element_text(size = 7, colour = INK),
    legend.key.size    = unit(8, "pt"),
    panel.spacing.x    = unit(14, "pt"),
    plot.margin        = margin(6, 10, 4, 6)
  )

place_rows <- function(x, key_order) {
  # numeric y position per row (bottom to top) plus the per-split offset; stops if a key is unknown
  x$row <- match(x$submission, key_order)
  stopifnot(!any(is.na(x$row)))
  x$y <- x$row + offset[as.character(x$split_label)]
  x
}

# ================= Panel a: gene identity, two datasets =================
keys_a <- c("base", "rewired", "ridge_train", "ridge_S7", "mean_response", "replicate")
a <- place_rows(d[d$panel == "a", ], keys_a)
stopifnot(all(keys_a %in% a$submission[a$dataset == "Tahoe-100M"]))
a$dataset <- factor(a$dataset, levels = c("Tahoe-100M", "sci-Plex"),
                    labels = c("Tahoe-100M, 14 cell lines", "sci-Plex, 3 cell lines"))
note <- data.frame(dataset = factor("sci-Plex, 3 cell lines", levels = levels(a$dataset)),
                   x = chance + 0.03, y = match("replicate", keys_a))

pa <- ggplot(a, aes(x = value, y = y, colour = split_label)) +
  geom_vline(xintercept = chance, linetype = "22", colour = INK2, linewidth = 0.4) +
  annotate("text", x = chance + 0.012, y = length(keys_a) + 0.45,
           label = sprintf("chance %.2f", chance), hjust = 0, size = 2.5, colour = INK2) +
  geom_point(size = 2.4) +
  geom_text(data = note, aes(x = x, y = y), label = "no plate replicate", inherit.aes = FALSE,
            hjust = 0, size = 2.5, colour = MUTED, fontface = "italic") +
  facet_wrap(~ dataset) +
  scale_colour_manual(values = cols, name = NULL) +
  scale_x_continuous(limits = c(0, 0.7), breaks = seq(0, 0.6, 0.2), expand = c(0, 0)) +
  scale_y_continuous(breaks = seq_along(keys_a), labels = unname(row_labels[keys_a]),
                     limits = c(0.4, length(keys_a) + 0.7), expand = c(0, 0)) +
  labs(x = "Gene identity (Jaccard, top-50 genes)", y = NULL) +
  base_theme

# ================= Panel b: population distance, Tahoe only =================
keys_b <- c("base", "rewired", "ridge_train", "ridge_S7", "mean_response", "control_paste", "replicate")
b <- place_rows(d[d$panel == "b", ], keys_b)
stopifnot(all(keys_b %in% b$submission), all(b$value > 0))

pb <- ggplot(b, aes(x = value, y = y, colour = split_label)) +
  geom_point(size = 2.4) +
  scale_colour_manual(values = cols, name = NULL) +
  scale_x_log10(limits = c(0.1, 3.2), breaks = c(0.1, 0.2, 0.5, 1, 2),
                labels = c("0.1", "0.2", "0.5", "1", "2"), expand = c(0, 0)) +
  scale_y_continuous(breaks = seq_along(keys_b), labels = unname(row_labels[keys_b]),
                     limits = c(0.4, length(keys_b) + 0.7), expand = c(0, 0)) +
  labs(x = "Population distance\n(energy distance, log scale; lower is better)", y = NULL) +
  base_theme +
  theme(axis.title.x = element_text(colour = INK2, size = 8, lineheight = 1.1, margin = margin(t = 4)))

ggsave(file.path(OUT, "fig2a_gene_identity_R.png"),       pa, width = 7.2, height = 3.4, units = "in", dpi = 600, bg = SURFACE)
ggsave(file.path(OUT, "fig2a_gene_identity_R.pdf"),       pa, width = 7.2, height = 3.4, units = "in", bg = SURFACE)
ggsave(file.path(OUT, "fig2b_population_distance_R.png"), pb, width = 4.6, height = 3.7, units = "in", dpi = 600, bg = SURFACE)
ggsave(file.path(OUT, "fig2b_population_distance_R.pdf"), pb, width = 4.6, height = 3.7, units = "in", bg = SURFACE)

# ---- print the values so they can be checked against the text ----
cat(sprintf("chance (exact hypergeometric expectation): %.4f\n", chance))
cat("panel a (gene identity)\n")
print(a[order(a$dataset, a$row, a$split_label), c("dataset", "label", "split_label", "value")], row.names = FALSE)
cat("panel b (population distance, Tahoe-100M)\n")
print(b[order(b$row, b$split_label), c("label", "split_label", "value")], row.names = FALSE)
cat("written to", OUT, ": fig2a_gene_identity_R.{png,pdf}, fig2b_population_distance_R.{png,pdf}, fig2_values.csv\n")
