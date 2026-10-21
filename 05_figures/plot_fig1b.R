# plot_fig1b.R - Fig. 1b: gene identity of the original model against the references (ggplot2 version).
#
# Purpose: draw Fig. 1b from figures/fig1b_values.csv, the table that plot_fig1b.py wrote from the result files.
# Logic:   no number is typed in. Values and the chance level come from the CSV; the chance level is its
#          'chance' row (exact hypergeometric expectation, computed in plot_fig1b.py). Same palette and
#          layout as the Python version: two panels, one dot per split, dashed chance line.
# Output:  figures/fig1b_gene_identity_R.png (600 dpi) and figures/fig1b_gene_identity_R.pdf (vector, for Figma).

suppressPackageStartupMessages(library(ggplot2))

d <- read.csv("figures/fig1b_values.csv", stringsAsFactors = FALSE)

# chance level is stored as its own row
chance <- d$gene_identity[d$submission == "chance"]
stopifnot(length(chance) == 1)
d <- d[d$submission != "chance", ]

# ---- labels and colours, as in the paper ----
split_map <- c(drug_split_1 = "Split 1", drug_split_0 = "Split 0", drug_split_2 = "Split 2")
cols      <- c("Split 1" = "#2a78d6", "Split 0" = "#eb6834", "Split 2" = "#1baf7a")
offset    <- c("Split 1" = 0.2, "Split 0" = 0, "Split 2" = -0.2)   # Split 1 on top, as in the legend
# rows are placed by the 'submission' key written by plot_fig1b.py (stable), not by the label text
# (which changed from "Like-for-like ridge" to "Training-set ridge"); row_order gives the axis text only
key_order <- c("base", "ridge_train", "ridge_S7", "mean_response", "replicate")   # bottom to top
row_order <- c("Original model", "Training-set ridge", "Cross-validated ridge",
               "Compound-blind average", "Plate replicate")
INK <- "#0b0b0b"; INK2 <- "#52514e"; MUTED <- "#898781"; GRID <- "#e1e0d9"; AXIS <- "#c3c2b7"; SURFACE <- "#fcfcfb"

d$split_label <- factor(split_map[d$split], levels = names(cols))
d$row         <- match(d$submission, key_order)            # numeric y position of each row
stopifnot(!any(is.na(d$row)))                              # stop if a submission key is not in key_order
stopifnot(all(key_order %in% d$submission[d$dataset == "Tahoe-100M"]))   # no row may be silently missing
d$label       <- row_order[d$row]                          # manuscript names, whatever the CSV says
d$y           <- d$row + offset[as.character(d$split_label)]
d$dataset     <- factor(d$dataset, levels = c("Tahoe-100M", "sci-Plex"),
                        labels = c("Tahoe-100M, 14 cell lines", "sci-Plex, 3 cell lines"))

# sci-Plex has no plate replicate: a note in that row, to the right of the chance line
note <- data.frame(dataset = factor("sci-Plex, 3 cell lines", levels = levels(d$dataset)),
                   x = chance + 0.03, y = match("replicate", key_order))

p <- ggplot(d, aes(x = gene_identity, y = y, colour = split_label)) +
  geom_vline(xintercept = chance, linetype = "22", colour = INK2, linewidth = 0.4) +
  annotate("text", x = chance + 0.012, y = length(row_order) + 0.45,
           label = sprintf("chance %.2f", chance), hjust = 0, size = 2.5, colour = INK2) +
  geom_point(size = 2.4) +
  geom_text(data = note, aes(x = x, y = y), label = "no plate replicate", inherit.aes = FALSE,
            hjust = 0, size = 2.5, colour = MUTED, fontface = "italic") +
  facet_wrap(~ dataset) +
  scale_colour_manual(values = cols, name = NULL) +
  scale_x_continuous(limits = c(0, 0.7), breaks = seq(0, 0.6, 0.2), expand = c(0, 0)) +
  scale_y_continuous(breaks = seq_along(row_order), labels = row_order,
                     limits = c(0.4, length(row_order) + 0.7), expand = c(0, 0)) +
  labs(x = "Gene identity (Jaccard, top-50 genes)", y = NULL) +
  theme_minimal(base_size = 9) +
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
    panel.spacing.x    = unit(14, "pt")
  )

dir.create("figures", showWarnings = FALSE)
ggsave("figures/fig1b_gene_identity_R.png", p, width = 7.2, height = 3.1, units = "in", dpi = 600, bg = SURFACE)
ggsave("figures/fig1b_gene_identity_R.pdf", p, width = 7.2, height = 3.1, units = "in", bg = SURFACE)

# print the values so they can be checked against the text
cat(sprintf("chance: %.4f\n", chance))
print(d[order(d$dataset, d$row, d$split_label), c("dataset", "label", "split_label", "gene_identity")], row.names = FALSE)
cat("written: figures/fig1b_gene_identity_R.png and .pdf\n")
