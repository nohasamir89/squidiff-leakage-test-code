#!/usr/bin/env Rscript
# Step 22, stage 2 of 4: the authors' package, unmodified, on the matrices written by prep_step22_fingerprint.py
# (PREREG_step22_fingerprint_identifiability.md, sections 3.1, 3.3 and 9).
#
# What it does, for one split folder (--dir):
#   1. Reads manifest.json, checks the sha256 of every matrix and metadata file it uses, and loads each float32 matrix
#      (written row-major as cells x genes) as a genes x cells matrix with gene and cell names.
#   2. Builds a Seurat object per matrix with an assay named "SCT" whose scale.data layer holds the standardised
#      values (prereg 3.2): the package reads exactly that layer and nothing else.
#   3. LearnDictionary on the dictionary object (half A of the real cells labelled by structure, control-A labelled
#      "control"; features = all genes; package defaults) -> dictionary.rds, dictionary.csv (label, structure, cells,
#      kept by the package's low-signal filter).
#   4. FingerprintCells at group level for every query set, and at cell level for the sets listed in the manifest
#      (the cells flagged cell_level = 1), each query object = the set's treated cells + control-B labelled "control";
#      package defaults (top = 100, seed 1448145). Writes map_<set>_group.csv (one row per condition) and
#      map_<set>_cell.csv (one row per cell) with the package's answers verbatim: top credible set, its log Bayes
#      factor, all credible sets, their log Bayes factors, the member signs.
#   5. Records timings.csv (set, level, queries, seconds, cores) and r_session.txt (R and package versions).
#   Existing output files are never overwritten: a rerun skips them (so a stopped job can resume) and recomputes only
#   what is missing; the dictionary is reloaded from dictionary.rds when it exists.
#
# Usage (R_env):  /scratch/ni2071/conda_envs/R_env/bin/Rscript run_step22_fingerprint.R --dir=<folder> --cores=<n>
#                 [--levels=group,cell] [--sets=<comma list, for tests>]
# Results depend on --cores only through the fork scheduling of the package's parallel loops; the value is recorded.

suppressPackageStartupMessages({
  library(Matrix)
  library(SeuratObject)
  library(Seurat)
  library(RNAFingerprinting)
  library(jsonlite)
  library(digest)
})
options(warn = 1, stringsAsFactors = FALSE)
T0 <- Sys.time()
logm <- function(...) cat(sprintf('[%6.0fs] ', as.numeric(difftime(Sys.time(), T0, units = 'secs'))), sprintf(...), '\n', sep = '')
fatal <- function(...) { cat('FATAL: ', sprintf(...), '\n', sep = ''); quit(status = 1) }

# ---------------------------------------------------------------- options
opt <- list(dir = NULL, cores = '1', levels = 'group,cell', sets = NULL)
for (a in commandArgs(trailingOnly = TRUE)) {
  kv <- strsplit(sub('^--', '', a), '=', fixed = TRUE)[[1]]
  if (length(kv) != 2 || !(kv[1] %in% names(opt))) fatal('unknown or malformed option %s', a)
  opt[[kv[1]]] <- kv[2]
}
if (is.null(opt$dir)) fatal('--dir=<split folder> is required')
dir <- opt$dir
cores <- as.integer(opt$cores)
if (is.na(cores) || cores < 1) fatal('--cores must be a positive integer')
levels <- strsplit(opt$levels, ',', fixed = TRUE)[[1]]
if (!all(levels %in% c('group', 'cell'))) fatal('--levels must be a subset of group,cell')
if (!file.exists(file.path(dir, 'manifest.json'))) fatal('no manifest.json in %s', dir)

# ---------------------------------------------------------------- the installed package must carry the Step 22 guard
# (provenance/R_env/rnafingerprinting_classify_guard.patch, 2026-10-14: the package's joint-fingerprint refit can find no
# effect, which crashed its calibrate(); the guard returns 'unassigned' there, as the package does in the first pass).
PATCH <- 'provenance/R_env/rnafingerprinting_classify_guard.patch'   # repository layout
if (!file.exists(PATCH)) PATCH <- '00_environment/R_env_provenance/rnafingerprinting_classify_guard.patch'   # release package layout
guard_ok <- any(grepl('length(keep_cs) == 0', deparse(RNAFingerprinting:::classify), fixed = TRUE))
if (!guard_ok) fatal('the installed RNAFingerprinting lacks the classify guard: apply %s to the clone and reinstall', PATCH)
patch_sha <- if (file.exists(PATCH)) digest::digest(PATCH, algo = 'sha256', file = TRUE) else 'patch file not found in this folder'

# ---------------------------------------------------------------- manifest and helpers
man <- jsonlite::fromJSON(file.path(dir, 'manifest.json'), simplifyVector = FALSE)
G <- as.integer(man$n_genes)
genes_file <- readLines(file.path(dir, 'genes.txt'))
if (length(genes_file) != G) fatal('genes.txt has %d names, manifest says %d', length(genes_file), G)
genes <- gsub('_', '-', genes_file, fixed = TRUE)       # Seurat replaces underscores in feature names; do it once, up front
if (anyDuplicated(genes)) fatal('gene names collide after replacing underscores')
if (!identical(genes, genes_file)) logm('%d gene names carried an underscore, replaced by a dash for Seurat', sum(genes != genes_file))
sets_group <- unlist(man$sets_group)
sets_cell <- unlist(man$sets_cell)
if (!is.null(opt$sets)) {
  sel <- strsplit(opt$sets, ',', fixed = TRUE)[[1]]
  if (!all(sel %in% sets_group)) fatal('--sets names a set that is not in the manifest')
  sets_group <- sel
  sets_cell <- intersect(sets_cell, sel)
}
logm('split %s: %d conditions, %d genes, %d structures; sets group %d, cell %d; cores %d; levels %s',
     man$split, as.integer(man$n_conditions), G, as.integer(man$n_structures), length(sets_group), length(sets_cell), cores,
     paste(levels, collapse = ','))

check_sha <- function(path, want) {
  got <- digest::digest(path, algo = 'sha256', file = TRUE)
  if (!identical(got, want)) fatal('%s: sha256 %s differs from the manifest (%s)', basename(path), got, want)
}

read_f32 <- function(path, n_rows, n_cols) {
  # On disk: row-major cells x genes (cell 0's G values, then cell 1's ...). R fills matrices column by column, so
  # matrix(v, nrow = G) puts cell i's values in column i: genes x cells, which is what Seurat expects.
  con <- file(path, 'rb')
  on.exit(close(con))
  v <- readBin(con, what = 'numeric', n = n_rows * n_cols, size = 4, endian = 'little')
  if (length(v) != n_rows * n_cols) fatal('%s: read %d values, expected %d', basename(path), length(v), n_rows * n_cols)
  matrix(v, nrow = n_cols, ncol = n_rows)
}

load_set <- function(name) {
  f <- man$files[[name]]
  if (is.null(f)) fatal('set %s is not in the manifest', name)
  mpath <- file.path(dir, f$matrix$path)
  cpath <- file.path(dir, f$meta$path)
  check_sha(mpath, f$matrix$sha256)
  check_sha(cpath, f$meta$sha256)
  meta <- read.csv(cpath, colClasses = 'character', check.names = FALSE)
  if (nrow(meta) != f$matrix$n_rows || f$matrix$n_cols != G) fatal('%s: metadata rows %d, matrix %d x %d', name, nrow(meta), f$matrix$n_rows, f$matrix$n_cols)
  if (anyDuplicated(meta$cell_id)) fatal('%s: duplicated cell ids', name)
  mat <- read_f32(mpath, f$matrix$n_rows, G)
  rownames(mat) <- genes
  colnames(mat) <- meta$cell_id
  rownames(meta) <- meta$cell_id
  list(mat = mat, meta = meta)
}

make_object <- function(mat, meta) {
  # The package reads GetAssayData(object, assay = 'SCT', layer = 'scale.data') and object@meta.data, nothing else.
  # The data layer is an all-zero placeholder of the same shape; the standardised values go into scale.data.
  placeholder <- Matrix::sparseMatrix(i = integer(0), j = integer(0), x = numeric(0), dims = dim(mat), dimnames = dimnames(mat))
  assay <- CreateAssayObject(data = placeholder)
  obj <- CreateSeuratObject(counts = assay, assay = 'SCT', meta.data = meta)
  obj <- tryCatch(SetAssayData(obj, assay = 'SCT', layer = 'scale.data', new.data = mat),
                  error = function(e) { logm('SetAssayData failed (%s); writing the scale.data slot directly', conditionMessage(e))
                                        obj[['SCT']]@scale.data <- mat; obj })
  sd <- GetAssayData(obj, assay = 'SCT', layer = 'scale.data')
  if (!('SCT' %in% Assays(obj)) || !identical(dim(sd), dim(mat)) || !identical(colnames(sd), colnames(mat)) || !identical(rownames(sd), rownames(mat)))
    fatal('the SCT scale.data layer does not hold the matrix as written')
  if (!identical(rownames(obj@meta.data), colnames(mat))) fatal('meta.data rows do not match the cells')
  obj
}

timing_path <- file.path(dir, 'timings.csv')
record_time <- function(set, level, n, secs) {
  row <- data.frame(set = set, level = level, n_queries = n, seconds = round(secs, 1), cores = cores, finished = format(Sys.time(), '%Y-%m-%d %H:%M:%S'))
  write.table(row, timing_path, sep = ',', row.names = FALSE, col.names = !file.exists(timing_path), append = file.exists(timing_path), quote = TRUE)
}

# ---------------------------------------------------------------- 1. the dictionary
dict_rds <- file.path(dir, 'dictionary.rds')
dict_csv <- file.path(dir, 'dictionary.csv')
if (file.exists(dict_rds) && file.exists(dict_csv)) {
  dict <- readRDS(dict_rds)
  logm('dictionary reloaded from %s (sha256 %s)', basename(dict_rds), digest::digest(dict_rds, algo = 'sha256', file = TRUE))
} else {
  d <- load_set('dictionary')
  n_lab <- table(d$meta$label)
  logm('dictionary cells: %d treated in %d structures (%d-%d cells each), %d control',
       sum(d$meta$role == 'treated'), sum(names(n_lab) != 'control'), min(n_lab[names(n_lab) != 'control']),
       max(n_lab[names(n_lab) != 'control']), sum(d$meta$role == 'control'))
  obj <- make_object(d$mat, d$meta)
  rm(d); invisible(gc())
  t1 <- Sys.time()
  dict <- LearnDictionary(obj, perturbation_meta = 'label', control_class = 'control', features = genes, mc.cores = cores)
  secs <- as.numeric(difftime(Sys.time(), t1, units = 'secs'))
  if (is.null(dict$fingerprints)) fatal('LearnDictionary kept no fingerprint')
  kept <- colnames(dict$fingerprints$Lambdas)
  labs <- setdiff(names(n_lab), 'control')
  lab2smi <- unlist(man$label_map)                      # structure -> label; invert
  smi_of <- setNames(names(lab2smi), unname(lab2smi))
  out <- data.frame(label = labs, structure = unname(smi_of[labs]), n_cells = as.integer(n_lab[labs]), kept = labs %in% kept)
  write.csv(out, dict_csv, row.names = FALSE)
  saveRDS(dict, dict_rds)
  record_time('dictionary', 'learn', length(labs), secs)
  logm('dictionary learned in %.0f s: K = %d of %d structures kept; filtered out: %s', secs, length(kept), length(labs),
       ifelse(length(setdiff(labs, kept)) > 0, paste(setdiff(labs, kept), collapse = ' '), 'none'))
  rm(obj); invisible(gc())
}
K <- ncol(dict$fingerprints$Lambdas)
if (!identical(rownames(dict$fingerprints$Lambdas), genes)) fatal('the dictionary genes differ from genes.txt')
logm('dictionary: K = %d fingerprints over %d genes', K, nrow(dict$fingerprints$Lambdas))

# ---------------------------------------------------------------- 2. the queries
ctrlB <- load_set('control_B')
logm('control-B: %d cells', ncol(ctrlB$mat))
cols_group <- c('top_credible_set', 'top_lbf', 'all_credible_sets', 'all_lbfs', 'all_signs')

for (name in sets_group) {
  out_group <- file.path(dir, sprintf('map_%s_group.csv', name))
  out_cell <- file.path(dir, sprintf('map_%s_cell.csv', name))
  do_group <- ('group' %in% levels) && !file.exists(out_group)
  do_cell <- ('cell' %in% levels) && (name %in% sets_cell) && !file.exists(out_cell) && !file.exists(paste0(out_cell, '.gz'))
  if (!do_group && !do_cell) { logm('%s: nothing to do (outputs exist or level not requested)', name); next }
  q <- load_set(paste0('query_', name))
  if (any(q$meta$role != 'treated')) fatal('%s: a query set must hold treated cells only', name)
  if (any(q$meta$condition == 'control')) fatal('%s: a treated cell is labelled control', name)

  if (do_group) {
    mat <- cbind(q$mat, ctrlB$mat)
    meta <- rbind(q$meta, ctrlB$meta)
    rownames(meta) <- meta$cell_id
    obj <- make_object(mat, meta)
    rm(mat); invisible(gc())
    t1 <- Sys.time()
    obj <- FingerprintCells(obj, condition_meta = 'condition', control_class = 'control', dictionary = dict, group_level = TRUE, mc.cores = cores)
    secs <- as.numeric(difftime(Sys.time(), t1, units = 'secs'))
    md <- obj@meta.data
    md <- md[md$role == 'treated', ]
    md <- md[!duplicated(md$condition), ]
    want <- paste0(cols_group, '.condition')              # the package appends the condition column's name at group level
    if (!all(want %in% colnames(md))) fatal('%s: expected columns %s in the group-level metadata', name, paste(want, collapse = ','))
    n_cells <- table(q$meta$condition)
    res <- data.frame(condition = md$condition, label = md$label, n_cells = as.integer(n_cells[md$condition]),
                      top_credible_set = md[[want[1]]], top_lbf = md[[want[2]]], all_credible_sets = md[[want[3]]],
                      all_lbfs = md[[want[4]]], all_signs = md[[want[5]]])
    if (nrow(res) != as.integer(man$n_conditions)) fatal('%s: %d group-level answers, %d conditions', name, nrow(res), man$n_conditions)
    write.csv(res, out_group, row.names = FALSE)
    record_time(name, 'group', nrow(res), secs)
    logm('%-16s group level: %d conditions in %.0f s; unassigned %d; mean |top set| %.2f', name, nrow(res), secs,
         sum(res$top_credible_set == 'unassigned'),
         mean(ifelse(res$top_credible_set == 'unassigned', 0, lengths(strsplit(res$top_credible_set, ',', fixed = TRUE)))))
    rm(obj, md, res); invisible(gc())
  }

  if (do_cell) {
    keep <- q$meta$cell_level == '1'
    mat <- cbind(q$mat[, keep, drop = FALSE], ctrlB$mat)
    meta <- rbind(q$meta[keep, ], ctrlB$meta)
    rownames(meta) <- meta$cell_id
    obj <- make_object(mat, meta)
    rm(mat); invisible(gc())
    t1 <- Sys.time()
    obj <- FingerprintCells(obj, condition_meta = 'condition', control_class = 'control', dictionary = dict, group_level = FALSE, mc.cores = cores)
    secs <- as.numeric(difftime(Sys.time(), t1, units = 'secs'))
    md <- obj@meta.data
    md <- md[md$role == 'treated', ]
    if (!all(cols_group %in% colnames(md))) fatal('%s: expected columns %s in the cell-level metadata', name, paste(cols_group, collapse = ','))
    res <- data.frame(cell_id = rownames(md), condition = md$condition, label = md$label,
                      top_credible_set = md$top_credible_set, top_lbf = md$top_lbf, all_credible_sets = md$all_credible_sets,
                      all_lbfs = md$all_lbfs, all_signs = md$all_signs)
    if (nrow(res) != sum(keep)) fatal('%s: %d cell-level answers, %d cells flagged', name, nrow(res), sum(keep))
    write.csv(res, out_cell, row.names = FALSE)
    record_time(name, 'cell', nrow(res), secs)
    logm('%-16s cell level: %d cells in %.0f s (%.2f s per cell per core); unassigned %d', name, nrow(res), secs,
         secs * cores / nrow(res), sum(res$top_credible_set == 'unassigned'))
    rm(obj, md, res); invisible(gc())
  }
  rm(q); invisible(gc())
}

# ---------------------------------------------------------------- 3. the session record
pk <- c('Seurat', 'SeuratObject', 'sctransform', 'irlba', 'presto', 'pbmcapply', 'matrixStats', 'Matrix', 'org.Hs.eg.db',
        'AnnotationDbi', 'ggplot2', 'ggrepel', 'RNAFingerprinting', 'jsonlite', 'digest')
lines <- c(sprintf('run_step22_fingerprint.R  %s  dir %s  cores %d  K %d', format(T0, '%Y-%m-%d %H:%M:%S'), dir, cores, K),
           R.version.string, sprintf('%-18s %s', pk, sapply(pk, function(p) as.character(packageVersion(p)))),
           sprintf('RNAFingerprinting classify guard: present (patch sha256 %s)', patch_sha),
           sprintf('dictionary.rds sha256 %s', digest::digest(dict_rds, algo = 'sha256', file = TRUE)), '', capture.output(sessionInfo()))
writeLines(lines, file.path(dir, 'r_session.txt'))
logm('done; session record written')
