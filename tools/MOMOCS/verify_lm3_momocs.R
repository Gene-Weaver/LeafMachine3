# =================================================================================================
#  verify_lm3_momocs.R  --  check that a LeafMachine3 Momocs export loads correctly in R
# =================================================================================================
#
#  WHAT IT CHECKS
#    LM3's Momocs Export module writes reports/Leaf_Momocs/ in every run. This script loads that
#    folder the two ways the guide (LM3_Momocs_Guide.html) describes and checks that both give the
#    same leaves:
#      Route 1  Momocs   import_jpg() on the leaf JPGs  +  momocs_fac.csv as the grouping table
#      Route 2  Momocs2  Momit::from_json() on momocs_outlines.json  (+ Momit::to_Momocs())
#    It prints one PASS / FAIL / SKIP line per check and a summary, saves a quick-look picture of
#    every outline next to your data (lm3_momocs_check.png), and exits with status 1 on any FAIL.
#
#  R REQUIREMENTS  (LeafMachine3 does not install R or any R package for you)
#    * R 4.1 or newer. Tested with R 4.1.2.
#    * Route 1 needs Momocs (CRAN, tested 1.5.0):
#          install.packages("Momocs")
#      On R older than 4.4, CRAN's current geomorph (a Momocs dependency) will not install. Install
#      the last versions that support older R first, then Momocs:
#          install.packages("remotes")
#          remotes::install_version("RRPP", version = "1.4.0")
#          remotes::install_version("geomorph", version = "4.0.6")
#          install.packages("Momocs")
#    * Route 2 needs Momocs2 and Momit (GitHub only, tested 0.1.0 of each; both need magick):
#          install.packages(c("remotes", "magick", "jsonlite"))
#          remotes::install_github("MomX/Momocs2")
#          remotes::install_github("MomX/Momit")
#    A route whose packages are missing is SKIPPED with an install hint; the other still runs.
#
#  HOW TO RUN
#    From a terminal:   Rscript verify_lm3_momocs.R  path/to/your_run/reports/Leaf_Momocs
#    With no path it checks the example shipped beside this script (example/Leaf_Momocs).
#    From RStudio:      set `leaf_momocs_dir` below, then Source the file.
# =================================================================================================

leaf_momocs_dir <- NULL   # e.g. "C:/LM3/runs/my_project/reports/Leaf_Momocs" when sourcing in RStudio

# ---- locate the folder ------------------------------------------------------------------------
script_dir <- function() {
  f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  if (length(f)) return(dirname(normalizePath(f)))
  of <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(of)) dirname(normalizePath(of)) else getwd()
}
args <- commandArgs(trailingOnly = TRUE)
if (length(args)) leaf_momocs_dir <- args[1]
if (is.null(leaf_momocs_dir)) leaf_momocs_dir <- file.path(script_dir(), "example", "Leaf_Momocs")
leaf_momocs_dir <- normalizePath(leaf_momocs_dir, mustWork = FALSE)

# ---- tiny reporting helpers (ASCII only, so any console shows them) ---------------------------
results <- c(PASS = 0, FAIL = 0, SKIP = 0)
report <- function(status, what, detail = "") {
  results[status] <<- results[status] + 1
  cat(sprintf("  %-4s  %s%s\n", status, what, if (nzchar(detail)) paste0("  (", detail, ")") else ""))
}
check <- function(ok, what, detail = "") report(if (isTRUE(ok)) "PASS" else "FAIL", what, detail)
have <- function(pkg) requireNamespace(pkg, quietly = TRUE)
signed_area <- function(m) { x <- m[, 1]; y <- m[, 2]; j <- c(2:length(x), 1); sum(x * y[j] - x[j] * y) / 2 }

cat("\nLM3 -> Momocs export check\n")
cat("Folder:", leaf_momocs_dir, "\n\n")

# ---- 1. the folder ----------------------------------------------------------------------------
cat("1. Files\n")
if (!dir.exists(leaf_momocs_dir)) {
  check(FALSE, "folder exists", "pass the path to a run's reports/Leaf_Momocs folder")
  quit(status = 1)
}
jpgs <- sort(list.files(leaf_momocs_dir, pattern = "\\.jpg$", full.names = TRUE))
fac_path <- file.path(leaf_momocs_dir, "momocs_fac.csv")
json_path <- file.path(leaf_momocs_dir, "momocs_outlines.json")
check(length(jpgs) > 0, "leaf images found", sprintf("%d JPG", length(jpgs)))
if (file.exists(fac_path)) check(TRUE, "momocs_fac.csv found") else
  report("SKIP", "momocs_fac.csv", "missing: turn on modules.momocs.write_fac_csv")
if (file.exists(json_path)) check(TRUE, "momocs_outlines.json found") else
  report("SKIP", "momocs_outlines.json", "missing: turn on modules.momocs.write_momit_json")
if (!length(jpgs)) quit(status = 1)
ids <- sub("\\.jpg$", "", basename(jpgs))

# The images must be a DARK leaf on WHITE with a white border. Momocs import_jpg reads other images
# without an error and traces the wrong region, so check the pixels themselves.
reader <- if (have("jpeg")) function(f) jpeg::readJPEG(f) else if (have("magick"))
  function(f) as.numeric(magick::image_data(magick::image_convert(magick::image_read(f), colorspace = "gray"), "gray"))[1, , ] / 255 else NULL
if (is.null(reader)) {
  report("SKIP", "image format", "needs the jpeg or magick package")
} else {
  bad <- character(0)
  for (f in jpgs) {
    im <- reader(f); if (length(dim(im)) == 3) im <- rowMeans(im, dims = 2)
    edge <- c(im[1, ], im[nrow(im), ], im[, 1], im[, ncol(im)])
    if (min(edge) < 0.5 || mean(im < 0.5) == 0) bad <- c(bad, basename(f))
  }
  check(!length(bad), "every image is a dark leaf on white with a white border",
        if (length(bad)) sprintf("%d wrong, e.g. %s; export with LM3's Momocs Export module", length(bad), bad[1]) else "")
}

# ---- 2. Route 1: Momocs ------------------------------------------------------------------------
cat("\n2. Route 1: Momocs import_jpg + momocs_fac.csv\n")
coo <- NULL
if (!have("Momocs")) {
  report("SKIP", "Momocs is not installed", 'install.packages("Momocs"); see the header of this file')
} else {
  suppressPackageStartupMessages(library(Momocs))
  options(verbose = FALSE)
  # import_jpg prints a line per file; keep the report readable
  invisible(capture.output(coo <- tryCatch(suppressMessages(import_jpg(jpgs)), error = function(e) e)))
  if (inherits(coo, "error")) {
    check(FALSE, "import_jpg reads every image", conditionMessage(coo)); coo <- NULL
  } else {
    check(length(coo) == length(jpgs), "import_jpg reads every image", sprintf("%d outlines", length(coo)))
    npts <- vapply(coo, nrow, 1L)
    check(all(npts >= 3), "every outline has at least 3 points", sprintf("smallest has %d", min(npts)))
    check(all(vapply(coo, signed_area, 1) < 0), "every outline runs clockwise")
    if (file.exists(fac_path)) {
      fac <- read.csv(fac_path, stringsAsFactors = FALSE)
      check(setequal(fac$id, names(coo)), "momocs_fac.csv has exactly one row per image",
            sprintf("%d rows, %d images", nrow(fac), length(coo)))
      O <- Out(coo, fac = fac[match(names(coo), fac$id), ])
      ef <- tryCatch({
        # Leaves smaller than ~20 px have fewer than 200 boundary points; interpolate before sampling.
        efourier(coo_scale(coo_center(coo_sample(coo_interpolate(O, 600), 200))), nb.h = 12, norm = FALSE)
      }, error = function(e) e)
      check(!inherits(ef, "error"), "efourier runs on the whole set",
            if (inherits(ef, "error")) conditionMessage(ef) else sprintf("%d leaves x %d coefficients", nrow(ef$coe), ncol(ef$coe)))
    }
  }
}

# ---- 3. Route 2: Momocs2 via Momit -------------------------------------------------------------
cat("\n3. Route 2: Momit::from_json -> Momocs2\n")
tb <- NULL
if (!file.exists(json_path)) {
  report("SKIP", "no momocs_outlines.json to read")
} else if (!have("Momit") || !have("Momocs2")) {
  report("SKIP", "Momit or Momocs2 is not installed", 'remotes::install_github(c("MomX/Momocs2", "MomX/Momit"))')
} else {
  tb <- tryCatch(Momit::from_json(json_path), error = function(e) e)
  if (inherits(tb, "error")) {
    check(FALSE, "from_json reads momocs_outlines.json", conditionMessage(tb)); tb <- NULL
  } else {
    check(nrow(tb) == length(jpgs), "one JSON row per leaf image", sprintf("%d rows, %d images", nrow(tb), length(jpgs)))
    check(setequal(tb$id, ids), "JSON ids match the image file names")
    check(inherits(tb$coo, "out"), "outlines load as a Momocs2 'out' column")
    xy <- lapply(tb$coo, function(m) unclass(m)[, 1:2, drop = FALSE])
    check(all(vapply(xy, signed_area, 1) < 0), "every JSON outline runs clockwise")
    check(all(vapply(xy, function(m) m[1, 2] == min(m[, 2]), TRUE)), "every JSON outline starts at its lowest point")
    e2 <- tryCatch(suppressMessages(Momocs2::eft(Momocs2::coo_sample(Momocs2::coo_scale(Momocs2::coo_center(tb)), 64), nb_h = 8)),
                   error = function(e) e)
    check(!inherits(e2, "error"), "Momocs2 eft runs", if (inherits(e2, "error")) conditionMessage(e2) else "")
    O2 <- tryCatch(Momit::to_Momocs(tb), error = function(e) e)
    check(!inherits(O2, "error") && length(O2$coo) == nrow(tb), "Momit::to_Momocs builds a legacy Out",
          if (inherits(O2, "error")) conditionMessage(O2) else "")
  }
}

# ---- 4. both routes describe the same leaves --------------------------------------------------
cat("\n4. Routes agree\n")
if (is.null(coo) || is.null(tb)) {
  report("SKIP", "needs both routes")
} else {
  # The two tracers place the edge differently: import_jpg about 1 px outside the JSON outline. So the
  # area difference divided by the perimeter is ~1 px for every leaf, from specks to whole blades.
  # A mirrored, misplaced or wrong outline lands far outside that band.
  band <- vapply(ids, function(i) {
    a <- unclass(tb$coo[[match(i, tb$id)]])[, 1:2]
    (abs(signed_area(coo[[i]])) - abs(signed_area(a))) / Momocs::coo_perim(a)
  }, 1)
  check(all(band > 0 & band < 2), "both routes trace the same outline (within 2 px)",
        sprintf("edge offset %.2f to %.2f px over %d leaves", min(band), max(band), length(band)))
}

# ---- quick look ------------------------------------------------------------------------------
if (!is.null(coo)) {
  png_path <- file.path(leaf_momocs_dir, "lm3_momocs_check.png")
  ok <- tryCatch({
    png(png_path, width = 1400, height = 900, res = 130)
    panel(Out(coo))
    dev.off(); TRUE
  }, error = function(e) { try(dev.off(), silent = TRUE); FALSE })
  if (ok) cat("\nQuick look saved:", png_path, "\n")
}

cat(sprintf("\nSummary: %d PASS, %d FAIL, %d SKIP\n", results["PASS"], results["FAIL"], results["SKIP"]))
if (results["FAIL"] > 0) {
  cat("Something did not load as expected. Check the FAIL lines above.\n")
  if (!interactive()) quit(status = 1)
} else {
  cat("Your LM3 export is ready for Momocs", if (!is.null(tb)) " and Momocs2" else "", ".\n", sep = "")
}
