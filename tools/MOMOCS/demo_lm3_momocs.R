# =================================================================================================
#  demo_lm3_momocs.R  --  a worked tour of LeafMachine3 output in Momocs and Momocs2
# =================================================================================================
#
#  Every step of LM3_Momocs_Guide.html, in order, as runnable code. Run it on the example shipped
#  beside this file, or point it at your own run. It saves one figure per step.
#
#  R REQUIREMENTS  (LeafMachine3 does not install R or any R package for you)
#    * R 4.1 or newer. Tested with R 4.1.2.
#    * Momocs (CRAN, tested 1.5.0), ggplot2, patchwork:
#          install.packages(c("Momocs", "ggplot2", "patchwork"))
#      On R older than 4.4, install these two first (CRAN's current geomorph needs R 4.4):
#          install.packages("remotes")
#          remotes::install_version("RRPP", version = "1.4.0")
#          remotes::install_version("geomorph", version = "4.0.6")
#    * For the Momocs2 steps: Momocs2 and Momit from GitHub (tested 0.1.0 of each):
#          install.packages(c("remotes", "magick", "jsonlite"))
#          remotes::install_github("MomX/Momocs2")
#          remotes::install_github("MomX/Momit")
#      Without them those steps are skipped.
#
#  HOW TO RUN
#    Rscript demo_lm3_momocs.R                                    # the example, figures -> ./lm3_momocs_demo
#    Rscript demo_lm3_momocs.R path/to/reports/Leaf_Momocs  out   # your run, figures -> out/
#    The landmark steps read <run>/reports/Data/landmarks.csv, the sibling of Leaf_Momocs/.
#
#  ABOUT THE GROUPS
#    One sheet has no biological groups, so the group steps split the leaves into three size
#    classes by area. That only shows the methods running; use your own factors (species, site,
#    year...) by joining your data to momocs_fac.csv on `image_stem`.
# =================================================================================================

suppressPackageStartupMessages({ library(Momocs); library(ggplot2) })
options(verbose = FALSE)
set.seed(1)

script_dir <- function() {
  f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  if (length(f)) dirname(normalizePath(f)) else getwd()
}
args <- commandArgs(trailingOnly = TRUE)
leaf_dir <- if (length(args) >= 1) args[1] else file.path(script_dir(), "example", "Leaf_Momocs")
out_dir <- if (length(args) >= 2) args[2] else "lm3_momocs_demo"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
landmarks_csv <- file.path(dirname(leaf_dir), "Data", "landmarks.csv")
has_momocs2 <- requireNamespace("Momocs2", quietly = TRUE) && requireNamespace("Momit", quietly = TRUE)

fig <- function(name, w = 1400, h = 900, expr) {   # base-graphics figure -> PNG
  png(file.path(out_dir, name), width = w, height = h, res = 150); on.exit(dev.off()); force(expr)
}
gg <- function(name, p, w = 9.3, h = 6) ggsave(file.path(out_dir, name), p, width = w, height = h, dpi = 150, bg = "white")
step <- function(label) cat("\n--", label, "\n")

# ---- Step 1. Load LM3's export ----------------------------------------------------------------
step("1. Load the leaf images and the grouping table")
jpgs <- list.files(leaf_dir, pattern = "\\.jpg$", full.names = TRUE)
invisible(capture.output(coo <- import_jpg(jpgs)))          # one outline per JPG, named by file
fac <- read.csv(file.path(leaf_dir, "momocs_fac.csv"), stringsAsFactors = FALSE)
fac <- fac[match(names(coo), fac$id), ]                       # same order as the outlines
fac$size_class <- cut(rank(sapply(coo, coo_area)), breaks = 3,
                      labels = c("small", "medium", "large"))                # demo grouping only
leaves <- Out(coo, fac = fac)
leaves
fig("01_panel.png", 1400, 700, panel(leaves, fac = "size_class"))   # names = TRUE labels each leaf

# ---- Step 2. Normalize ------------------------------------------------------------------------
step("2. Center, scale and resample")
# LM3 already turned every leaf tip up, so no alignment step is needed. coo_sample needs at least
# 200 points per outline; a leaf only ~20 px across has fewer, so drop such specks first
# (leaves <- filter(leaves, outline_points >= 200)) or use coo_interpolate on them.
norm <- leaves %>% coo_center() %>% coo_scale() %>% coo_sample(200)
fig("02_stack.png", 1000, 1000, stack(norm, title = "13 Ulmus americana leaves, tip up"))

# ---- Step 3. Shape descriptors ----------------------------------------------------------------
step("3. Scalar shape descriptors")
desc <- data.frame(leaf = sub(".*__", "", names(leaves)),
                   area_px = sapply(leaves$coo, coo_area), perimeter_px = sapply(leaves$coo, coo_perim),
                   elongation = sapply(leaves$coo, coo_elongation), circularity = sapply(leaves$coo, coo_circularity),
                   convexity = sapply(leaves$coo, coo_convexity), solidity = sapply(leaves$coo, coo_solidity))
desc[-1] <- lapply(desc[-1], signif, 4)
write.csv(desc[order(desc$area_px), ], file.path(out_dir, "descriptors.csv"), row.names = FALSE)
print(head(desc[order(desc$area_px), ]))

# ---- Step 4. How many harmonics? ----------------------------------------------------------------
step("4. Calibrate the number of harmonics")
pdf(NULL); hp <- calibrate_harmonicpower_efourier(norm, nb.h = 30); dev.off()
print(hp$minh)
gg("03_harmonic_power.png", hp$gg + ggtitle("Cumulative harmonic power"))
big <- which.max(sapply(leaves$coo, coo_area))
gg("04_reconstructions.png", calibrate_reconstructions_efourier(norm, id = big, range = c(1, 2, 4, 6, 8, 12, 16, 20, 30)) +
     ggtitle("Elliptic Fourier reconstructions (black = the real outline)"), 9.3, 7)

# ---- Step 5. Elliptic Fourier analysis ----------------------------------------------------------
step("5. Elliptic Fourier analysis")
# norm = FALSE keeps LM3's tip-up orientation. The default (TRUE) re-aligns every leaf on its first
# ellipse, which turns these leaves on their side and can flip some 180 degrees.
ef <- efourier(norm, nb.h = 12, norm = FALSE)
ef

# ---- Step 6. Morphospace --------------------------------------------------------------------------
step("6. PCA and morphospace")
pc <- PCA(ef)
fig("05_pca.png", 1400, 1000, plot_PCA(pc, ~size_class, chull = TRUE, morphospace_position = "range",
                                       title = "Morphospace of 12-harmonic outlines"))
gg("06_pc_contrib.png", PCcontrib(pc, nax = 1:3, sd.r = c(-2, -1, 0, 1, 2))$gg + ggtitle("Shape along PC1-PC3 (-2 to +2 SD)"), 9.3, 5)
cat("PC1-PC3 variance (%):", round(100 * pc$eig[1:3] / sum(pc$eig), 1), "\n")

# ---- Step 7. Groups -------------------------------------------------------------------------------
step("7. Mean shapes, tests and clusters")
ms <- MSHAPES(ef, ~size_class)
fig("07_mshapes.png", 1400, 900, plot_MSHAPES(ms))
print(MANOVA(pc, ~size_class, retain = 3))
gg("08_clust.png", CLUST(ef, ~size_class, hclust_method = "ward.D2", k = 3))

# ---- Step 8. Momocs2 ------------------------------------------------------------------------------
step("8. The same leaves in Momocs2")
if (has_momocs2) {
  tb <- Momit::from_json(file.path(leaf_dir, "momocs_outlines.json"))   # a Momocs2 table
  print(tb)
  tb2 <- tb %>% Momocs2::coo_center() %>% Momocs2::coo_scale() %>% Momocs2::coo_sample(200)
  fig("09_momocs2_mosaic.png", 1400, 700, Momocs2::mosaic(tb2, relative = FALSE, lwd = 2))
  png(file.path(out_dir, "10_momocs2_reconstruction.png"), width = 1400, height = 900, res = 150)
  plot(Momocs2::eft_calibrate_reconstruction(tb2, id = big)); dev.off()
  print(Momocs2::eft(tb2, nb_h = 12))
  legacy <- Momit::to_Momocs(tb)                                       # and back to a legacy Out
  print(legacy)
} else {
  cat("Momocs2 / Momit not installed: skipped (see the header of this file).\n")
}

# ---- Step 9. LM3 landmarks as a Momocs landmark set ----------------------------------------------
step("9. LM3's 31 keypoints in Momocs")
if (file.exists(landmarks_csv)) {
  lm <- read.csv(landmarks_csv, stringsAsFactors = FALSE)
  lm <- lm[lm$instance_index == 0, ]
  lm$y <- -lm$y                                                         # image y runs down; Momocs y runs up
  lm$box <- sub(".*__", "", lm$crop_file_token)                         # joins to the image names
  keep <- intersect(unique(lm$box), sub(".*__", "", names(leaves)))
  L <- lapply(keep, function(b) { d <- lm[lm$box == b, ]; as.matrix(d[order(d$kpt_index), c("x", "y")]) })
  names(L) <- keep
  links <- rbind(cbind(c(0, 4:18), c(4:18, 22)), cbind(22:27, 23:28),    # midvein, petiole
                 cbind(c(1, 2), c(2, 3)), cbind(c(19, 20), c(20, 21)), c(29, 30)) + 1   # apex, base, width
  invisible(capture.output(Lp <- fgProcrustes(Ldk(L, links = links), tol = 1e-8)))   # quiet the per-iteration log
  m0 <- MSHAPES(Lp); v <- m0[23, ] - m0[1, ]                            # tip -> lamina_base
  Lp <- coo_rotate(Lp, -pi / 2 - atan2(v[2], v[1]))                     # tip up, for display
  fig("11_landmarks.png", 1400, 700, {
    par(mfrow = c(1, 2))
    stack(Ldk(L) %>% coo_center() %>% coo_scale(), title = "Keypoints as found on the sheet")
    stack(Lp, title = "After Procrustes", ldk_links = TRUE, ldk_cex = 0.6)
  })
  mean_conf <- MSHAPES(Lp)
  fig("12_landmark_mean.png", 900, 1100, {
    plot(mean_conf, asp = 1, type = "n", axes = FALSE, xlab = "", ylab = "", main = "Mean keypoint configuration")
    ldk_links(mean_conf, links, col = "#2d6a4c", lwd = 2); points(mean_conf, pch = 21, bg = "white", cex = 1.3)
  })
  pcl <- PCA(Lp)
  cat("Landmark PC1-PC3 variance (%):", round(100 * pcl$eig[1:3] / sum(pcl$eig), 1), "\n")
} else {
  cat("No", landmarks_csv, "- skipped.\n")
}

cat("\nFigures written to", normalizePath(out_dir), "\n")
