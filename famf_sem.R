# famf_sem.R -- Metadata-only Feature-Augmented Method Factor (FAMF-SEM)
# lavaan reference implementation.
#
# Pipeline: baseline trait-only CFA -> standardized residual correlations ->
# leading-eigenvector signal -> ridge regression on item metadata
# (intercept unpenalized, lambda by GCV) -> normalized fixed loading
# PATTERN for a single orthogonal method factor whose variance phi is
# freely estimated.
#
# Requires: lavaan (install.packages("lavaan"))

library(lavaan)

## ---------------------------------------------------------------------
## 1. Calibration utilities
## ---------------------------------------------------------------------

encode_metadata <- function(meta, binary = c("Reversed", "Polarity")) {
  Z <- sapply(names(meta), function(cn) {
    x <- as.numeric(meta[[cn]])
    if (cn %in% binary) x - mean(x)
    else if (sd(x) > 1e-12) (x - mean(x)) / sd(x)
    else x * 0
  })
  colnames(Z) <- names(meta)
  Z
}

residual_corr <- function(fit, data, items) {
  R_obs <- cor(data[, items])
  S_imp <- lavInspect(fit, "implied")$cov[items, items]
  d <- sqrt(diag(S_imp))
  R_imp <- S_imp / tcrossprod(d)
  R_obs - R_imp
}

eigen_signal <- function(R_res) {
  S <- (R_res + t(R_res)) / 2
  e <- eigen(S, symmetric = TRUE)
  v <- e$vectors[, 1] * sqrt(max(e$values[1], 0))
  if (sum(v) < 0) v <- -v          # orientation convention
  v
}

absmean_signal <- function(R_res) {   # magnitude-only fallback
  k <- nrow(R_res); A <- abs(R_res); diag(A) <- 0
  rowSums(A) / (k - 1)
}

ridge_gcv <- function(Z, m, lambdas = c(.01, .03, .1, .3, 1, 3, 10, 30)) {
  k <- nrow(Z); X <- cbind(1, Z)
  D <- diag(ncol(X)); D[1, 1] <- 0   # intercept unpenalized
  XtX <- crossprod(X); Xtm <- crossprod(X, m)
  best <- NULL
  for (lam in lambdas) {
    beta <- solve(XtX + lam * D, Xtm)
    H <- X %*% solve(XtX + lam * D, t(X))
    resid <- m - X %*% beta
    gcv <- sum(resid^2) / k / (1 - sum(diag(H)) / k)^2
    if (is.null(best) || gcv < best$gcv)
      best <- list(lam = lam, gcv = gcv, beta = beta)
  }
  w_raw <- as.numeric(cbind(1, Z) %*% best$beta)
  r2 <- 1 - sum((m - w_raw)^2) / sum((m - mean(m))^2)
  list(gamma0 = best$beta[1], gamma = best$beta[-1],
       w_raw = w_raw, lambda = best$lam, r2 = r2)
}

normalize_pattern <- function(w_raw) {
  k <- length(w_raw); ss <- sum(w_raw^2)
  if (ss < 1e-12) return(rep(1, k))  # degenerate -> CLF pattern
  w_raw * sqrt(k / ss)
}

## ---------------------------------------------------------------------
## 2. Model builders
## ---------------------------------------------------------------------

famf_model <- function(trait_model, items, w, traits) {
  m_line <- paste0("M =~ ",
                   paste0(sprintf("%.5f", w), "*", items, collapse = " + "))
  orth <- paste0("M ~~ 0*", traits, collapse = "\n")
  paste(trait_model, m_line, "M ~~ M", orth, sep = "\n")  # phi free
}

## ---------------------------------------------------------------------
## 3. Worked example: psych::bfi (SAPA), 25 items, 5 factors
##    (any data.frame of items + metadata works the same way)
## ---------------------------------------------------------------------

run_example <- function() {
  if (!requireNamespace("psychTools", quietly = TRUE) &&
      !requireNamespace("psych", quietly = TRUE))
    stop("install.packages('psychTools') for the bfi data")
  bfi <- if (requireNamespace("psychTools", quietly = TRUE))
    psychTools::bfi else psych::bfi
  items <- c(paste0("A", 1:5), paste0("C", 1:5), paste0("E", 1:5),
             paste0("N", 1:5), paste0("O", 1:5))
  dat <- na.omit(bfi[, items])
  # standard scoring: reflect reverse-keyed items (6-point scale)
  keying <- c(A1=-1,A2=1,A3=1,A4=1,A5=1, C1=1,C2=1,C3=1,C4=-1,C5=-1,
              E1=-1,E2=-1,E3=1,E4=1,E5=1, N1=-1,N2=-1,N3=-1,N4=-1,N5=-1,
              O1=1,O2=-1,O3=1,O4=1,O5=-1)
  # N items all keyed the same direction; reflect so that higher = more stable
  for (it in items) if (keying[it] < 0) dat[[it]] <- 7 - dat[[it]]

  # item metadata: keying direction and item length (words);
  # SAPA administers items in randomized order, so order/page are omitted.
  lengths <- c(A1=7,A2=5,A3=6,A4=2,A5=5, C1=6,C2=5,C3=6,C4=7,C5=3,
               E1=5,E2=6,E3=6,E4=3,E5=2, N1=3,N2=3,N3=4,N4=3,N5=2,
               O1=5,O2=4,O3=7,O4=5,O5=7)
  meta <- data.frame(Polarity = keying[items], Length = lengths[items])

  trait_model <- "
    A =~ A1 + A2 + A3 + A4 + A5
    C =~ C1 + C2 + C3 + C4 + C5
    E =~ E1 + E2 + E3 + E4 + E5
    N =~ N1 + N2 + N3 + N4 + N5
    O =~ O1 + O2 + O3 + O4 + O5"

  base <- cfa(trait_model, data = dat, std.lv = TRUE)
  R_res <- residual_corr(base, dat, items)
  m <- eigen_signal(R_res)
  Z <- encode_metadata(meta)
  cal <- ridge_gcv(Z, m)
  w <- normalize_pattern(cal$w_raw)
  cat(sprintf("GCV lambda = %.2f,  R2(m, m_hat) = %.3f\n", cal$lambda, cal$r2))
  print(round(setNames(cal$gamma, colnames(Z)), 4))

  famf <- cfa(famf_model(trait_model, items, w, c("A","C","E","N","O")),
              data = dat, std.lv = TRUE)
  phi <- lavInspect(famf, "est")$psi["M", "M"]
  cat(sprintf("phi (method variance) = %.4f\n", phi))
  cat("\nFactor correlations, baseline vs FAMF:\n")
  print(round(lavInspect(base, "cor.lv"), 3))
  print(round(lavInspect(famf, "cor.lv")[1:5, 1:5], 3))
  invisible(list(base = base, famf = famf, w = w, cal = cal))
}

## run_example()
