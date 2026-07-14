#!/usr/bin/env Rscript
# tools/r/parity_generator.r
#
# Generates glmnet parity fixtures for quarreLM.
# Run from the repo root:  Rscript tools/r/parity_generator.r

suppressMessages(library(glmnet))

DATA_DIR <- "benchmarks/data"
OUT_DIR <- "py-quarrelm/tests/fixtures"
STANDARDIZE <- FALSE
INTERCEPT <- FALSE
THRESH <- 1e-12 # glmnet default (1e-7) is looser than the test tolerance
MAXIT <- 1e7

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ---- datasets ----------------------------------------------------------

load_prostate <- function() {
    df <- read.table(file.path(DATA_DIR, "prostate.data.gz"), header = TRUE)
    df$train <- NULL
    list(
        name = "prostate",
        x = as.matrix(df[, setdiff(names(df), "lpsa")]),
        y = df$lpsa
    )
}

load_diabetes <- function() {
    df <- read.table(file.path(DATA_DIR, "diabetes.data.gz"), header = TRUE)
    list(
        name = "diabetes",
        x = as.matrix(df[, setdiff(names(df), "Y")]),
        y = df$Y
    )
}

datasets <- list(load_prostate(), load_diabetes())

# ---- constraint presets -------------------------------------------------
# pf_mixed sums to p on purpose: glmnet rescales penalty.factor to sum to
# nvars internally, so a sum-of-p vector makes that a no-op and the fixture
# is valid both before and after quarreLM implements the rescale.

make_presets <- function(p) {
    list(
        baseline = list(
            pf = rep(1, p),
            lower = rep(-Inf, p), upper = rep(Inf, p)
        ),
        pf_mixed = list(
            pf = {
                v <- rep(1, p)
                v[1] <- 0.5
                v[2] <- 1.5
                v
            },
            lower = rep(-Inf, p), upper = rep(Inf, p)
        ),
        nonneg2 = list(
            pf = rep(1, p),
            lower = {
                v <- rep(-Inf, p)
                v[1] <- 0
                v[2] <- 0
                v
            },
            upper = rep(Inf, p)
        )
    )
}

ALPHAS <- c(1.0, 0.5, 0.05)
LAMBDA_FRACS <- c(0.5, 0.1, 0.01) # fractions of lambda_max: meaningful
# sparsity on any data scale

fit_rows <- list()
path_rows <- list()
preset_rows <- list()

for (ds in datasets) {
    n <- nrow(ds$x)
    p <- ncol(ds$x)
    presets <- make_presets(p)


    for (preset_name in names(presets)) {
        ps <- presets[[preset_name]]

        preset_rows[[length(preset_rows) + 1]] <- data.frame(
            dataset = ds$name,
            preset  = preset_name,
            feature = colnames(ds$x), # aligns pf/lower/upper to features by position
            pf      = ps$pf,
            lower   = ps$lower,
            upper   = ps$upper
        )


        for (alpha in ALPHAS) {
            # target lambdas: explicit decreasing vector handed to glmnet, so
            # coefficients are read off exactly - NO interpolation via coef(s=)
            lmax <- max(abs(crossprod(ds$x, ds$y))) / (n * alpha)
            lambdas <- lmax * LAMBDA_FRACS # already decreasing

            fit <- glmnet(ds$x, ds$y,
                alpha = alpha, lambda = lambdas,
                standardize = STANDARDIZE, intercept = INTERCEPT,
                penalty.factor = ps$pf,
                lower.limits = ps$lower, upper.limits = ps$upper,
                control = list(thresh = THRESH, maxit = MAXIT)
            )

            B <- as.matrix(coef(fit)) # (p+1) x length(lambdas), row 1 = intercept
            for (k in seq_along(fit$lambda)) {
                fit_rows[[length(fit_rows) + 1]] <- data.frame(
                    dataset = ds$name, preset = preset_name,
                    alpha = alpha, lambda = fit$lambda[k],
                    feature = rownames(B), coef = B[, k]
                )
            }
        }
    }

    # ---- path fixtures (glmnet picks its own sequence) --------------------
    for (alpha in c(1.0, 0.5)) {
        fit <- glmnet(ds$x, ds$y,
            alpha = alpha, nlambda = 20,
            standardize = STANDARDIZE, intercept = INTERCEPT,
            control = list(thresh = THRESH, maxit = MAXIT)
        )
        B <- as.matrix(coef(fit))
        for (k in seq_along(fit$lambda)) { # may be < 20: glmnet stops early
            path_rows[[length(path_rows) + 1]] <- data.frame(
                dataset = ds$name, alpha = alpha,
                lambda_index = k, lambda = fit$lambda[k],
                feature = rownames(B), coef = B[, k]
            )
        }
    }
}
write.csv(do.call(rbind, preset_rows),
    file.path(OUT_DIR, "parity_presets.csv"),
    row.names = FALSE
)
write.csv(do.call(rbind, fit_rows),
    file.path(OUT_DIR, "parity_fits.csv"),
    row.names = FALSE
)
write.csv(do.call(rbind, path_rows),
    file.path(OUT_DIR, "parity_paths.csv"),
    row.names = FALSE
)
write.csv(
    data.frame(
        generated = format(Sys.time(), "%Y-%m-%d"),
        r_version = R.version.string,
        glmnet = as.character(packageVersion("glmnet")),
        standardize = STANDARDIZE, intercept = INTERCEPT,
        thresh = THRESH, maxit = MAXIT
    ),
    file.path(OUT_DIR, "parity_meta.csv"),
    row.names = FALSE
)

cat(sprintf(
    "wrote %d fit rows, %d path rows to %s\n",
    length(fit_rows), length(path_rows), OUT_DIR
))
