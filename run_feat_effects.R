################################################################################
#                   Feature Effect plots (PDP + ICE, ALE)
#
#     This script runs PDP + ICE and ALE on the 
#     given model, dataset and synthesizer. It saves the results in the
#    `results` folder.
################################################################################
library(ranger)
library(xgboost)
library(cli)
library(cowplot)
library(iml)
library(Metrics)
library(data.table)
library(doParallel)
library(parallel)
library(ggplot2)

cli_div(theme = list(span.emph = list(color = "#3c77b9")))

# Set seed for reproducibility
set.seed(42)

# Manage number of cores and RAM
n_threads <- 200L

options(future.globals.maxSize = 25000 * 1024^2)
Sys.setenv(R_RANGER_NUM_THREADS = n_threads)
Sys.setenv(OMP_THREAD_LIMIT = n_threads)

# Global arguments for the PFI method
NUM_SAMPLES <- 1000
NUM_SAMPLES_ICE <- 100
NUM_GRID_POINTS <- 200

# Define global arguments
filter_df <- data.table(
  dataset_name = c("adult_complete", "nursery"),
  model_name = c("xgboost"),
  syn_name = c("TabSyn", "CTGAN"),
  run_model = c(2, 8)
)

# Load utility methods and create dirs -----------------------------------------

# Load global utility functions
source("utils.R")

################################################################################
#                Main script for running PDP + ICE and ALE
################################################################################

# Create data.frame for all settings -------------------------------------------
cli_progress_step("Creating settings data.frame for running PDP, ICE and ALE")

# Find all available datasets and trained model
model_names <- list.files("./models/")
model_names <- model_names[file.info(paste0("./models/", model_names))$isdir]
df <- rbindlist(lapply(model_names, function(model_name) {
  all_files <- list.files(paste0("./models/", model_name))
  args <- strsplit(all_files, "--")
  data.frame(
    dataset_name = unlist(lapply(args, function(x) x[1])),
    syn_name = unlist(lapply(args, function(x) x[2])),
    run_model = as.integer(unlist(lapply(args, function(x) gsub(".rds", "", x[3])))),
    pth = paste0("./models/", model_name, "/", all_files),
    model_name = model_name,
    file_name = gsub(".rds", "", all_files)
  )
}))

# Filter settings (as defined in the global settings)
df <- df[filter_df, on = c("dataset_name", "model_name", "syn_name", "run_model")]

# Running Feature Effect plots (PDP + ICE, ALE) --------------------------------
cli_h1("Running Feature Effect plots (PDP + ICE, ALE)")

res <- lapply(seq_len(nrow(df)), function(i) {
  
  # Set seed
  set.seed(42)
  
  cli_progress_step(paste0(
    "[{i}/{nrow(df)}] ", 
    "Dataset: {.emph {df$dataset_name[i]}} --- ", 
    "Synthesizer: {.emph {df$syn_name[i]}} --- ",
    "Run: {.emph {df$run_model[i]}}"))
  
  # Load model and data
  if (df$model_name[i] == "xgboost") {
    model <- xgboost::xgb.load(df$pth[i])
    xgboost::xgb.parameters(model) <- list(nthread = n_threads)
  } else {
    model <- readRDS(df$pth[i])
  }
  data <- load_data(df$dataset_name[i], df$syn_name[i])
  
  # Select relevant dataset
  data <- data[[df$file_name[i]]]
  data <- data[data$train == 0, -ncol(data)]
  
  # Get predict function
  pred_fun <- get_predict_fun(df$model_name[i])
  
  # Get sample of data
  num_samples <- min(NUM_SAMPLES, nrow(data))
  num_samples_ice <- min(NUM_SAMPLES_ICE, num_samples)
  df_test <- data[sample(1:nrow(data), num_samples), ]
  
  # Get predictions
  preds <- pred_fun(model, df_test[, -ncol(df_test)])
  preds <- data.table(preds, id = seq_len(nrow(df_test)))
  preds <- preds[order(-rank(preds)), ]
  
  # Create predictor
  predictor <- Predictor$new(model, data = df_test, y = "real", class = "Real",
                             predict.function = pred_fun)
  
  # ICE  -----------------------------------------------------------------------
  feat_effect <- FeatureEffects$new(predictor, method = "ice",
                                    features = names(df_test)[1:(ncol(df_test) - 1)],
                                    grid.size = NUM_GRID_POINTS)
  results <- lapply(feat_effect$results, function(x) {
    x$feat_type <- if (is.numeric(x$.borders)) "numeric" else "categorical"
    x
  })
  # Combine results
  res <- rbindlist(results)
  res <- merge(
    res, 
    data.frame(.id = seq_len(nrow(df_test)), real = df_test[["real"]]), 
    by = ".id")
  
  # ICE results
  top_idx <- preds$id[seq_len(num_samples_ice %/% 3)]
  mid_idx <- preds$id[seq(num_samples %/% 2 - num_samples_ice %/% 6, 
                          num_samples %/% 2 + num_samples_ice %/% 6)]
  low_idx <- preds$id[seq(num_samples - num_samples_ice %/% 3 + 1, num_samples)]
  idx <- c(top_idx, mid_idx, low_idx)
  
  res_ice <- res[.id %in% idx, ]
  
  # PDP ------------------------------------------------------------------------
  res_pdp <- rbind(
    cbind(res[, .(.value = mean(.value)), 
                  by = c(".borders", ".type", ".feature", "feat_type")], real = "both"),
    cbind(res[real == "Real", .(.value = mean(.value)), 
                  by = c(".borders", ".type", ".feature", "feat_type")], real = "Real"),
    cbind(res[real != "Real", .(.value = mean(.value)), 
                  by = c(".borders", ".type", ".feature", "feat_type")], real = "Synthetic")
  )
  res_pdp$`.id` <- NA
  res_pdp$`.type` <- "pdp"
  
  
  # ALE ------------------------------------------------------------------------
  feat_effect <- FeatureEffects$new(predictor, method = "ale",
                                    features = names(df_test)[1:(ncol(df_test) - 1)],
                                    grid.size = NUM_GRID_POINTS)
  results <- lapply(feat_effect$results, function(x) {
    x$feat_type <- if (is.numeric(x$.value)) "numeric" else "categorical"
    x
  })
  res_ale <- rbindlist(results)
  res_ale$`.id` <- NA
  res_ale$real <- NA
  
  # ----------------------------------------------------------------------------
  # Summarize results
  res <- rbind(res_ice, res_ale, res_pdp)
  setnames(res, c(".borders", ".value", ".type", ".id", ".feature"),
           c("gridpoint", "value", "method", "id", "feature"))
  
  # Summarize and return results
  suppressWarnings({
    df_rug <- 
      cbind(
        melt(data.table(df_test),
             measure.vars = names(df_test)[-ncol(df_test)], id.vars = "real"),
        dataset_name = df$dataset_name[i],
        syn_name = df$syn_name[i],
        run_model = df$run_model[i],
        model_name = df$model_name[i]
      )
  })
  
  list(
    cbind(res,
        dataset_name = df$dataset_name[i],
        syn_name = df$syn_name[i],
        run_model = df$run_model[i],
        model_name = df$model_name[i]),
    df_rug
  )
})

# Combine results
rugs <- do.call(rbind, lapply(res, `[[`, 2))
res <- do.call(rbind, lapply(res, `[[`, 1))

# Save results
cli_progress_step("Saving results")
if (!dir.exists(paste0("./results/Q2"))) dir.create(paste0("./results/Q2"), recursive = TRUE)
saveRDS(res, "./results/Q2/feat_effects.rds")
saveRDS(rugs, "./results/Q2/feat_effects_rugs.rds")

cli_progress_done("Done!")
