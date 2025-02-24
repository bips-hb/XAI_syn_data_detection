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
n_threads <- 100L

options(future.globals.maxSize = 25000 * 1024^2)
Sys.setenv(R_RANGER_NUM_THREADS = n_threads)
Sys.setenv(OMP_THREAD_LIMIT = n_threads)

# Global arguments for the PFI method
NUM_SAMPLES <- 2000 # Number of samples for the calculation

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
  df_test <- data[sample(1:nrow(data), NUM_SAMPLES), ]
  
  # Create predictor object
  predictor <- Predictor$new(model, data = df_test, y = "real", class = "Real",
                             predict.function = pred_fun)
  
  # ICE + ALE ------------------------------------------------------------------
  # ICE
  feat_effect <- FeatureEffects$new(predictor, method = "ice",
                                    features = names(df_test)[1:(ncol(df_test) - 1)],
                                    grid.size = 1000)
  # Combine results
  res_ice <- rbindlist(feat_effect$results)
  res_ice <- merge(
    res_ice, 
    data.frame(.id = seq_len(nrow(df_test)), real = df_test[["real"]]), 
    by = ".id")
  
  # PDP
  res_pdp <- res_ice[, .(.value = mean(.value)), by = c(".borders", ".type", ".feature")]
  res_pdp$`.id` <- NA
  res_pdp$real <- NA
  res_pdp$`.type` <- "pdp"
  
  
  # ALE
  feat_effect <- FeatureEffects$new(predictor, method = "ale",
                                    features = names(df_test)[1:(ncol(df_test) - 1)],
                                    grid.size = 1000)
  res_ale <- rbindlist(feat_effect$results)
  res_ale$`.id` <- NA
  res_ale$real <- NA
  
  # Summarize results
  res <- rbind(res_ice, res_ale, res_pdp)
  setnames(res, c(".borders", ".value", ".type", ".id", ".feature"),
           c("gridpoint", "value", "method", "id", "feature"))
  # ----------------------------------------------------------------------------
  
  # Summarize and return results
  cbind(res,
        dataset_name = df$dataset_name[i],
        syn_name = df$syn_name[i],
        run_model = df$run_model[i],
        model_name = df$model_name[i])
})

# Combine results
res <- do.call(rbind, res)

# Save results
cli_progress_step("Saving results")
if (!dir.exists(paste0("./results/Q2"))) dir.create(paste0("./results/Q2"), recursive = TRUE)
saveRDS(res, "./results/Q2/feat_effects.rds")

cli_progress_done("Done!")
