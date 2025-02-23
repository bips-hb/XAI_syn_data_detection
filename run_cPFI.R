################################################################################
#              Conditional Permutation Feature Importance (cPFI)
#
#     This script runs conditional Permutation Feature Importance (PFI) on the 
#     given model, dataset and synthesizer. It saves the results in the
#    `results` folder.
################################################################################
library(ranger)
library(xgboost)
library(cli)
library(cowplot)
library(Metrics)
library(data.table)
library(doParallel)
library(arf)
library(parallel)
library(ggplot2)
cli_div(theme = list(span.emph = list(color = "#3c77b9")))

# Set seed for reproducibility
set.seed(42)

# Manage number of cores and RAM
# Note: The total number of cores used will be 'mc.cores * n_threads'
n_threads <- 100L
mc.cores <-10L

options(future.globals.maxSize = 25000 * 1024^2)
Sys.setenv(R_RANGER_NUM_THREADS = n_threads)
Sys.setenv(OMP_THREAD_LIMIT = n_threads)
options(mc.cores = mc.cores)

# Global arguments for the cPFI method
NUM_SAMPLES <- Inf # Number of samples for the calculation
train_or_test <- 0 # 0 for test, 1 for train data

# Define global arguments
filter_df <- data.table(
  dataset_name = rev(c("adult_complete", "nursery")),
  model_name = c("xgboost"),
  syn_name = rev(c("TabSyn", "CTGAN")),
  run_model = c(1:10, 1:10)
)

# Load utility methods and create dirs -----------------------------------------

# Load global utility functions
source("utils.R")

################################################################################
#                     Main script for running cPFI
################################################################################

# Create data.frame for all settings -------------------------------------------
cli_progress_step("Creating settings data.frame for running cPFI")

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


# Running PFI ------------------------------------------------------------------
cli_h1("Running conditional Permutation Feature Importance (cPFI)")

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
  
  # Get predict function
  pred_fun <- get_predict_fun(df$model_name[i])
  
  # Main part for cPFI ---------------------------------------------------------
  
  # Get all data
  full_data <- data[, -ncol(data)]
  full_x <- data.table(full_data[, -ncol(full_data)])
  
  # Get test data
  data_test <- data[data$train == train_or_test, -ncol(data)]
  data_test <- data_test[sample.int(nrow(data_test), min(NUM_SAMPLES, nrow(data_test))), ]
  x <- data.table(data_test[, -ncol(data_test)])
  y <- as.numeric(data_test$real) - 1
  
  # Calculate full loss
  full_loss <- logLoss(y, pred_fun(model, newdata = x))
  
  # Generate conditional samples
  cli_progress_step("------- Fitting ARF")
  doParallel::registerDoParallel(cores = n_threads)
  arf <- adversarial_rf(full_x, parallel = TRUE, min_node_size = 15L, num_trees = 200L, verbose = FALSE)
  cli_progress_step("------- Estimating ARF density")
  psi <- forde(arf, full_x,  parallel = TRUE, finite_bounds = "local", epsilon = 1e-9)
  cli_progress_step("------- Sampling conditional values")
  x_tilde <- lapply(seq_len(ncol(x)), function(k) {
    evi <- x
    evi[[k]] <- NA
    arf::forge(psi, n_synth = 1, evidence = evi)[, ..k]
  })
  x_tilde <- do.call(cbind, x_tilde)
  
  # Calculate PFI values
  cli_progress_step("------- Calculating cPFI values")
  cpfi_values <- unlist(lapply(seq_len(ncol(x)), function(k) {
    x_copy <- x
    x_copy[[k]] <- x_tilde[[k]]
    pred_copy <- pred_fun(model, newdata = x_copy)
    logLoss(y, pred_copy) - full_loss
  }))
  
  res_cpfi <- data.frame(
    feature = colnames(x), 
    value = cpfi_values,
    method = "cPFI"
  )
  
  # ----------------------------------------------------------------------------
  
  # Summarize and return results
  cbind(res_cpfi,
        dataset_name = df$dataset_name[i],
        syn_name = df$syn_name[i],
        run_model = df$run_model[i],
        model_name = df$model_name[i])
})

# Combine results
res <- do.call(rbind, res)

# Save results
cli_progress_step("Saving results")
if (!dir.exists(paste0("./results/Q1"))) dir.create(paste0("./results/Q1"), recursive = TRUE)
saveRDS(res, "./results/Q1/cond_feature_importance.rds")

cli_progress_done("Done!")

################################################################################
#                           Plot results
################################################################################

ggplot(res, aes(y = feature, x = value, fill = method)) +
  geom_boxplot(position = "dodge") +
  facet_wrap(~dataset_name, scales = "free") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) 
