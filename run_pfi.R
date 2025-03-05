################################################################################
#                   Permutation Feature Importance (PFI)
#
#     This script runs Permutation Feature Importance (PFI) on the 
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
NUM_SAMPLES <- Inf # Number of samples for the calculation
train_or_test <- 0 # 0 for test, 1 for train data

# Define global arguments
filter_df <- data.table(
  dataset_name = c("adult_complete", "nursery"),
  model_name = c("xgboost"),
  syn_name = c("TabSyn", "CTGAN"),
  run_model = c(1:10, 1:10)
)

# Load utility methods and create dirs -----------------------------------------

# Load global utility functions
source("utils.R")

################################################################################
#                     Main script for running PFI
################################################################################

# Create data.frame for all settings -------------------------------------------
cli_progress_step("Creating settings data.frame for running PFI")

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
cli_h1("Running Permutation Feature Importance (PFI)")

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
  
  # Main part for PFI ----------------------------------------------------------
  
  # Get test data
  data_test <- data[data$train == train_or_test, -ncol(data)]
  data_test <- data_test[sample.int(nrow(data_test), min(NUM_SAMPLES, nrow(data_test))), ]
  x <- data_test[, -ncol(data_test)]
  y <- as.numeric(data_test$real) - 1
  
  # Calculate full loss
  full_loss <- logLoss(y, pred_fun(model, newdata = x))
  
  # Calculate PFI values
  pfi_values <- unlist(lapply(seq_len(ncol(x)), function(k) {
    x_copy <- x
    x_copy[[k]] <- sample(x[[k]])
    pred_copy <- pred_fun(model, newdata = x_copy)
    logLoss(y, pred_copy) - full_loss
  }))
  
  res_pfi <- data.frame(
    feature = colnames(x), 
    value = pfi_values,
    method = "PFI"
  )
  # ----------------------------------------------------------------------------
  
  # Calculate model-specific importance ----------------------------------------
  if (df$model_name[i] == "ranger") {
    model_imp <- data.frame(
      feature = names(model$variable.importance),
      value = as.numeric(model$variable.importance),
      method = "Model (impurity)"
    )
  } else if (df$model_name[i] == "logReg") {
    model_imp <- do.call("rbind", lapply(colnames(x), function(var) {
      idx <- which(startsWith(names(model$coefficients), var))
      data.frame(
        feature = var,
        value = abs(sum(as.numeric(model$coefficients[idx]))),
        method = "Model (|coef|)"
      )
    }))
  } else if (df$model_name[i] == "xgboost") {
    imp <- xgboost::xgb.importance(model = model)
    feat_idx <- as.numeric(gsub("f", "", imp$Feature))
    labels <- colnames(encode_cat_vars(x))
    imp$Feature <- factor(imp$Feature, 
                          levels = paste0("f", seq(min(feat_idx), max(feat_idx))),
                          labels = gsub(":.*", "", labels))
    model_imp <- data.table(imp)[,  .(value = sum(Gain)), by = "Feature"]
    model_imp <- model_imp[, .(feature = Feature, value = value, method = "Model (impurity)")]
  }
  # ----------------------------------------------------------------------------
  
  # Summarize and return results
  cbind(rbind(res_pfi, model_imp),
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
saveRDS(res, "./results/Q1/feature_importance.rds")

cli_progress_done("Done!")

################################################################################
#                           Plot results
################################################################################

ggplot(res, aes(y = feature, x = value, fill = method)) +
  geom_boxplot(position = "dodge") +
  facet_wrap(~dataset_name, scales = "free") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) 
