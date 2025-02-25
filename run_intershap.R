################################################################################
#              Computing Interaction Shapley values (interSHAP)
#
#     This script computes Shapley interaction values for local feature attributions
#     using path dependent treeSHAP given model, dataset, synthesizer,
#     and saves the results in the `results` folder.
################################################################################
library(ranger)
library(xgboost)
library(cli)
library(cowplot)
#library(Metrics)
library(data.table)
library(doParallel)
library(arf)
library(parallel)
library(ggplot2)
library(shapviz)
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

# Global arguments for the interSHAP method
train_or_test <- 0 # 0 for test, 1 for train data
NUM_SAMPLES <- Inf # Number of samples for the calculation

# Define global arguments
filter_df <- data.table(
  dataset_name = rev(c("adult_complete", "nursery")),
  model_name = c("xgboost"),
  syn_name = rev(c("TabSyn", "CTGAN")),
  run_model = rep(1:10,each=2)
)

### TEMP SETTINGS ONLY FOR TESTING ####

filter_df <- data.table(
  dataset_name = rev(c("adult_complete", "nursery")),
  model_name = c("xgboost"),
  syn_name = rev(c("TabSyn", "CTGAN")),
  run_model = c(8,2)
)

NUM_SAMPLES <- 20 # Number of samples for the calculation

### TEMP SETTINGS ONLY FOR TESTING  ENDS ####


# Load utility methods and create dirs -----------------------------------------

# Load global utility functions
source("utils.R")

################################################################################
#                     Main script for running cPFI
################################################################################

# Create data.frame for all settings -------------------------------------------
cli_progress_step("Creating settings data.frame for running CE")

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



# Running cPFI -----------------------------------------------------------------
cli_h1("Running interSHAP")

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
    cli_alert("Only xgboost models are currently supported")
    return(NULL)
  }
  data <- load_data(df$dataset_name[i], df$syn_name[i])

  # Select relevant dataset
  data <- as.data.table(data[[df$file_name[i]]])

  # Get feature columns
  feature_cols <- setdiff(names(data), c("real", "train", "rowid"))

  # Define training data (preliminary)
  data_train <- data[data$train==1]
  data_train[,rowid := .I]


  # Get test data
  data_test <- data[data$train == 0]
  data_test[,rowid := .I]

  if(train_or_test==0){
    x_explain0 <- data_test[sample.int(.N, min(NUM_SAMPLES, .N))]
  } else {
    x_explain0 <- data_train[sample.int(.N, min(NUM_SAMPLES, .N))]
  }
  x_explain <- x_explain0[,..feature_cols]

  x_explain_enc <- as.matrix(encode_cat_vars(x_explain))

  factor_cols <- names(which(sapply(x_explain, is.factor)))
  cols <- lapply(factor_cols, function(col) {
    grep(paste0(col, ": "), colnames(x_explain_enc), value = TRUE)
  })
  names(cols) <- factor_cols

  # Run shapviz with interactions
  res_intershap <- shapviz(model,
                           X_pred = x_explain_enc,
                           X = x_explain,
                           interactions = TRUE,
                           collapse = cols)

  timestamp <- format(Sys.time())


  out <- list(results = res_intershap,
              info=list(
                dataset = df$dataset[i],
                syn = df$syn[i],
                run_model = df$run_model[i],
                detect_model = df$model_name[i],
                rowid = x_explain0[,rowid],
                train_test = ifelse(train_or_test==0,"test","train"),
                timestamp = timestamp)
  )

  out
})

cli_progress_step("Saving results")
if (!dir.exists(paste0("./results/Q3"))) dir.create(paste0("./results/Q3"), recursive = TRUE)
saveRDS(res, "./results/Q3/intershap.rds")




cli_progress_done("Done!")
