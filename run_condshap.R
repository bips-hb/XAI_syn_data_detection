################################################################################
#              Conditional SHAP (condSHAP)
#
#     This script computes Shapley values for local feature attributions with
#     properly estimated conditional distributions
#     given model, dataset, synthesizer and a set of real/synthetic observations,
#     it saves the results in the `results` folder.
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
library(shapr) # CRAN version OK
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

# Global arguments for the condSHAP method
SHAPR_PARALLEL <- TRUE # Run v(S) computations in parallel within shapr?
SHARP_NUM_WORKERS <- 10 # Number of workers to use for the parallelization in shapr
NUM_TRAIN <- 10^4 # Number of observations to use to train the feature distributions
NUM_COAL_SAMPLES <- 2000 # Maximum number of coalitions considered
NUM_MC_SAMPLES <- 5*10^2 # Number of Monte Carlo samples in the numerical integration for computing the v(S)
APPROACH <- "independence" # "ctree" or "independence"
PATH_relevant_test_obs <- "./prepare_local/relevant_test_obs.csv"

# Define global arguments
filter_df <- data.table(
  dataset_name = rev(c("adult_complete", "nursery")),
  model_name = c("xgboost"),
  syn_name = rev(c("TabSyn", "CTGAN")),
  run_model = rep(1:10,each=2)
)

# Load utility methods and create dirs -----------------------------------------

# Setting up parallellizaion for shapr::explain()
if(SHAPR_PARALLEL){
  future::plan("multisession", workers = SHARP_NUM_WORKERS)
}

# Load global utility functions
source("utils.R")

################################################################################
#                     Main script for running condSHAP
################################################################################

# Create data.frame for all settings -------------------------------------------
cli_progress_step("Creating settings data.frame for running condSHAP")

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

# Load relevant test observations

if(file.exists(PATH_relevant_test_obs)){
  dt_test_obs <- fread(PATH_relevant_test_obs)
} else {
  stop("The file with relevant test observations does not exist. Please run the prepare_local script first.")
}

dt_test_obs <- fread("./prepare_local/relevant_test_obs.csv")
dt_test_obs <- dt_test_obs[filter_df, on = c("dataset_name", "model_name", "syn_name", "run_model")]



# Running cPFI -----------------------------------------------------------------
cli_h1("Running condSHAP with shapr")

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
  data <- as.data.table(data[[df$file_name[i]]])

  # Get predict function
  pred_fun <- get_predict_fun_shapr(df$model_name[i])

  # Get feature columns
  feature_cols <- setdiff(names(data), c("real", "train", "rowid"))

  # Define training data (preliminary)
  data_train_full <- data[data$train==1]
  x_train <- rbind(
    data_train_full[real=="Real"][sample.int(.N,size=min(.N,round(NUM_TRAIN/2))),..feature_cols],
    data_train_full[real=="Synthetic"][sample.int(.N,size=min(.N,round(NUM_TRAIN/2))),..feature_cols]
  )

  # Get test data
  data_test <- data[data$train == 0]

  # Set row id
  data_test[,rowid := .I]

  # Get specific test observation to use
  dt_test_obs_i <- dt_test_obs[dataset_name == df$dataset_name[i] &
                                 model_name == df$model_name[i] &
                                 syn_name == df$syn_name[i] &
                                 run_model == df$run_model[i]]


  rowid0 <- dt_test_obs_i[, rowid]
  type0 <- dt_test_obs_i[,type]

  x_explain <- data_test[rowid %in% rowid0, ..feature_cols]

  class(model)="" # Required for workaround with pre-implemented model class in shapr.

  # Calculate conditional shaps
  expl <- shapr::explain(model,
                         x_train = x_train,
                         x_explain = x_explain,
                         phi0 = 0.5,
                         n_MC_samples = NUM_MC_SAMPLES,
                         approach = APPROACH,
                         predict_model = pred_fun,
                         verbose = c("basic","vS_details"),
                         iterative = FALSE,
                         seed = NULL,
                         max_n_coalitions = NUM_COAL_SAMPLES)

  timestamp <- format(Sys.time())

  res_condSHAP <- data.frame(
    feature = rep(feature_cols, each = length(rowid0)),
    rowid_test = rowid0,
    value = unlist(expl$shapley_values_est[,-c(1,2)]),
    method = "condSHAP",
    approach = APPROACH,
    type = rep(type0,each=length(feature_cols)),
    timestamp = timestamp
  )

  cli_progress_step("Saving results")
  if (!dir.exists(paste0("./results/Q3"))) dir.create(paste0("./results/Q3"), recursive = TRUE)
  fwrite(res_condSHAP, "./results/Q3/condshap.csv",append = TRUE)

  NULL # We store things below, so don't return anything
})


cli_progress_done("Done!")


