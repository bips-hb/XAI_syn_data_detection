################################################################################
#              Counterfactual Explanations (CE)
#
#     This script runs Counterfactual Explanations (CE) with the MCCE method
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
library(mcceR) # Installed with remotes::install_github("NorskRegnesentral/mcceR")
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

# Global arguments for the CE method
NUM_TRAIN <- 10^4 # Number of samples for the calculation
GENERATE_K = 5*10^5#10^4 # TODO: Increse to at least 10^5
TO_EXPLAIN = "syn" # Which type of explanatins to explain (one or both of "real", "syn")
PATH_relevant_test_obs <- "./prepare_local/relevant_test_obs.csv"
specific_test_obs <- c(1353)
NO_CF <- 100

# Define global arguments
filter_df <- data.table(
  dataset_name = "adult_complete",
  model_name = c("xgboost"),
  syn_name = c("TabSyn"),
  run_model = c(2)
)



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

# Load relevant test observations

if(file.exists(PATH_relevant_test_obs)){
  dt_test_obs <- fread(PATH_relevant_test_obs)
} else {
  stop("The file with relevant test observations does not exist. Please run the prepare_local script first.")
}

dt_test_obs <- fread("./prepare_local/relevant_test_obs.csv")
dt_test_obs <- dt_test_obs[filter_df, on = c("dataset_name", "model_name", "syn_name", "run_model")]
if(!is.null(specific_test_obs)){
  dt_test_obs <- dt_test_obs[rowid %in% specific_test_obs]
}


# Running cPFI -----------------------------------------------------------------
cli_h1("Running Counterfactual Explanations (CE) with MCCE")

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
  pred_fun <- get_predict_fun(df$model_name[i])

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

  res_ce_values <- res_ce_measures <- NULL

  if("real" %in% TO_EXPLAIN){
    cli_progress_step("Computing counterfactuals for real observations")

    rowid_real <- dt_test_obs_i[type=="real", rowid]

    x_explain_real <- data_test[dt_test_obs_i[type=="real", .(rowid)], ..feature_cols, on = "rowid"]

    expl_real <- mcceR::explain_mcce(model = model,
                                     x_explain = x_explain_real,
                                     x_train = x_train,
                                     predict_model = pred_fun,
                                     c_int = c(0,0.5), # IMPORTANT
                                     fixed_features = NULL,
                                     process.measures = c("validation","L0","gower"),
                                     fit.seed = 123,
                                     fit.autoregressive_model = "rpart",
                                     generate.K = GENERATE_K,
                                     generate.seed = 123,
                                     process.return_best_k = NO_CF)

    melted_ce_values <- melt(data.table(rowid_test = rowid_syn, expl_syn$cf[,-c(1)]),
                             id.vars="rowid_test",variable.factor = FALSE,value.factor = FALSE)
    melted_org_values <- melt(data.table(rowid_test = rowid_syn, x_explain_syn),
                              id.vars="rowid_test",variable.factor = FALSE,value.factor = FALSE)

    res_ce_values <- rbind(res_ce_values,
                           data.table(melted_ce_values,row_type="cf",type="real"),
                           data.table(melted_org_values,row_type="org",type="real"),
                           fill=TRUE
    )

    res_ce_measures <- rbind(res_ce_measures,
                             data.table(rowid_test = rowid_real, expl_real$cf_measures[,-1],type="real")
    )

  }
  if("syn" %in% TO_EXPLAIN){
    cli_progress_step("Computing counterfactuals for synthetic observations")

    rowid_syn <- dt_test_obs_i[type=="syn", rowid]

    x_explain_syn <- data_test[dt_test_obs_i[type=="syn", .(rowid)], ..feature_cols, on = "rowid"]

    expl_syn <- mcceR::explain_mcce(model = model,
                                    x_explain = x_explain_syn,
                                    x_train = x_train,
                                    predict_model = pred_fun,
                                    c_int = c(0.5,1), # IMPORTANT
                                    fixed_features = NULL,
                                    process.measures = c("validation","L0","gower"),
                                    fit.seed = 123,
                                    fit.autoregressive_model = "rpart",
                                    generate.K = GENERATE_K,
                                    generate.seed = 123,
                                    process.return_best_k = NO_CF)

    melted_ce_values <- melt(data.table(rowid_test = rowid_syn, expl_syn$cf[,-c(1)]),
                             id.vars=c("rowid_test","counterfactual_rank"),variable.factor = FALSE, value.factor = FALSE)
    melted_org_values <- melt(data.table(rowid_test = rowid_syn, x_explain_syn),
                              id.vars="rowid_test",variable.factor = FALSE, value.factor = FALSE)

    res_ce_values <- rbind(res_ce_values,
                           data.table(melted_ce_values,row_type="cf",type="syn"),
                           data.table(melted_org_values,row_type="org",type="syn"),
                           fill=TRUE
    )

    res_ce_measures <- rbind(res_ce_measures,
                             data.table(rowid_test = rowid_syn, expl_syn$cf_measures[,-1],type="syn")
    )

  }

  timestamp <- format(Sys.time())

  # Summarize and return results
  out_ce_values <- cbind(res_ce_values,
                         dataset_name = df$dataset_name[i],
                         syn_name = df$syn_name[i],
                         run_model = df$run_model[i],
                         model_name = df$model_name[i],
                         timestamp = timestamp)

  out_ce_measures <- cbind(res_ce_measures,
                           dataset_name = df$dataset_name[i],
                           syn_name = df$syn_name[i],
                           run_model = df$run_model[i],
                           model_name = df$model_name[i],
                           timestamp = timestamp)

  cli_progress_step("Saving results")
  if (!dir.exists(paste0("./results/Q4"))) dir.create(paste0("./results/Q4"), recursive = TRUE)
  fwrite(out_ce_values, "./results/Q4/ce_values_extra2.csv",append = TRUE)
  fwrite(out_ce_measures, "./results/Q4/ce_measures_extra2.csv",append = TRUE)

  NULL # We store things below, so don't return anything
})


cli_progress_done("Done!")
