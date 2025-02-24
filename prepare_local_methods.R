################################################################################
#              Prepare local explanations
#
#     This script prepares the local explanation runs by extracing
#     a given number of the largest real predicitons and
#     smallest synthetic predicitons in the test data, and
#     stores them in the `prepare_local` folder.
################################################################################
library(cli)
library(data.table)
cli_div(theme = list(span.emph = list(color = "#3c77b9")))

# Load global utility functions
source("utils.R")


# Set seed for reproducibility
set.seed(42)

# Manage number of cores and RAM
# Note: The total number of cores used will be 'mc.cores * n_threads'
n_threads <- 100L
# mc.cores <-10L
#
# options(future.globals.maxSize = 25000 * 1024^2)
# Sys.setenv(R_RANGER_NUM_THREADS = n_threads)
# Sys.setenv(OMP_THREAD_LIMIT = n_threads)
# options(mc.cores = mc.cores)

# Global arguments for the cPFI method
top_k_real = 100 # The number of largest real predictions to extract
top_k_syn = 100 # The number of smallest synthetic predictions

# Define global arguments
filter_df <- data.table(
  dataset_name = rev(c("adult_complete", "nursery")),
  model_name = c("xgboost"),
  syn_name = rev(c("TabSyn", "CTGAN")),
  run_model = rep(1:10,each=2)
)



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

# Running prepare local -----------------------------------------------------------------
cli_h1("Running computation smallest and largest predicitons")

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

  # Get all data
  full_data <- data[, -ncol(data)]
  full_x <- data.table(full_data[, -ncol(full_data)])

  # Get test data
  data_test <- data.table(data[data$train == 0, -ncol(data)])

  # Set row id
  data_test[,rowid := .I]

  # Get feature columns
  feature_cols <- setdiff(names(data_test), c("real", "train", "rowid"))

  # Predict
  data_test[,pred:=pred_fun(model,.SD), .SDcols=feature_cols]


  # Split real and synthetic data
  data_test_real <- data_test[real=="Real"]
  data_test_syn <- data_test[real=="Synthetic"]

  # Rank predictions (real by largest, synthetic by smallest)
  data_test_real[,predrank:=frank(-pred,ties.method = "first")]
  data_test_syn[,predrank:=frank(pred,ties.method = "first")]

  # Extract the relevant rowids
  rowid_real <- data_test_real[predrank<=top_k_real,.(rowid,predrank,pred)]
  rowid_syn <- data_test_syn[predrank<=top_k_syn,.(rowid,predrank,pred)]

  res0 <- rbind(cbind(type="real",rowid_real),
                cbind(type="syn",rowid_syn))

  # ----------------------------------------------------------------------------

  cbind(res0,
        dataset_name = df$dataset_name[i],
        syn_name = df$syn_name[i],
        run_model = df$run_model[i],
        model_name = df$model_name[i])


})


if (!dir.exists(paste0("./prepare_local"))) dir.create(paste0("./prepare_local"), recursive = TRUE)
fwrite(rbindlist(res), paste0("./prepare_local/relevant_test_obs.csv"))
