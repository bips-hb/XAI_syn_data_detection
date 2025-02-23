################################################################################
#                        Fit Detection Models
################################################################################
library(Metrics)
library(ranger)
library(cli)
library(cowplot)
library(ggplot2)
library(ParBayesianOptimization) # only for xgboost
library(doParallel)
library(parallelly)
library(xgboost)
library(parallel)
library(data.table)

cli_div(theme = list(span.emph = list(color = "#3c77b9")))

# Set seed
set.seed(2024)

################################################################################
#                         Global Settings
################################################################################

# Define the settings for fitting the detection models -------------------------
# Note: Can be very time consuming (especially for XGBoost)
filter_df <- data.table(expand.grid(
  dataset_name =  c(
    'adult_complete', 'car_evaluation', 'chess_king_rook_vs_king', 'connect_4', 
    'diabetes', 'diabetes_HI', 'diamonds', 'letter_recognition', 
    'magic_gamma_telescope', 'nursery', 'statlog_landsat_satellite'
  ), 
  model_name = "xgboost",#  c("ranger", "logReg", "xgboost"), 
  syn_name = c(
    "ARF", "CTAB-GAN+", "CTGAN", "synthpop", "TabSyn", "TVAE"
  )
))
max_runs <- 10


# The setting used in the main text of the manuscript
#filter_df <- data.table(
#  dataset_name = rev(c("adult_complete", "nursery")),
#  model_name = "xgboost", 
#  syn_name = rev(c("TabSyn", "CTGAN"))
#)

# Other global settings --------------------------------------------------------

# Threading/Number of CPUS
# Note: The total number of cores used will be 'mc.cores * n_threads'
n_threads <- 15 # number of threads for each mc parallel run
mc.cores <- 16L # number of cores for parallel processing

options(mc.cores = mc.cores)
options(ranger.num.threads = n_threads)
Sys.setenv("OMP_NUM_THREADS" = n_threads)

# XGBoost tuning parameters
n_parallel <- 15 #18
init_points <- n_parallel
time_limit <- 60 *30 # 45 minutes

# Load utility methods and create dirs -----------------------------------------

# Load global utility functions
source("utils.R")

# Create folder for tuning logs
if (!dir.exists("./tmp/tuning_logs")) dir.create("./tmp/tuning_logs", recursive = TRUE)

################################################################################
#                 Main script to fit detection models
################################################################################


# Create data.frame for all settings -------------------------------------------
cli_progress_step("Creating settings data.frame for fitting models")

# Find all available datasets
args <- lapply(list.files("./data"), function(dat_name) {
  expand.grid(
      dataset_name = dat_name, 
      model_name = c("ranger", "logReg", "xgboost"), 
      syn_name = list.files(paste0("./data/", dat_name, "/syn/")))
})
args <- do.call(rbind, args)
args <- data.table(args[order(args$dataset_name, args$model_name), ])

# Filter settings (as defined in the global settings)
args <- args[filter_df, on = c("dataset_name", "model_name", "syn_name")]


# Fitting detection models -----------------------------------------------------
cli_h1("Fitting detection models")

# Vector to store indices of rows with errors
error_idx <- c()

# Fit models
result <- mclapply(seq_len(nrow(args)), function(i) {
  # Set seed
  set.seed(42)
  
  cli_progress_step(paste0(
    "[{i}/{nrow(args)}] ",
    "Dataset: {.emph {args$dataset_name[i]}} --- ",
    "Model: {.emph {args$model_name[i]}} --- ",
    "Synthesizer: {.emph {args$syn_name[i]}}"))
  
  # Load data
  data <- load_data(args$dataset_name[i], args$syn_name[i], test_split = 0.3)
  
  # Fit model for the first 'max_runs' runs
  tryCatch({
    log <- fit_model(data, args$model_name[i], max_runs = max_runs, 
                     n_threads = n_threads, time_limit = time_limit, 
                     n_parallel = n_parallel, init_points = init_points)
    return(log)
  }, error = function(e) {
    error_idx <<- c(error_idx, i)
    cli_alert(col_red(paste0("[RUN {i}] Error in fitting model. Skipping.")))
    print(e)
    return(NULL)
  })
})
result <- data.table(do.call(rbind, result))

# Show errors if any
if (length(error_idx) > 0) {
  cli_alert(col_red(paste0("[ERROR SUMMARY] Error in fitting models for runs: ", 
                           paste(error_idx, collapse = ", "))))
}

# Update and save results ------------------------------------------------------
if (!dir.exists("./results/model_performance")) {
  dir.create("./results/model_performance", recursive = TRUE)
}

if (file.exists("./results/model_performance/model_performance.rds")) {
  result_old <- data.table(readRDS("./results/model_performance/model_performance.rds"))
  
  # Update result
  result_old <- rbind(
    result_old, 
    result[!result_old, on = c("dataset", "syn_name", "run", "model_name",
                               "train", "metric")])
  
  result_old[result, on = c("dataset", "syn_name", "run", "model_name",
                            "train", "metric"), value := i.value]
  result <- result_old
  
  cli_progress_step("Updating results")
}

cli_progress_step("Saving results")

saveRDS(result, file = "./results/model_performance/model_performance.rds")
cli_progress_done()


################################################################################
#                             Plot results
################################################################################
dat_levels <- unique(result$train)
cols <- RColorBrewer::brewer.pal(n = length(dat_levels), name = "Dark2")
plots <- lapply(unique(result$metric), function(met) {
  ggplot(result[result$metric == met, ], aes(x = syn_name, y = value, color = train)) +
    geom_boxplot() +
    facet_grid(rows = vars(model_name), cols = vars(dataset), scales = "fixed") +
    scale_color_manual(values = cols[which(dat_levels %in% unique(result[result$metric == met, ]$train))]) +
    geom_hline(yintercept = 0.5, color = "black", linetype = "dashed") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1),
          legend.position = "top") +
    labs(title = paste0("Detection model performance (", met, ")"), x = "Syntheziser",
         y = "Metric value", color = "")
  
  ggsave(paste0("./results/model_performance/plot_", met, ".pdf"), 
         plot = last_plot(), width = 16, height = 5)
})
