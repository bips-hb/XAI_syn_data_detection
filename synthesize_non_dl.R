suppressPackageStartupMessages({
  library(data.table)
  library(foreach)
  library(doParallel)
  library(arf)
  library(synthpop)
})

synthesize <- function(dataset, synthesizer, reps = 1) {
  set.seed(2024)
  data <- fread(paste0("data/", dataset, "/real/", dataset, ".csv"))

  # create synthesizer folder if it does not exist
  dir.create(paste0("data/", dataset, "/syn/", synthesizer), showWarnings = F)

  synthesizer_fn <- synthesizers[[synthesizer]]
  colnames_orig <- colnames(data)
  colnames(data) <- gsub("-", "_", colnames(data)) #rename colnames "-" to "_"
  
  syn_list <- foreach(rep = seq_len(reps)) %do% {
    
    print(paste0("Synthesizing dataset: ", dataset, " with synthesizer: ", synthesizer, " (rep: ", rep, "/", reps, ")"))
  
    syn_file <- paste0("data/", dataset, "/syn/", synthesizer, "/syn_", dataset, "_", synthesizer, "_", rep, ".csv")
    # if synthetic file does not exists yet, synthesize
    if (file.exists(syn_file)) {
      print(paste0("Synthetic data already exists at: ", syn_file))
    } else {
      syn <- synthesizer_fn(data)
      colnames(syn) <- colnames_orig #rename back
      fwrite(syn, file = paste0("data/", dataset, "/syn/", synthesizer, "/syn_", dataset, "_", synthesizer, "_", rep, ".csv"))
      print(paste0("Synthetic data saved at: ", syn_file))
    }
  }
  return(invisible(NULL))
}

# if called from command line
args_cmd = commandArgs(trailingOnly=TRUE)
args <- list(
  dataset = args_cmd[1],
  synthesizer = args_cmd[2],
  reps = as.integer(args_cmd[3]),
  n_cores = as.integer(args_cmd[4])
)


if (length(args_cmd) >= 2) {
  
  if(is.na(args$reps)) args$reps = 1
  if(is.na(args$n_cores)) args$n_cores = 1
  
  # register parallel backend if requested
  parallel = (args$n_cores>0)
  if (parallel) {
    registerDoParallel(args$n_cores)
  }
  
  # define synthesizers list
  synthesizers <- list(
    ARF  = \(data) rarf(data, finite_bounds = "local", parallel = parallel),
    synthpop = \(data) syn(data, print.flag = F)$syn
  )
  
  # synthesize
  synthesize(args$dataset, args$synthesizer, reps = args$reps)
}
