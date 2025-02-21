################################################################################
#                   Permutation Feature Importance (PFI)
#
#     This script runs Permutation Feature Importance (PFI) on the 
#     given model, dataset and synthesizer. It saves the results in the
#    `results` folder.
################################################################################

# Define global arguments
model_type <- "xgboost"
dataset_name <- "adult_complete"
synthesizer_name <- "TabSyn"
runs <- "all"


# TODO