################################################################################
#
#                  Create all Figures for the manuscript
#
#              "What’s Wrong with Your Synthetic Tabular Data?
#             Using Explainable AI to Evaluate Generative Models"
#
# ------------------------------------------------------------------------------
#
# This file is structured as follows:
#    - Header: Load libraries, utility functions and set theme
#    - Section 5.1: Model Performance
#    - Research Question 1 (Q1): Feature Importance
#    - Research Question 2 (Q2): Feature Effects
#    - Research Question 3 (Q3): Feature Interactions
#    - Research Question 4 (Q4): Counterfactuals
#
################################################################################
library(ggplot2)
library(cowplot)
library(patchwork)
library(rlang)
library(data.table)
library(xtable)
library(shapviz)
library(flextable)
library(rsvg)
library(svglite)

# Set theme
theme_set(theme_minimal(base_size = 15))

# Create folder for plots
if (!dir.exists("figures")) dir.create("figures")
if (!dir.exists("tables")) dir.create("tables")

# Load utility functions
source("utils.R")

################################################################################
#                      SECTION 5.1: MODEL PERFORMANCE
#
#       Show the performance of the models on the different datasets,
#       synthesizers, model types and runs.
################################################################################
res_perf <- data.table(readRDS("./results/model_performance/model_performance.rds"))

# Model Performance Nursery and Adult ------------------------------------------
df <- res_perf[
  ((dataset == "adult_complete" & syn_name == "TabSyn") |
     (dataset == "nursery" & syn_name == "CTGAN")) &
    model_name == "xgboost" & metric == "Accuracy"]
df$dataset <- factor(df$dataset, levels = c("adult_complete", "nursery"),
                     labels = c("Adult", "Nursery"))
df$train <- factor(df$train, levels = c("train data", "test data"),
                   labels = c("Train", "Test"))
p1 <- ggplot(df, aes(x = train, y = value)) +
  geom_boxplot(fill = "darkgray") +
  facet_grid(cols = vars(dataset), scales = "free") +
  labs(x = NULL, fill = "Metric", y = "Accuracy (XGBoost)") +
  theme(legend.position = "top") +
  scale_y_continuous(limits = c(0.5, 1), labels = scales::percent) +
  geom_hline(yintercept = 0.5, linetype = "dashed")

# Performance on all datasets/models -------------------------------------------
thres1 <- 0.6
thres2 <- 0.8

# Use only up to 5 runs
df <- res_perf[metric == "Accuracy" & run <= 5,
               .(correct = sum(value > thres2),
                 middle = sum(value > thres1 & value <= thres2),
                 incorrect = sum(value <= thres1)),
               by = c("syn_name", "model_name", "train")]
df <- melt(df, id.vars = c("syn_name", "model_name", "train"))

res_fill <- df[, .(value = sum(value)), by = c("syn_name", "train", "model_name")]
max_sum <- res_fill[, max(value)]
res_fill <- res_fill[value != max_sum, ]
res_fill$variable <- "NA"
res_fill$value <- max_sum - res_fill$value
df <- rbind(df, res_fill)
df$syn_name <- factor(df$syn_name,
                      levels = rev(c("TabSyn", "CTGAN", "TVAE", "CTAB-GAN+", "ARF", "synthpop")))
df$model_name <- factor(df$model_name, levels = c("logReg", "ranger", "xgboost"),
                        labels = c("Logistic Regression", "Random Forest", "XGBoost"))

df$variable <- factor(df$variable, levels = c("correct", "middle", "incorrect", "NA"),
                      labels = c(paste0("Above ", thres2 * 100, "%"),
                                 paste0("Between ", thres1 * 100, "% and ", thres2 * 100, "%"),
                                 paste0("Below ", thres1 * 100, "%"), "NA"))

# Create plot
p2 <- ggplot(df[train == "test data"],
             aes(x = value, y = syn_name, fill = variable)) +
  geom_bar(stat = "identity", position = "stack", width = 0.9) +
  geom_text(aes(label = ifelse(value > 10, value, "")), position = position_stack(vjust = 0.5)) +
  scale_fill_manual(values = c("darkgreen", "darkorange", "darkred", "gray")) +
  scale_x_continuous(expand = c(0,0)) +
  facet_grid(cols = vars(model_name), scales = "free") +
  labs(x = "Frequency", y = "Synthesizer", fill = "Test Accuracy") +
  theme(legend.position = "top",
        plot.margin = margin(0,0,0,0))

# Combine plots
p <- plot_grid (p2, p1, nrow = 1, rel_widths = c(8, 4), labels = c("(a)", "(b)"))

ggsave("figures/FIG_1_model_performance.pdf", width = 12, height = 4)

################################################################################
#                           RESEARCH QUESTION 1 (Q1)
#
#       Which features and feature interactions were most challenging
#                       for the generative model?
################################################################################
res_pfi <- data.table(readRDS("./results/Q1/feature_importance.rds"))
res_treeshap <- readRDS("./results/Q3/intershap.rds")

# Plot for adult_complete ------------------------------------------------------

# PFI values
df_pfi <- res_pfi[dataset_name == "adult_complete" & method == "PFI",
                  c("feature", "value", "method")]

# TreeSHAP values (marginal)
res_adult <- lapply(res_treeshap, function(a) {
  if (a$info$dataset == "adult_complete") {
    return(a$results)
  }
})
res_adult <- res_adult[!sapply(res_adult, is.null)]
df_marginal_shap <- rbindlist(lapply(res_adult, function(a) {
  values <- colMeans(abs(a$S))
  data.frame(value = as.numeric(values),
             feature = names(values),
             method = "global TreeSHAP")
}))

# Create plot
df <- rbind(df_pfi, df_marginal_shap)
df$type <- ifelse(df$method == "global TreeSHAP", "Prediction-based", "Loss-based")
df$method <- factor(df$method, levels = c("PFI", "global TreeSHAP"))

# Create plot for loss-based and prediction-based feature importance
p1 <- ggplot(df, aes(y = feature, x = value, fill = method)) +
  geom_boxplot(position = "dodge") +
  geom_vline(xintercept = 0, linetype = "dashed") +
  facet_grid(cols = vars(type), scales = "free_x") +
  scale_fill_brewer(palette = "Dark2") +
  theme(legend.position = "top") +
  labs(x = "Importance", y = "Feature", fill = NULL)


# Interactions using TreeSHAP
intshap_df <- get_top_df(res_adult, top = 20)
p2 <- ggplot(intshap_df, aes(x = var, y = value, fill = degree)) +
  geom_boxplot() +
  coord_flip() +
  scale_fill_viridis_d(direction = -1) +
  theme(legend.position = "top") +
  facet_grid(cols = vars(type), scales = "free_x") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(y = "Importance", x = NULL, fill = "degree")


p <- plot_grid(p1, p2, nrow = 1, rel_widths = c(0.6, 0.4), labels = c("(a)", "(b)"))
ggsave("figures/FIG_2_Q1_adult_complete.pdf", plot = p, width = 12, height = 5)

# Plot for nursery (appendix) --------------------------------------------------

# PFI values
df_pfi <- res_pfi[dataset_name == "nursery" & method == "PFI", c("feature", "value", "method")]

# TreeSHAP values (marginal)
res_nursery <- lapply(res_treeshap, function(a) {
  if (a$info$dataset == "nursery") {
    return(a$results)
  }
})
res_nursery <- res_nursery[!sapply(res_nursery, is.null)]
df_marginal_shap <- rbindlist(lapply(res_nursery, function(a) {
  values <- colMeans(abs(a$S))
  data.frame(value = as.numeric(values),
             feature = names(values),
             method = "global TreeSHAP")
}))

# Create plot
df <- rbind(df_pfi, df_marginal_shap)
df$type <- ifelse(df$method == "global TreeSHAP", "Prediction-based", "Loss-based")
df$method <- factor(df$method, levels = c("PFI", "global TreeSHAP"))

p1 <- ggplot(df, aes(y = feature, x = value, fill = method)) +
  geom_boxplot(position = "dodge") +
  geom_vline(xintercept = 0, linetype = "dashed") +
  facet_grid(cols = vars(type), scales = "free_x") +
  scale_fill_brewer(palette = "Dark2") +
  theme(legend.position = "top") +
  labs(x = "Importance", y = "Feature", fill = NULL)


# Interactions using TreeSHAP
intshap_df <- get_top_df(res_nursery, top = 20)
p2 <- ggplot(intshap_df, aes(x = var, y = value, fill = degree)) +
  geom_boxplot() +
  coord_flip() +
  scale_fill_viridis_d(direction = -1) +
  theme(legend.position = "top") +
  facet_grid(cols = vars(type), scales = "free_x") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(y = "Importance", x = NULL, fill = "degree")

p <- plot_grid(p1, p2, nrow = 1, rel_widths = c(0.6, 0.4), labels = c("(a)", "(b)"))

ggsave("figures/FIG_8_APP_Q1_nursery.pdf", plot = p, width = 12, height = 5)

################################################################################
#                         RESEARCH QUESTION 2 (Q2)
#
#     How do the generative models behave in low and high density areas
#     of feature distributions? Which areas are under- or overrepresented
#     in the synthetic data?
################################################################################
res_q2 <- data.table(readRDS("./results/Q2/feat_effects.rds"))
res_q2_rugs <- data.table(readRDS("./results/Q2/feat_effects_rugs.rds"))

# Set number of rugs and ICE plots
num_rugs <- 1000
num_ice <- 50

# Set seed
set.seed(42)

# Plot for adult_complete ------------------------------------------------------
df <- res_q2[dataset_name == "adult_complete", ]
df_rug <- res_q2_rugs[dataset_name == "adult_complete", ]

# Plot for feature 'education_num'
df_ice <- df[feature == "education_num" & method == "ice", ]
df_pdp <- df[feature == "education_num" & method == "pdp" & real == "both", ]
df_rug_feat <- df_rug[variable == "education_num", ]
df_rug_feat$gridpoint <- df_rug_feat$value
ids <- sample(unique(df_ice$id), num_ice)
df_ice <- df_ice[id %in% ids]
p1 <- ggplot(mapping = aes(x = as.numeric(as.character(gridpoint)))) +
  geom_line(data = df_ice, aes(group = id, color = real, y = value), alpha = 0.5, linewidth = 0.3) +
  geom_line(aes(group = real, y = value), data = df_pdp, linewidth = 1.5, color = "black") +
  geom_rug(data = df_rug_feat[sample(nrow(df_rug_feat), min(num_rugs, nrow(df_rug_feat)))],
           aes(color = real), sides = "b", alpha = 0.5) +
  facet_grid(cols = vars(feature), labeller = function(s) paste0("Feature: ", s)) +
  labs(x = "Feature value", y = "Prediction", color = NULL) +
  theme(legend.position = "top")

# Plot for feature 'occupation'
df_ice <- df[feature == "occupation" & method == "ice", ]
df_pdp <- df[feature == "occupation" & method == "pdp" & real == "both", ]
df_rug_feat <- df_rug[variable == "occupation", ]
df_rug_feat <- df_rug_feat[, .(count = .N / nrow(df_rug_feat)), by = c("real", "value")]
p2 <- ggplot(df_ice, aes(y = gridpoint, x = value)) +
  geom_bar(stat = "identity", data = df_rug_feat, aes(y = value, x = count, fill = real), alpha = 0.5, inherit.aes = FALSE) +
  labs(x = "Prediction", y = "Feature value", fill = NULL) +
  facet_grid(cols = vars(feature), labeller = function(s) paste0("Feature: ", s)) +
  geom_boxplot(fill = "darkgray") +
  stat_summary(geom = "crossbar", fun = "mean", color = "darkred", width = 0.75) +
  geom_vline(xintercept = 0.5, linetype = "dashed") +
  theme(legend.position = "top")

# Combine plots
p <- plot_grid(p1, p2, nrow = 1, labels = c("(a)", "(b)"))
ggsave("figures/FIG_3_Q2_adult_complete.pdf", plot = p, width = 12, height = 5)


# Plot for nursery -------------------------------------------------------------
df <- res_q2[dataset_name == "nursery", ]
df_rug <- res_q2_rugs[dataset_name == "nursery", ]

# Plot for feature 'form'
df_ice <- df[feature == "form" & method == "ice", ]
df_rug_feat <- df_rug[variable == "form", ]
df_rug_feat <- df_rug_feat[, .(count = .N / nrow(df_rug_feat)), by = c("real", "value")]
p1 <- ggplot(df_ice, aes(y = gridpoint, x = value)) +
  geom_bar(stat = "identity", data = df_rug_feat, aes(y = value, x = count, fill = real), alpha = 0.5, inherit.aes = FALSE) +
  facet_grid(cols = vars(feature), labeller = function(s) paste0("Feature: ", s)) +
  geom_boxplot(fill = "darkgray") +
  stat_summary(geom = "crossbar", fun = "mean", color = "darkred", width = 0.75) +
  geom_vline(xintercept = 0.5, linetype = "dashed") +
  labs(x = "Prediction", y = "Feature value", fill = NULL) +
  theme(legend.position = "top")

# Plot for feature 'class'
df_ice <- df[feature == "class" & method == "ice", ]
df_rug_feat <- df_rug[variable == "class", ]
df_rug_feat <- df_rug_feat[, .(count = .N / nrow(df_rug_feat)), by = c("real", "value")]
p2 <- ggplot(df_ice, aes(y = gridpoint, x = value)) +
  geom_bar(stat = "identity", data = df_rug_feat, aes(y = value, x = count, fill = real), alpha = 0.5, inherit.aes = FALSE) +
  facet_grid(cols = vars(feature), labeller = function(s) paste0("Feature: ", s)) +
  geom_boxplot(fill = "darkgray") +
  stat_summary(geom = "crossbar", fun = "mean", color = "darkred", width = 0.75) +
  geom_vline(xintercept = 0.5, linetype = "dashed") +
  labs(x = "Prediction", y = "Feature value", fill = NULL) +
  theme(legend.position = "top")


# Combine plots
p <- plot_grid(p1, p2, nrow = 1, align = "h", labels = c("(a)", "(b)"))
ggsave("figures/FIG_9_APP_Q2_nursery.pdf", plot = p, width = 12, height = 4)


################################################################################
#                         RESEARCH QUESTION 3 (Q3)
#
#     Which features and feature dependencies/interactions contributed most
#     to the detection of an individual real or synthetic observation?
################################################################################
res_condshap <- fread("./results/Q3/condshap.csv")
res_intershap <- readRDS("./results/Q3/intershap.rds")

#-------------------------------------------------------------------------------
# TreeSHAP (marginal vs. conditional) for adult_complete (FIG. 4)
#-------------------------------------------------------------------------------

# Adult data specification
dataset_name0 = "adult_complete"
model_name0 = "xgboost"
syn_name0 = "TabSyn"
run_model0 = 2

# Preprocessing ----------------------------------------------------------------

# First considering synthetic observations
this_res_condshap <- res_condshap[dataset_name==dataset_name0 &
                                    syn_name == syn_name0 &
                                    run_model==run_model0 &
                                    model_name == model_name0 &
                                    type == "syn"]


features_cols <- this_res_condshap[,unique(feature)]

# Get original data values
data <- load_data(dataset_name0, syn_name)

data <- as.data.table(data[[paste0(dataset_name0,"--",syn_name0,"--",run_model0)]])

data_test <- data[data$train == 0]
data_test[,rowid_test := .I]

features_dt <- data_test[rowid_test %in% this_res_condshap[,unique(rowid_test)],]

features_dt[,(features_cols) := lapply(.SD, function(x) as.character(x)), .SDcols = features_cols]

####

# Flattening ctree and indep shapley values
ctree_dt <- dcast(this_res_condshap[approach=="ctree",.(rowid_test,feature,value)],
                  formula = rowid_test~feature)
indep_dt <- dcast(this_res_condshap[approach=="independence",.(rowid_test,feature,value)],
                  formula = rowid_test~feature)


# Plot with ctree and indep shapley values -------------------------------------
this_rowid_test= 1353

# Conditional (ctree)
p1 <- sv_force_shapviz_mod3(ctree_dt[rowid_test==this_rowid_test,..features_cols],
                          b=0.5,
                          features_dt[rowid_test==this_rowid_test,..features_cols],
                          row_id = 1,
                          max_display=5,
                          fill_colors = c("darkgreen","darkred"),
                          bar_label_size = 4,
                          annotation_size = 4,show_annotation = FALSE)+
  labs(y="Conditional") +
  theme(plot.title = element_text(size = 16,face = "bold",hjust=0.5),
        axis.text = element_text(size = 14),
        axis.title = element_text(size = 14),
        axis.title.y = element_text(size=16, face = "bold"),
        axis.line.x = element_blank(),
        axis.title.x = element_blank(),
        axis.text.x = element_blank())

# Marginal (indep)
p2 <- sv_force_shapviz_mod3(indep_dt[rowid_test==this_rowid_test,..features_cols],
                          b=0.5,
                          features_dt[rowid_test==this_rowid_test,..features_cols],
                          row_id = 1,
                          max_display=5,
                          fill_colors = c("darkgreen","darkred"),
                          bar_label_size  = 4,
                          annotation_size = 4)+
  labs(y="Marginal") +
  theme(plot.title = element_text(size = 16,face = "bold",hjust=0.5),
        axis.text = element_text(size = 14),
        axis.title = element_text(size = 14),
        axis.title.y = element_text(size=16, face = "bold"))


# Combine plots
pl_cond_vs_marg <- (p1/p2)
ggsave("figures/FIG_4_Q3_adult_complete.pdf", plot = pl_cond_vs_marg, width = 10, height = 3.2)


#-------------------------------------------------------------------------------
# TreeSHAP (interaction) for adult (FIG. 5)
#-------------------------------------------------------------------------------

# Preprocessing ----------------------------------------------------------------
info_dt_intershap <- as.data.table(t(sapply(res_intershap, function(x) unlist(x$info[c("dataset","syn","run_model","detect_model")]))))
info_dt_intershap[,row_id:=.I]

# First considering synthetic observations
this_intershap_index <- info_dt_intershap[dataset==dataset_name0 &
                                            syn==syn_name0 &
                                            run_model==run_model0 &
                                            detect_model==model_name0,row_id]
this_res_intershap <- res_intershap[[this_intershap_index]]$results
this_info_intershap <- res_intershap[[this_intershap_index]]$info

# Getting observations to plot
relevant_test_obs <- fread("./results/prepare_local/relevant_test_obs.csv")
this_relevant_test_obs <- relevant_test_obs[dataset_name==dataset_name0 &
                                              syn_name == syn_name0 &
                                              run_model==run_model0 &
                                              model_name == model_name0 &
                                              type == "syn"]
shapviz_row_mapper <- data.table(rowid_testobs = this_relevant_test_obs[,rowid])
shapviz_row_mapper[,shapviz_rowno := match(rowid_testobs,this_info_intershap$rowid)]


# TreeSHAP (interactions) Waterfall plot (FIG. 5) ------------------------------
this_rowid_test= 1353
this_pred_logodds <- sum(this_res_intershap$S[shapviz_row_mapper[rowid_testobs==this_rowid_test,shapviz_rowno],])+this_res_intershap$baseline
this_pred <- 1/(1+exp(-this_pred_logodds))

pl_inter <- plot_waterfall(this_res_intershap,
                           row_id = shapviz_row_mapper[rowid_testobs==this_rowid_test,shapviz_rowno],
                           fill_colors = c("darkgreen","darkred"),
                           marg_int_colors = c("orange","purple4"),annotation_size = 5) +
  theme(plot.title = element_text(size = 16,face = "bold",hjust=0.5),
        axis.text = element_text(size = 14),
        axis.title = element_text(size = 14),
        axis.title.y = element_text(size=16, face = "bold"))

ggsave("figures/FIG_5_Q3_adult_complete.pdf", plot = pl_inter, width = 10, height = 3.6)

#-------------------------------------------------------------------------------
# TreeSHAP (conditional) for adult (FIG. 6a)
#-------------------------------------------------------------------------------

# Preprocessing ----------------------------------------------------------------

this_res_condshap <- res_condshap[dataset_name==dataset_name0 &
                                    syn_name == syn_name0 &
                                    run_model==run_model0 &
                                    model_name == model_name0 &
                                    type == "real"]

# Get original data values
data <- load_data(dataset_name0, syn_name0)

data <- as.data.table(data[[paste0(dataset_name0,"--",syn_name0,"--",run_model0)]])

data_test <- data[data$train == 0]
data_test[,rowid_test := .I]

features_dt <- data_test[rowid_test %in% this_res_condshap[,unique(rowid_test)],]

features_dt[,(features_cols) := lapply(.SD, function(x) as.character(x)), .SDcols = features_cols]

####

# Flattening ctree and indep shapley values
ctree_dt <- dcast(this_res_condshap[approach=="ctree",.(rowid_test,feature,value)],
                  formula = rowid_test~feature)
indep_dt <- dcast(this_res_condshap[approach=="independence",.(rowid_test,feature,value)],
                  formula = rowid_test~feature)


# Plot with ctree shapley values -----------------------------------------------
this_rowid_test= 16025
p1 <- sv_force_shapviz_mod3(ctree_dt[rowid_test==this_rowid_test,..features_cols],
                          b=0.5,
                          features_dt[rowid_test==this_rowid_test,..features_cols],
                          row_id = 1,
                          max_display=5,
                          fill_colors = c("darkgreen","darkred"),
                          bar_label_size = 5,
                          annotation_size = 5)+
  labs(y="Conditional") +
  theme(plot.title = element_text(size = 16,face = "bold",hjust=0.5),
        axis.text = element_text(size = 14),
        axis.title = element_text(size = 14),
        axis.title.y = element_text(size=16, face = "bold"))

#-------------------------------------------------------------------------------
# TreeSHAP (interaction) for adult (FIG. 6b)
#-------------------------------------------------------------------------------

# Preprocessing ----------------------------------------------------------------
this_relevant_test_obs <- relevant_test_obs[dataset_name==dataset_name0 &
                                              syn_name == syn_name0 &
                                              run_model==run_model0 &
                                              model_name == model_name0 &
                                              type == "real"]

shapviz_row_mapper <- data.table(rowid_testobs = this_relevant_test_obs[,rowid])
shapviz_row_mapper[,shapviz_rowno := match(rowid_testobs,this_info_intershap$rowid)]


# TreeSHAP (interactions) Waterfall plot ---------------------------------------
this_rowid_test= 16025
this_pred_logodds <- sum(this_res_intershap$S[shapviz_row_mapper[rowid_testobs==this_rowid_test,shapviz_rowno],])+this_res_intershap$baseline
this_pred <- 1/(1+exp(-this_pred_logodds))


p2 <- plot_waterfall(this_res_intershap,
                           row_id = shapviz_row_mapper[rowid_testobs==this_rowid_test,shapviz_rowno],
                           fill_colors = c("darkgreen","darkred"),
                           marg_int_colors = c("orange","purple4"),annotation_size = 4)+
  theme(plot.title = element_text(size = 16,face = "bold",hjust=0.5),
        axis.text = element_text(size = 14),
        axis.title = element_text(size = 14),
        axis.title.y = element_text(size=16, face = "bold"))

# Combine plots
p <- plot_grid(p1, p2, nrow = 2, labels = c("(a)", "(b)"), rel_heights = c(1, 2))
ggsave("figures/FIG_6_Q3_adult_complete.pdf", plot = p, width = 14, height = 8)



#-------------------------------------------------------------------------------
# TreeSHAP (interactions) for nursery (FIG. 10 in appendix)
#-------------------------------------------------------------------------------

# nursery data specification
dataset_name = "nursery"
model_name = "xgboost"
syn_name = "CTGAN"
run_model = 2


# Preprocessing ----------------------------------------------------------------

# First considering synthetic observations
this_intershap_index <- info_dt_intershap[dataset==dataset_name0 &
                                            syn == syn_name0 &
                                            run_model==run_model0 &
                                            detect_model == model_name0, row_id]
this_res_intershap <- res_intershap[[this_intershap_index]]$results
this_info_intershap <- res_intershap[[this_intershap_index]]$info

# Getting observations to plot
relevant_test_obs <- fread("./results/prepare_local/relevant_test_obs.csv")

this_relevant_test_obs <- relevant_test_obs[dataset_name==dataset_name0 &
                                              syn_name == syn_name0 &
                                              run_model==run_model0 &
                                              model_name == model_name0 &
                                              type == "syn"]

shapviz_row_mapper <- data.table(rowid_testobs = this_relevant_test_obs[,rowid])
shapviz_row_mapper[,shapviz_rowno := match(rowid_testobs,this_info_intershap$rowid)]


# Plot TreeSHAP (interactions) with Waterfallplot ------------------------------
this_rowid_test= 1342

this_pred_logodds <- sum(this_res_intershap$S[shapviz_row_mapper[rowid_testobs==this_rowid_test,shapviz_rowno],])+this_res_intershap$baseline
this_pred <- 1/(1+exp(-this_pred_logodds))


pl_inter <- plot_waterfall(this_res_intershap,
                           row_id = shapviz_row_mapper[rowid_testobs==this_rowid_test,shapviz_rowno],
                           fill_colors = c("darkgreen","darkred"),
                           marg_int_colors = c("orange","purple4"),annotation_size = 5)+
  theme(plot.title = element_text(size = 16,face = "bold",hjust=0.5),
        axis.text = element_text(size = 14),
        axis.title = element_text(size = 14),
        axis.title.y = element_text(size=16, face = "bold"))

ggsave("figures/FIG_10_APP_Q3_nursery.pdf", plot = pl_inter, width = 10, height = 3.6)


################################################################################
#                         RESEARCH QUESTION 4 (Q4)
#
#     Which minimal changes to a correctly classified synthetic observation
#     could be performed to make it look realistic?
################################################################################

res_ce_values <- fread("./results/Q4/ce_values.csv")
res_ce_measures <- fread("./results/Q4/ce_measures.csv")


# nursery data specification
dataset_name0 = "adult_complete"
model_name0 = "xgboost"
syn_name0 = "TabSyn"
run_model0 = 2

this_res_ce_measures <- res_ce_measures[dataset_name==dataset_name0 &
                                          syn_name == syn_name0 &
                                          run_model==run_model0 &
                                          model_name == model_name0 &
                                          type == "syn"]

this_res_ce_values <- res_ce_values[dataset_name==dataset_name0 &
                                      syn_name == syn_name0 &
                                      run_model==run_model0 &
                                      model_name == model_name0 &
                                      type == "syn"]

# Reduce to those with the largest L0 measure
features_cols <- unique(this_res_ce_values$variable)

this_rowid <- 1353
these_cf_ranks <- c(3,17,9,16) # 13

# Create table
tab_list <- list()
tab_final <- NULL
for(i in seq_along(these_cf_ranks)){
  this_cf_rank <- these_cf_ranks[i]
  tmp <- this_res_ce_values[rowid_test==this_rowid & counterfactual_rank%in%c(this_cf_rank,NA),.(variable,value,row_type)]
  tab <- data.table(org=tmp[row_type=="org"][,value], cf=tmp[row_type=="cf"][,value])

  if(i==1){
    tab_final <- cbind(tab_final,tab)
  } else {
    tab_final <- cbind(tab_final,tab[,-1])
  }
}

# Set names
colnames(tab_final)[1] <- "Original"
colnames(tab_final)[-1] <- paste0("CF",seq_along(these_cf_ranks))

# Create and save table
cf_table <- make_cf_table(tab_final, "adult_complete")
save_as_image(x = cf_table, path = "figures/FIG_7_Q4_adult_complete.svg")
rsvg_pdf("figures/FIG_7_Q4_adult_complete.svg", file = "figures/FIG_7_Q4_adult_complete.pdf")


# Nursery dataset --------------------------------------------------------------

# nursery data specification
dataset_name0 = "nursery"
model_name0 = "xgboost"
syn_name0 = "CTGAN"
run_model0 = 2

this_res_ce_measures <- res_ce_measures[dataset_name==dataset_name0 &
                                          syn_name == syn_name0 &
                                          run_model==run_mode0l &
                                          model_name == model_name0 &
                                          type == "syn"]

this_res_ce_values <- res_ce_values[dataset_name==dataset_name0 &
                                      syn_name == syn_name0 &
                                      run_model==run_model0 &
                                      model_name == model_name0 &
                                      type == "syn"]

# Reduce to those with the largest L0 measure
features_cols <- unique(this_res_ce_values$variable)

this_rowid <- 1342
these_cf_ranks <- this_res_ce_measures[,which(!duplicated(pred))]
these_cf_ranks <- this_res_ce_measures[,which(!duplicated(pred))][c(1,2,3,5)]

tab_list <- list()
tab_final <- NULL
for(i in seq_along(these_cf_ranks)){
  this_cf_rank <- these_cf_ranks[i]
  tmp <- this_res_ce_values[rowid_test==this_rowid & counterfactual_rank%in%c(this_cf_rank,NA),.(variable,value,row_type)]
  tab <- data.table(org=tmp[row_type=="org"][,value], cf=tmp[row_type=="cf"][,value])
  if(i==1){
    tab_final <- cbind(tab_final,tab)
  } else {
    tab_final <- cbind(tab_final,tab[,-1])
  }
}

# Set names
colnames(tab_final)[1] <- "Original"
colnames(tab_final)[-1] <- paste0("CF",seq_along(these_cf_ranks))
cols <- colnames(tab_final)

# Create and save table
cf_table <- make_cf_table(tab_final, "nursery")
save_as_image(x = cf_table, path = "figures/FIG_11_APP_nursery.svg")
rsvg_pdf("figures/FIG_11_APP_nursery.svg", file = "figures/FIG_11_APP_nursery.pdf")
