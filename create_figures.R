################################################################################
#                 Create all Figures for the manuscript
################################################################################
library(ggplot2)
library(patchwork)
library(geomtextpath)
library(rlang)
library(data.table)
library(xtable)

# Set theme
theme_set(theme_minimal(base_size = 15))

# Create folder for plots
if (!dir.exists("figures")) dir.create("figures")
if (!dir.exists("figures/model_performance")) dir.create("figures/model_performance")
if (!dir.exists("figures/Q1")) dir.create("figures/Q1")
if (!dir.exists("figures/Q2")) dir.create("figures/Q2")
if (!dir.exists("figures/Q3")) dir.create("figures/Q3")
if (!dir.exists("figures/Q4")) dir.create("figures/Q4")
if (!dir.exists("tables/Q4")) dir.create("tables/Q4",recursive = TRUE)

################################################################################
#                           MODEL PERFORMANCE
#
#       Show the performance of the models on the different datasets,
#       synthesizers, model types and runs.
################################################################################
res_perf <- data.table(readRDS("./results/model_performance/model_performance.rds"))

# Model Performance Examples ---------------------------------------------------
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

#ggsave("figures/model_performance/model_performance_examples.pdf", width = 4, height = 4)


# Whole model performance ------------------------------------------------------

# Plot for correct and incorrect predictions
thres1 <- 0.6
thres2 <- 0.8
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

# Selection
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

ggsave("figures/model_performance/model_performance.pdf", width = 12, height = 4)

'
# Full plot
ggplot(df, aes(x = value, y = syn_name, fill = variable)) +
  geom_bar(stat = "identity", position = "stack", width = 0.9) +
  geom_text(aes(label = ifelse(value > 10, value, "")), position = position_stack(vjust = 0.5)) +
  scale_fill_manual(values = c("darkgreen", "darkorange", "darkred", "gray")) +
  scale_x_continuous(expand = c(0,0)) +
  facet_grid(cols = vars(model_name), rows = vars(train), scales = "free") +
  labs(x = "Frequency", y = "Synthesizer", fill = "Accuracy") +
  theme(legend.position = "top",
        plot.margin = margin(0,0,0,0))
ggsave("figures/model_performance/model_performance_full.pdf", width = 12, height = 7)
'
################################################################################
#                           RESEARCH QUESTION 1 (Q1)
#
#       Which features and feature interactions were most challenging
#                       for the generative model?
################################################################################
res_q1 <- data.table(readRDS("./results/Q1/feature_importance.rds"))
res_treeshap <- c(
  readRDS("./results/Q3/intershap_1.rds"),
  readRDS("./results/Q3/intershap_2.rds"),
  readRDS("./results/Q3/intershap_3.rds"),
  readRDS("./results/Q3/intershap_4.rds")
)


# Plot for adult_complete ------------------------------------------------------

# PFI values
df_pfi <- res_q1[dataset_name == "adult_complete" & method == "PFI", c("feature", "value", "method")]

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

p1 <- ggplot(df, aes(y = feature, x = value, fill = method)) +
  geom_boxplot(position = "dodge") +
  geom_vline(xintercept = 0, linetype = "dashed") +
  facet_grid(cols = vars(type), scales = "free_x") +
  scale_fill_brewer(palette = "Dark2") +
  theme(legend.position = "top") +
  labs(x = "Importance", y = "Feature", fill = NULL)


# Interactions
top <- 20
df_interact_shap <- rbindlist(lapply(res_adult, sv_vi))
df_interact_mean <- df_interact_shap[, .(value = median(value)), by = c("var", "degree")]
vi_top <- df_interact_mean[order(-abs(value))][1:top]
df <- df_interact_shap[var %in% vi_top$var]
df[, degree := factor(degree)]
df[, var := factor(var, levels = rev(vi_top$var))]
df$type <- "global TreeSHAP (interactions)"
p2 <- ggplot(df, aes(x = var, y = value, fill = degree)) +
  geom_boxplot() +
  coord_flip() +
  scale_fill_viridis_d(direction = -1) +
  theme(legend.position = "top") +
  facet_grid(cols = vars(type), scales = "free_x") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(y = "Importance", x = NULL, fill = "degree")


p <- plot_grid(p1, p2, nrow = 1, rel_widths = c(0.6, 0.4), labels = c("(a)", "(b)"))
ggsave("figures/Q1/Q1_adult_complete.pdf", plot = p, width = 12, height = 5)

# Plot for nursery -------------------------------------------------------------

# PFI values
df_pfi <- res_q1[dataset_name == "nursery" & method == "PFI", c("feature", "value", "method")]

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


# Interactions
top <- 20
df_interact_shap <- rbindlist(lapply(res_nursery, sv_vi))
df_interact_mean <- df_interact_shap[, .(value = median(value)), by = c("var", "degree")]
vi_top <- df_interact_mean[order(-abs(value))][1:top]
df <- df_interact_shap[var %in% vi_top$var]
df[, degree := factor(degree)]
df[, var := factor(var, levels = rev(vi_top$var))]
df$type <- "global TreeSHAP (interactions)"
p2 <- ggplot(df, aes(x = var, y = value, fill = degree)) +
  geom_boxplot() +
  coord_flip() +
  scale_fill_viridis_d(direction = -1) +
  theme(legend.position = "top") +
  facet_grid(cols = vars(type), scales = "free_x") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(y = "Importance", x = NULL, fill = "degree")

p <- plot_grid(p1, p2, nrow = 1, rel_widths = c(0.6, 0.4), labels = c("(a)", "(b)"))

ggsave("figures/Q1/Q1_nursery.pdf", plot = p, width = 12, height = 5)


################################################################################
#                         RESEARCH QUESTION 2 (Q2)
#
#             TODO: Add description of the research question
################################################################################
res_q2 <- data.table(readRDS("./results/Q2/feat_effects.rds"))
res_q2_rugs <- data.table(readRDS("./results/Q2/feat_effects_rugs.rds"))
num_rugs <- 1000
num_ice <- 50

set.seed(42)

# Plot for adult_complete ------------------------------------------------------
df <- res_q2[dataset_name == "adult_complete", ]
df_rug <- res_q2_rugs[dataset_name == "adult_complete", ]

# ICE and grouped PDP plots
tmp <- lapply(unique(df$feature), function(feat) {
  df_ice <- df[feature == feat & method == "ice", ]
  df_pdp <- df[feature == feat & method == "pdp" & real == "both", ]
  df_rug_feat <- df_rug[variable == feat, ]

  if (all(df_ice$feat_type == "numeric")) {
    df_rug_feat$gridpoint <- df_rug_feat$value
    ids <- sample(unique(df_ice$id), num_ice)
    df_ice <- df_ice[id %in% ids]
    p <- ggplot(mapping = aes(x = as.numeric(as.character(gridpoint)))) +
      geom_line(data = df_ice, aes(group = id, color = real, y = value), alpha = 0.5, linewidth = 0.3) +
      geom_line(aes(group = real, y = value), data = df_pdp, linewidth = 1.5, color = "black") +
      geom_rug(data = df_rug_feat[sample(nrow(df_rug_feat), min(num_rugs, nrow(df_rug_feat)))], aes(color = real), sides = "b", alpha = 0.5) +
      facet_grid(cols = vars(feature), labeller = function(s) paste0("Feature: ", s)) +
      labs(x = "Feature value", y = "Prediction", color = NULL) +
      theme(legend.position = "top")
  } else {
    df_rug_feat <- df_rug_feat[, .(count = .N / nrow(df_rug_feat)), by = c("real", "value")]
    p <- ggplot(df_ice, aes(y = gridpoint, x = value)) +
      geom_bar(stat = "identity", data = df_rug_feat, aes(y = value, x = count, fill = real), alpha = 0.5, inherit.aes = FALSE) +
      labs(x = "Prediction", y = "Feature value", fill = NULL) +
      facet_grid(cols = vars(feature), labeller = function(s) paste0("Feature: ", s)) +
      geom_boxplot(fill = "darkgray") +
      stat_summary(geom = "crossbar", fun = "mean", color = "darkred", width = 0.75) +
      geom_vline(xintercept = 0.5, linetype = "dashed") +
      theme(legend.position = "top")
  }

  ggsave(paste0("figures/Q2/ICE_adult_complete_", feat, ".pdf"), p, width = 8, height = 5)

  p
})


# Create plot for paper
idx <- which(unique(df$feature) %in% c("education_num", "occupation"))
plots <- tmp[idx]
p <- plot_grid(plotlist = plots, nrow = 1, labels = c("(a)", "(b)"))
ggsave("figures/Q2/ICE_adult_complete.pdf", plot = p, width = 12, height = 5)


# ALE plots
tmp <- lapply(unique(df$feature), function(feat) {
  df_ale <- df[feature == feat & method == "ale", ]
  df_rug_feat <- df_rug[variable == feat, ]
  if (all(df_ale$feat_type == "numeric")) {
    df_rug_feat$gridpoint <- df_rug_feat$value
    p <- ggplot(df_ale, aes(x = as.numeric(as.character(gridpoint)))) +
      geom_rug(data = df_rug_feat[sample(nrow(df_rug_feat), min(num_rugs, nrow(df_rug_feat)))],
               aes(color = real), sides = "b", alpha = 0.5,
               show.legend = FALSE) +
      geom_hline(yintercept = 0) +
      facet_grid(cols = vars(feature), labeller = function(s) paste0("Feature: ", s)) +
      geom_line(aes(y = value), alpha = 0.5) +
      labs(x = "Feature value", y = "Prediction", color = NULL, linetype = NULL) +
      theme(legend.position = "top")
  } else {
    df_rug_feat <- df_rug_feat[, .(count = .N / nrow(df_rug_feat)), by = c("real", "value")]
    p <- ggplot(df_ale, aes(y = gridpoint, x = value)) +
      geom_bar(stat = "identity", position = "dodge", fill = "darkgray") +
      facet_grid(cols = vars(feature), labeller = function(s) paste0("Feature: ", s)) +
      geom_vline(xintercept = 0) +
      labs(x = "Prediction", y = "Feature value", fill = NULL) +
      theme(legend.position = "top")
  }

  ggsave(paste0("figures/Q2/ALE_adult_complete_", feat, ".pdf"), p, width = 8, height = 5)
  p
})

# Create plot for paper
idx <- which(unique(df$feature) %in% c("education_num", "occupation"))
plots <- tmp[idx]
p <- plot_grid(plotlist = plots, nrow = 1, labels = c("(a)", "(b)"))
ggsave("figures/Q2/ALE_adult_complete.pdf", plot = p, width = 12, height = 5)

# Plot for nursery -------------------------------------------------------------
df <- res_q2[dataset_name == "nursery", ]
df_rug <- res_q2_rugs[dataset_name == "nursery", ]

# PDP/ICE plots
tmp_pdp <- lapply(unique(df$feature), function(feat) {
  df_ice <- df[feature == feat & method == "ice", ]
  df_rug_feat <- df_rug[variable == feat, ]
  df_rug_feat <- df_rug_feat[, .(count = .N / nrow(df_rug_feat)), by = c("real", "value")]
  p <- ggplot(df_ice, aes(y = gridpoint, x = value)) +
    geom_bar(stat = "identity", data = df_rug_feat, aes(y = value, x = count, fill = real), alpha = 0.5, inherit.aes = FALSE) +
    facet_grid(cols = vars(feature), labeller = function(s) paste0("Feature: ", s)) +
    geom_boxplot(fill = "darkgray") +
    stat_summary(geom = "crossbar", fun = "mean", color = "darkred", width = 0.75) +
    geom_vline(xintercept = 0.5, linetype = "dashed") +
    labs(x = "Prediction", y = "Feature value", fill = NULL) +
    theme(legend.position = "top")

  ggsave(paste0("figures/Q2/ICE_nursery_", feat, ".pdf"), p, width = 8, height = 5)
  p
})

# ALE plots
tmp_ale <- lapply(unique(df$feature), function(feat) {
  df_ale <- df[feature == feat & method == "ale", ]
  df_rug_feat <- df_rug[variable == feat, ]
  df_rug_feat <- df_rug_feat[, .(count = .N / nrow(df_rug_feat)), by = c("real", "value")]
  p <- ggplot(df_ale, aes(y = gridpoint, x = value)) +
    geom_bar(stat = "identity", position = "dodge", fill = "darkgray") +
    facet_grid(cols = vars(feature), labeller = function(s) paste0("Feature: ", s)) +
    geom_vline(xintercept = 0) +
    labs(x = "Prediction", y = "Feature value", fill = NULL) +
    theme(legend.position = "top")

  ggsave(paste0("figures/Q2/ALE_adult_complete_", feat, ".pdf"), p, width = 8, height = 5)
  p
})


# Create figure from the paper
ids <- which(unique(df$feature) %in% c("class", "form"))

p1 <- plot_grid(tmp_pdp[[ids[1]]],
                tmp_ale[[ids[1]]] +
                  ylab(NULL) + theme(axis.text.y = element_blank()),
                nrow = 1, align = "h", rel_widths = c(0.6, 0.4), labels = c("(a)", "(b)"))
p2 <- plot_grid(tmp_pdp[[ids[2]]],
                tmp_ale[[ids[2]]] +
                  ylab(NULL) + theme(axis.text.y = element_blank()),
                nrow = 1, align = "h", rel_widths = c(0.6, 0.4), labels = c("(c)", "(d)"))
p <- plot_grid(p1, p2, nrow = 1, align = "h")

ggsave("figures/Q2/ICE_ALE_nursery.pdf", plot = p, width = 12, height = 4)


################################################################################
#                         RESEARCH QUESTION 3 (Q3)
#
#             TODO: Add description of the research question
################################################################################

res_condshap <- fread("./results/Q3/condshap_final.csv")


if (!dir.exists("figures/Q3")) dir.create("figures/Q3",recursive = TRUE)

source("utils.R")

# Consider these types of observations:
# large differences between ctree and indep. Check if where it differs is seen from pairplots of the data
#

# adult_complete

# First considering synthetic observations
this_res_condshap <- res_condshap[dataset_name=="adult_complete" &
                                    syn_name == "TabSyn" &
                                    run_model==2 &
                                    model_name == "xgboost" &
                                    type == "syn"]

features_cols <- this_res_condshap[,unique(feature)]

#### Simplest way to get feature values
res_ce_values <- fread("./results/Q4/ce_values_final.csv")
this_res_ce_values <- res_ce_values[dataset_name=="adult_complete" &
                                      syn_name == "TabSyn" &
                                      run_model==2 &
                                      model_name == "xgboost" &
                                      type == "syn"]

this_res_ce_values[row_type=="org"]
features_dt <- dcast(this_res_ce_values[row_type=="org",.(rowid_test,variable,value)],formula = rowid_test~variable)

feature_vals_dt <- features_dt[,..features_cols]
####

# Flattening ctree and indep shapley values
ctree_dt <- dcast(this_res_condshap[approach=="ctree",.(rowid_test,feature,value)],formula = rowid_test~feature)
indep_dt <- dcast(this_res_condshap[approach=="independence",.(rowid_test,feature,value)],formula = rowid_test~feature)


# Plot with ctree and indep shapley values

this_rowid_test= 1353


a <- sv_force_shapviz_mod3(ctree_dt[rowid_test==this_rowid_test,..features_cols],
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

b <- sv_force_shapviz_mod3(indep_dt[rowid_test==this_rowid_test,..features_cols],
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

pl_cond_vs_marg <- (a/b) #+
  #patchwork::plot_annotation(
  #title = paste0("Shapley value feature attributions, syntehtic data, test id = ",this_rowid_test),
  #theme = theme(plot.title = element_text(hjust = 0.5,size = 18)))

pdf(paste0("figures/Q3/Q3_adult_complete_cond_vs_marg_syn_id_",this_rowid_test,".pdf"),width = 10, height = 3.2)
print(pl_cond_vs_marg)
dev.off()

### Real observations ###

this_res_condshap <- res_condshap[dataset_name=="adult_complete" &
                                    syn_name == "TabSyn" &
                                    run_model==2 &
                                    model_name == "xgboost" &
                                    type == "real"]


#### Simplest way to get feature values
res_ce_values <- fread("./results/Q4/ce_values_final.csv")
this_res_ce_values <- res_ce_values[dataset_name=="adult_complete" &
                                      syn_name == "TabSyn" &
                                      run_model==2 &
                                      model_name == "xgboost" &
                                      type == "real"]

this_res_ce_values[row_type=="org"]
features_dt <- dcast(this_res_ce_values[row_type=="org",.(rowid_test,variable,value)],formula = rowid_test~variable)

feature_vals_dt <- features_dt[,..features_cols]
####

# Flattening ctree and indep shapley values
ctree_dt <- dcast(this_res_condshap[approach=="ctree",.(rowid_test,feature,value)],formula = rowid_test~feature)
indep_dt <- dcast(this_res_condshap[approach=="independence",.(rowid_test,feature,value)],formula = rowid_test~feature)





this_rowid_test= 16025



a <- sv_force_shapviz_mod3(ctree_dt[rowid_test==this_rowid_test,..features_cols],
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


pl_cond_only <- a #+ patchwork::plot_annotation(
#  title = paste0("Shapley value feature attributions, real data, test id = ",this_rowid_test),
#  theme = theme(plot.title = element_text(hjust = 0.5,size = 18)))

pdf(paste0("figures/Q3/Q3_adult_complete_cond_only_real_id_",this_rowid_test,".pdf"),width = 10, height = 2.3)
print(pl_cond_only)
dev.off()




### intershap

res_intershap <- readRDS("./results/Q3/intershap.rds")


info_dt_intershap <- as.data.table(t(sapply(res_intershap, function(x) unlist(x$info[c("dataset","syn","run_model","detect_model")]))))
info_dt_intershap[,row_id:=.I]

# First considering synthetic observations
this_intershap_index <- info_dt_intershap[dataset=="adult_complete" &
                                            syn=="TabSyn" &
                                            run_model==2 &
                                            detect_model=="xgboost",row_id]

this_res_intershap <- res_intershap[[this_intershap_index]]$results
this_info_intershap <- res_intershap[[this_intershap_index]]$info

# Getting observations to plot
relevant_test_obs <- fread("./prepare_local/relevant_test_obs.csv")

this_relevant_test_obs <- relevant_test_obs[dataset_name=="adult_complete" &
                                              syn_name == "TabSyn" &
                                              run_model==2 &
                                              model_name == "xgboost" &
                                              type == "syn"]

shapviz_row_mapper <- data.table(rowid_testobs = this_relevant_test_obs[,rowid])
shapviz_row_mapper[,shapviz_rowno := match(rowid_testobs,this_info_intershap$rowid)]


this_rowid_test= 1353

this_pred_logodds <- sum(this_res_intershap$S[shapviz_row_mapper[rowid_testobs==this_rowid_test,shapviz_rowno],])+this_res_intershap$baseline
this_pred <- 1/(1+exp(-this_pred_logodds))

pl_inter <- plot_waterfall(this_res_intershap,
                           row_id = shapviz_row_mapper[rowid_testobs==this_rowid_test,shapviz_rowno],
                           fill_colors = c("darkgreen","darkred"),
                           marg_int_colors = c("orange","purple4"),annotation_size = 5)+
  #               marg_int_colors = viridisLite::viridis(2))+
  theme(plot.title = element_text(size = 16,face = "bold",hjust=0.5),
        axis.text = element_text(size = 14),
        axis.title = element_text(size = 14),
        axis.title.y = element_text(size=16, face = "bold"))#+
#  patchwork::plot_annotation(
#    title = paste0("Shapley interaction values, synthetic data, test id = ",shapviz_row_mapper[rowid_testobs==this_rowid_test,rowid_testobs], " with C(x) = ",round(this_pred,3)),
#    theme = theme(plot.title = element_text(hjust = 0.5,size = 18)))


pdf(paste0("figures/Q3/Q3_adult_complete_inter_syn_id_",this_rowid_test,".pdf"),width = 10, height = 3.6)
print(pl_inter)
dev.off()


# real
this_relevant_test_obs <- relevant_test_obs[dataset_name=="adult_complete" &
                                              syn_name == "TabSyn" &
                                              run_model==2 &
                                              model_name == "xgboost" &
                                              type == "real"]

shapviz_row_mapper <- data.table(rowid_testobs = this_relevant_test_obs[,rowid])
shapviz_row_mapper[,shapviz_rowno := match(rowid_testobs,this_info_intershap$rowid)]


this_rowid_test= 16025

this_pred_logodds <- sum(this_res_intershap$S[shapviz_row_mapper[rowid_testobs==this_rowid_test,shapviz_rowno],])+this_res_intershap$baseline
this_pred <- 1/(1+exp(-this_pred_logodds))


pl_inter <- plot_waterfall(this_res_intershap,
                           row_id = shapviz_row_mapper[rowid_testobs==this_rowid_test,shapviz_rowno],
                           fill_colors = c("darkgreen","darkred"),
                           marg_int_colors = c("orange","purple4"),annotation_size = 4)+
  #               marg_int_colors = viridisLite::viridis(2))+
  theme(plot.title = element_text(size = 16,face = "bold",hjust=0.5),
        axis.text = element_text(size = 14),
        axis.title = element_text(size = 14),
        axis.title.y = element_text(size=16, face = "bold"))#+
#  patchwork::plot_annotation(
#    title = paste0("Shapley interaction values, real data, test id = ",shapviz_row_mapper[rowid_testobs==this_rowid_test,rowid_testobs], " with C(x) = ",round(this_pred,3)),
#    theme = theme(plot.title = element_text(hjust = 0.5,size = 18)))

pdf(paste0("figures/Q3/Q3_adult_complete_inter_real_id_",this_rowid_test,".pdf"),width = 10, height = 3.6)
print(pl_inter)
dev.off()


# TODO: Merge the two plots

pl_cond_only/pl_inter



#### nursery dataset


############# intershap for nursery


# First considering synthetic observations
this_intershap_index <- info_dt_intershap[dataset=="nursery" &
                                            syn == "CTGAN" &
                                            run_model==8 &
                                            detect_model == "xgboost", row_id]

this_res_intershap <- res_intershap[[this_intershap_index]]$results
this_info_intershap <- res_intershap[[this_intershap_index]]$info

# Getting observations to plot
relevant_test_obs <- fread("./prepare_local/relevant_test_obs.csv")

this_relevant_test_obs <- relevant_test_obs[dataset_name=="nursery" &
                                              syn_name == "CTGAN" &
                                              run_model==8 &
                                              model_name == "xgboost" &
                                              type == "syn"]

shapviz_row_mapper <- data.table(rowid_testobs = this_relevant_test_obs[,rowid])
shapviz_row_mapper[,shapviz_rowno := match(rowid_testobs,this_info_intershap$rowid)]


this_rowid_test= 1342

this_pred_logodds <- sum(this_res_intershap$S[shapviz_row_mapper[rowid_testobs==this_rowid_test,shapviz_rowno],])+this_res_intershap$baseline
this_pred <- 1/(1+exp(-this_pred_logodds))


pl_inter <- plot_waterfall(this_res_intershap,
                           row_id = shapviz_row_mapper[rowid_testobs==this_rowid_test,shapviz_rowno],
                           fill_colors = c("darkgreen","darkred"),
                           marg_int_colors = c("orange","purple4"),annotation_size = 5)+
  #               marg_int_colors = viridisLite::viridis(2))+
  theme(plot.title = element_text(size = 16,face = "bold",hjust=0.5),
        axis.text = element_text(size = 14),
        axis.title = element_text(size = 14),
        axis.title.y = element_text(size=16, face = "bold"))#+
#  patchwork::plot_annotation(
#    title = paste0("Shapley interaction values, synthetic data, test id = ",shapviz_row_mapper[rowid_testobs==this_rowid_test,rowid_testobs], " with C(x) = ",round(this_pred,3)),
#    theme = theme(plot.title = element_text(hjust = 0.5)))

pdf(paste0("figures/Q3/Q3_nursery_inter_syn_id_",this_rowid_test,".pdf"),width = 10, height = 3.4)
print(pl_inter)
dev.off()



################################################################################
#                         RESEARCH QUESTION 4 (Q4)
#
#             TODO: Add description of the research question
################################################################################

# Without fixing flnwgt

res_ce_values <- fread("./results/Q4/ce_values_extra2.csv")
res_ce_measures <- fread("./results/Q4/ce_measures_extra2.csv")

# adult_complete


this_res_ce_measures <- res_ce_measures
this_res_ce_values <- res_ce_values



# Reduce to those with the largest L0 measure
features_cols <- unique(this_res_ce_values$variable)

this_rowid <- 1353

these_cf_ranks <- c(3,17,9,16) # 13


tab_list <- list()
tab_final <- NULL
for(i in seq_along(these_cf_ranks)){

  this_cf_rank <- these_cf_ranks[i]


  tmp <- this_res_ce_values[rowid_test==this_rowid & counterfactual_rank%in%c(this_cf_rank,NA),.(variable,value,row_type)]

  tab <- data.table(org=tmp[row_type=="org"][,value], cf=tmp[row_type=="cf"][,value])


  #  tab <- dcast(this_res_ce_values[rowid_test==this_rowid,.(variable,value,row_type)],formula = row_type~variable)

  #tab[org!=cf, `:=`(org=paste0("\\textcolor{red}{",org,"}"),
  #                 cf=paste0("\\textcolor{red}{",cf,"}"))]
  tab[org!=cf, `:=`(cf=paste0("\\textcolor{red}{",cf,"}"))]

  if(i==1){
    tab_final <- cbind(tab_final,tab)
  } else {
    tab_final <- cbind(tab_final,tab[,-1])
  }


}

colnames(tab_final)[1] <- "Original"
colnames(tab_final)[-1] <- paste0("CF",seq_along(these_cf_ranks))


tab_all <- as.data.frame(tab_final)

feature_cols <- unique(this_res_ce_values$variable)
rownames(tab_all) <- feature_cols
#addtorow <- list()
#addtorow$pos <- list(0)
#addtorow$command <- paste0("Feature",paste0('& \\multicolumn{3}{c|}{ test id = ',this_rowid , '}', collapse=''), '\\\\')

align_vector <- paste0("|l|r|",paste0(rep("r",length(these_cf_ranks)),collapse=""),"|")

caption0 <- paste0("Four counterfactual explanations for a synthetic sample with test id ",this_rowid, " in the adult data set.",
                   " Features in red font are changed.")

print(xtable(tab_all,align = align_vector,
             caption = caption0, label = paste0("tab:Q4_adult_complete_ce_syn_id_",this_rowid)),
      sanitize.text.function = identity,
      sanitize.rownames.function = NULL,
      include.rownames = TRUE,
      include.colnames = TRUE,
      booktabs = TRUE,
      size="\\fontsize{8pt}{8pt}\\selectfont",
      #add.to.row=addtorow,
      file = paste0("tables/Q4/Q4_adult_complete_ce_syn_id_",this_rowid,"_v2.tex")
)


#### nursery

res_ce_values <- fread("./results/Q4/ce_values_extra3.csv")
res_ce_measures <- fread("./results/Q4/ce_measures_extra3.csv")

this_res_ce_measures <- res_ce_measures
this_res_ce_values <- res_ce_values



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


  #  tab <- dcast(this_res_ce_values[rowid_test==this_rowid,.(variable,value,row_type)],formula = row_type~variable)

  #tab[org!=cf, `:=`(org=paste0("\\textcolor{red}{",org,"}"),
  #                 cf=paste0("\\textcolor{red}{",cf,"}"))]
  tab[org!=cf, `:=`(cf=paste0("\\textcolor{red}{",cf,"}"))]

  if(i==1){
    tab_final <- cbind(tab_final,tab)
  } else {
    tab_final <- cbind(tab_final,tab[,-1])
  }


}

colnames(tab_final)[1] <- "Original"
colnames(tab_final)[-1] <- paste0("CF",seq_along(these_cf_ranks))

cols <- colnames(tab_final)
tab_final[, (cols) := lapply(.SD, function(x) gsub("_", "-", x)), .SDcols = cols]


tab_all <- as.data.frame(tab_final)

feature_cols <- unique(this_res_ce_values$variable)
rownames(tab_all) <- feature_cols



#addtorow <- list()
#addtorow$pos <- list(0)
#addtorow$command <- paste0("Feature",paste0('& \\multicolumn{3}{c|}{ test id = ',this_rowid , '}', collapse=''), '\\\\')

align_vector <- paste0("|l|r|",paste0(rep("r",length(these_cf_ranks)),collapse=""),"|")

caption0 <- paste0("Four counterfactual explanations for a synthetic sample with test id ",this_rowid, " in the nursery data set.",
                   " Features in red font are changed.")

print(xtable(tab_all,align = align_vector,
             caption = caption0, label = paste0("tab:Q4_nursery_ce_syn_id_",this_rowid)),
      sanitize.text.function = identity,
      sanitize.rownames.function = NULL,
      include.rownames = TRUE,
      include.colnames = TRUE,
      booktabs = TRUE,
      size="\\fontsize{8pt}{8pt}\\selectfont",
      #add.to.row=addtorow,
      file = paste0("tables/Q4/Q4_nursery_ce_syn_id_",this_rowid,".tex")
)



