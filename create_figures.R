################################################################################
#                 Create all Figures for the manuscript
################################################################################
library(ggplot2)
library(patchwork)
library(geomtextpath)
library(rlang)
library(data.table)
library(xtable)################################################################################
#                 Create all Figures for the manuscript
################################################################################
library(ggplot2)
library(cowplot)
library(geomtextpath)
library(rlang)
library(data.table)

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
     (dataset == "nursery" & syn_name == "CTGAN")) & model_name == "xgboost"]
ggplot(df, aes(x = dataset, y = value, fill = metric)) +
  geom_boxplot() +
  geom_texthline(yintercept = 0.5, linetype = "dashed", label = "Random guessing") +
  facet_grid(cols = vars(train), scales = "free") +
  scale_fill_brewer(palette = "Dark2") +
  labs(x = "Dataset", fill = "Metric", y = NULL) +
  theme(legend.position = "top")

ggsave("figures/model_performance/model_performance_examples.pdf", width = 8, height = 5)


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

df$variable <- factor(df$variable, levels = c("correct", "middle", "incorrect", "NA"),
                      labels = c(paste0("Above ", thres2 * 100, "%"), 
                                 paste0("Between ", thres1 * 100, "% and ", thres2 * 100, "%"),
                                 paste0("Below ", thres1 * 100, "%"), "NA"))

# Selection
ggplot(df[train == "test data"], 
       aes(x = value, y = syn_name, fill = variable)) +
  geom_bar(stat = "identity", position = "stack", width = 0.9) +
  geom_text(aes(label = ifelse(value > 10, value, "")), position = position_stack(vjust = 0.5)) +
  scale_fill_manual(values = c("darkgreen", "darkorange", "darkred", "gray")) +
  scale_x_continuous(expand = c(0,0)) +
  facet_grid(cols = vars(model_name), rows = vars(train), scales = "free") +
  labs(x = "Frequency", y = "Synthesizer", fill = "Accuracy") +
  theme(legend.position = "top",
        plot.margin = margin(0,0,0,0))
ggsave("figures/model_performance/model_performance_selection.pdf", width = 8, height = 4)

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

################################################################################
#                           RESEARCH QUESTION 1 (Q1)
#
#       Which features and feature interactions were most challenging 
#                       for the generative model?
################################################################################
res_q1 <- data.table(readRDS("./results/Q1/feature_importance.rds"))
res_treeshap <- readRDS("./results/Q3/intershap.rds")

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
             method = "|TreeSHAP|")
}))

# Create plot
df <- rbind(df_pfi, df_marginal_shap)
df$type <- ifelse(df$method == "|TreeSHAP|", "Prediction-based", "Loss-based")

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
df$type <- "TreeSHAP (interactions)"
p2 <- ggplot(df, aes(x = var, y = value, fill = degree)) + 
  geom_boxplot() +
  coord_flip() + 
  scale_fill_viridis_d(direction = -1) +
  theme(legend.position = "top") +
  facet_grid(cols = vars(type), scales = "free_x") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(y = "Importance", x = NULL, fill = "degree")


p <- plot_grid(p1, p2, nrow = 1, rel_widths = c(0.6, 0.4), labels = c("A)", "B)"))
ggsave("figures/Q1/adult_complete.pdf", plot = p, width = 12, height = 5)

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
             method = "|TreeSHAP|")
}))

# Create plot
df <- rbind(df_pfi, df_marginal_shap)
df$type <- ifelse(df$method == "|TreeSHAP|", "Prediction-based", "Loss-based")

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
df$type <- "TreeSHAP (interactions)"
p2 <- ggplot(df, aes(x = var, y = value, fill = degree)) + 
  geom_boxplot() +
  coord_flip() + 
  scale_fill_viridis_d(direction = -1) +
  theme(legend.position = "top") +
  facet_grid(cols = vars(type), scales = "free_x") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(y = "Importance", x = NULL, fill = "degree")

p <- plot_grid(p1, p2, nrow = 1, rel_widths = c(0.6, 0.4), labels = c("A)", "B)"))

ggsave("figures/Q1/nursery.pdf", plot = p, width = 12, height = 5)


################################################################################
#                         RESEARCH QUESTION 2 (Q2)
#
#             TODO: Add description of the research question
################################################################################
res_q2 <- data.table(readRDS("./results/Q2/feat_effects.rds"))
res_q2_rugs <- data.table(readRDS("./results/Q2/feat_effects_rugs.rds"))

# Plot for adult_complete ------------------------------------------------------
df <- res_q2[dataset_name == "adult_complete", ]
df_rug <- res_q2_rugs[dataset_name == "adult_complete", ]

# ICE and grouped PDP plots
tmp <- lapply(unique(df$feature), function(feat) {
  df_ice <- df[feature == feat & method == "ice", ]
  df_pdp <- df[feature == feat & method == "pdp" & real != "both", ]
  df_pdp_full <- df[feature == feat & method == "pdp", ]
  df_rug_feat <- df_rug[variable == feat, ]
  
  if (all(df_ice$feat_type == "numeric")) {
    df_rug_feat$gridpoint <- df_rug_feat$value
    p <- ggplot(mapping = aes(x = as.numeric(as.character(gridpoint)))) +
      geom_line(data = df_ice, aes(group = id, color = real, y = value), alpha = 0.5, linewidth = 0.3) +
      geom_line(aes(group = real, y = value), data = df_pdp, linewidth = 1.5, color = "black") +
      geom_line(aes(color = real, y = value), data = df_pdp, linewidth = 1, linetype = "dashed") +
      geom_rug(data = df_rug_feat, aes(color = real), sides = "b", alpha = 0.5) +
      facet_grid(cols = vars(feature), labeller = function(s) paste0("Feature: ", s)) +
      labs(x = "Feature value", y = "Prediction", color = NULL) +
      theme(legend.position = "top")
  } else {
    df_rug_feat <- df_rug_feat[, .(count = .N / nrow(df_rug_feat)), by = c("real", "value")]
    p <- ggplot(df_ice, aes(y = gridpoint, x = value, fill = real)) +
      geom_bar(stat = "identity", data = df_rug_feat, aes(y = value, x = count, fill = real), alpha = 0.5, inherit.aes = FALSE) +
      labs(x = "Prediction", y = "Feature value", fill = NULL) +
      facet_grid(cols = vars(feature), labeller = function(s) paste0("Feature: ", s)) +
      geom_boxplot() + 
      theme(legend.position = "top")
  }
  
  ggsave(paste0("figures/Q2/ICE_adult_complete_", feat, ".pdf"), p, width = 8, height = 5)
})

# ALE and PDP plots
tmp <- lapply(unique(df$feature), function(feat) {
  df_ale <- df[feature == feat & method == "ale", ]
  df_pdp <- df[feature == feat & method == "pdp" & real == "both", ]
  df_ale <- rbind(df_ale, df_pdp)
  df_rug_feat <- df_rug[variable == feat, ]
  
  if (all(df_ale$feat_type == "numeric")) {
    df_rug_feat$gridpoint <- df_rug_feat$value
    p <- ggplot(df_ale, aes(x = as.numeric(as.character(gridpoint)))) +
      geom_rug(data = df_rug_feat, aes(color = real), sides = "b", alpha = 0.5,
               show.legend = FALSE) +
      geom_hline(yintercept = 0) +
      facet_grid(cols = vars(feature), labeller = function(s) paste0("Feature: ", s)) +
      geom_line(aes(y = value, linetype = method), alpha = 0.5) +
      labs(x = "Feature value", y = "Prediction", color = NULL, linetype = NULL) +
      theme(legend.position = "top")
  } else {
    df_rug_feat <- df_rug_feat[, .(count = .N / nrow(df_rug_feat)), by = c("real", "value")]
    p <- ggplot(df_ale, aes(y = gridpoint, x = value, fill = method)) +
      geom_bar(stat = "identity", position = "dodge") +
      facet_grid(cols = vars(feature), labeller = function(s) paste0("Feature: ", s)) +
      geom_vline(xintercept = 0) +
      labs(x = "Prediction", y = "Feature value", fill = NULL) +
      theme(legend.position = "top")
  }
  
  ggsave(paste0("figures/Q2/ALE_adult_complete_", feat, ".pdf"), p, width = 8, height = 5)
})

# Plot for nursery -------------------------------------------------------------
df <- res_q2[dataset_name == "nursery", ]
df_rug <- res_q2_rugs[dataset_name == "nursery", ]

# ALE and PDP plots
tmp <- lapply(unique(df$feature), function(feat) {
  df_ale <- df[feature == feat & method == "ale", ]
  df_pdp <- df[feature == feat & method == "pdp" & real == "both", ]
  df_ale <- rbind(df_ale, df_pdp)
  df_rug_feat <- df_rug[variable == feat, ]
  
  if (all(df_ale$feat_type == "numeric")) {
    df_rug_feat$gridpoint <- df_rug_feat$value
    p <- ggplot(df_ale, aes(x = as.numeric(as.character(gridpoint)))) +
      geom_rug(data = df_rug_feat, aes(color = real), sides = "b", alpha = 0.5,
               show.legend = FALSE) +
      geom_hline(yintercept = 0) +
      facet_grid(cols = vars(feature), labeller = function(s) paste0("Feature: ", s)) +
      geom_line(aes(y = value, linetype = method), alpha = 0.5) +
      labs(x = "Feature value", y = "Prediction", color = NULL, linetype = NULL) +
      theme(legend.position = "top")
  } else {
    df_rug_feat <- df_rug_feat[, .(count = .N / nrow(df_rug_feat)), by = c("real", "value")]
    p <- ggplot(df_ale, aes(y = gridpoint, x = value, fill = method)) +
      geom_bar(stat = "identity", position = "dodge") +
      facet_grid(cols = vars(feature), labeller = function(s) paste0("Feature: ", s)) +
      geom_vline(xintercept = 0) +
      labs(x = "Prediction", y = "Feature value", fill = NULL) +
      theme(legend.position = "top")
  }
  
  ggsave(paste0("figures/Q2/ALE_nursery_", feat, ".pdf"), p, width = 8, height = 5)
})


################################################################################
#                         RESEARCH QUESTION 4 (Q4)
#
#             TODO: Add description of the research question
################################################################################
res_ce_values <- fread("./results/Q4/ce_values_final.csv")
res_ce_measures <- fread("./results/Q4/ce_measures_final.csv")

# adult_complete

this_res_ce_measures <- res_ce_measures[dataset_name=="adult_complete" &
                                          syn_name == "TabSyn" &
                                          run_model==2 &
                                          model_name == "xgboost" &
                                          type == "syn"]

this_res_ce_values <- res_ce_values[dataset_name=="adult_complete" &
                                          syn_name == "TabSyn" &
                                          run_model==2 &
                                          model_name == "xgboost" &
                                          type == "syn"]


# Reduce to those with the largest L0 measure
this_res_ce_measures <- this_res_ce_measures[measure_L0 ==max(measure_L0)]
features_cols <- unique(this_res_ce_values$variable)

tab_final <- NULL
for(i in seq_len(nrow(this_res_ce_measures))){
  this_rowid <- this_res_ce_measures[i,rowid_test]

  tmp <- this_res_ce_values[rowid_test==this_rowid,.(variable,value,row_type)]
  tab <- data.table(org=tmp[row_type=="org"][,value], cf=tmp[row_type=="cf"][,value])

#  tab <- dcast(this_res_ce_values[rowid_test==this_rowid,.(variable,value,row_type)],formula = row_type~variable)

  tab[org!=cf, `:=`(org=paste0("\\textcolor{red}{",org,"}"),
                    cf=paste0("\\textcolor{red}{",cf,"}"))]

  tab_final <- cbind(tab_final,tab)
}

tab_all <- as.data.frame(tab_final)
rownames(tab_all) <- features_cols

no_plots <- ceiling(length(this_res_ce_measures[,rowid_test])/2)

for(i in seq_len(no_plots)){
  these_plots <- 4*(i-1)+1:4
  these_plots <- these_plots[these_plots<=ncol(tab_all)]

  these_test_ids <- this_res_ce_measures[,rowid_test][2*(i-1)+1:2]
  these_test_ids <- these_test_ids[!is.na(these_test_ids)]

  tab <- tab_all[,these_plots]

  addtorow <- list()
  addtorow$pos <- list(0)
  addtorow$command <- paste0("Feature",paste0('& \\multicolumn{2}{c|}{ test id = ',these_test_ids , '}', collapse=''), '\\\\')

  align_vector <- paste0("|l|",paste0(rep("rr|",length(these_test_ids)),collapse=""))

  caption0 <- paste0("Counterfactual explanations for the synthetic observations with small probabilities of being real ",
                     "for the adult data set.",collapse = "")

  print(xtable(tab,align = align_vector,
               caption = caption0),
        sanitize.text.function = identity,
        sanitize.rownames.function = NULL,
        include.rownames = TRUE,
        include.colnames = FALSE,
        booktabs = TRUE,
        add.to.row=addtorow,
        file = paste0("tables/Q4/ce_plot_",i,".tex")
  )

}




res_ce_values


# force plots

# something we didnt

# Some comoinations which are not shown

# counterfactuals for the same

# age integer va non.integer






