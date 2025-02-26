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
dir.create("figures", showWarnings = FALSE)

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

ggsave("figures/model_performance_examples.pdf", width = 8, height = 5)


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
  scale_fill_manual(values = c("darkred", "darkorange", "darkgreen", "gray")) +
  scale_x_continuous(expand = c(0,0)) +
  facet_grid(cols = vars(model_name), rows = vars(train), scales = "free") +
  labs(x = "Frequency", y = "Synthesizer", fill = "Accuracy") +
  theme(legend.position = "top",
        plot.margin = margin(0,0,0,0))
ggsave("figures/model_performance_selection.pdf", width = 8, height = 4)

# Full plot
ggplot(df, aes(x = value, y = syn_name, fill = variable)) +
  geom_bar(stat = "identity", position = "stack", width = 0.9) +
  geom_text(aes(label = ifelse(value > 10, value, "")), position = position_stack(vjust = 0.5)) +
  scale_fill_manual(values = c("darkred", "darkorange", "darkgreen", "gray")) +
  scale_x_continuous(expand = c(0,0)) +
  facet_grid(cols = vars(model_name), rows = vars(train), scales = "free") +
  labs(x = "Frequency", y = "Synthesizer", fill = "Accuracy") +
  theme(legend.position = "top",
        plot.margin = margin(0,0,0,0))
ggsave("figures/model_performance_full.pdf", width = 12, height = 7)

################################################################################
#                           RESEARCH QUESTION 1 (Q1)
#
#       Which features and feature interactions were most challenging
#                       for the generative model?
################################################################################
res_q1 <- data.table(rbind(
  readRDS("./results/Q1/feature_importance.rds"),
  readRDS("./results/Q1/cond_feature_importance.rds")
))

# Plot for adult_complete ------------------------------------------------------
df <- res_q1[dataset_name == "adult_complete", ]
df$type <- ifelse(df$method == "Model (impurity)", "model-specific", "model-agnostic")

ggplot(df, aes(y = feature, x = value, fill = method)) +
  geom_boxplot(position = "dodge") +
  geom_vline(xintercept = 0, linetype = "dashed") +
  facet_grid(cols = vars(type), scales = "free_x") +
  scale_fill_brewer(palette = "Dark2") +
  theme(legend.position = "top") +
  labs(x = "Importance", y = "Feature", fill = NULL)

ggsave("figures/Q1_adult_complete.pdf", width = 9, height = 5)

# Plot for nursery -------------------------------------------------------------
df <- res_q1[dataset_name == "nursery", ]
df$type <- ifelse(df$method == "Model (impurity)", "model-specific", "model-agnostic")

ggplot(df, aes(y = feature, x = value, fill = method)) +
  geom_boxplot(position = "dodge") +
  geom_vline(xintercept = 0, linetype = "dashed") +
  facet_grid(cols = vars(type), scales = "free_x") +
  scale_fill_brewer(palette = "Dark2") +
  theme(legend.position = "top") +
  labs(x = "Importance", y = "Feature", fill = NULL)

ggsave("figures/Q1_nursery.pdf", width = 9, height = 5)


################################################################################
#                         RESEARCH QUESTION 2 (Q2)
#
#             TODO: Add description of the research question
################################################################################
res_q2 <- data.table(readRDS("./results/Q2/feat_effects.rds"))
res_q2_rugs <- data.table(readRDS("./results/Q2/feat_effects_rugs.rds"))

if (!dir.exists("figures/Q2")) dir.create("figures/Q2")

# Plot for adult_complete ------------------------------------------------------
df <- res_q2[dataset_name == "adult_complete", ]
df_rug <- res_q2_rugs[dataset_name == "adult_complete", ]
tmp <- lapply(unique(df$feature), function(feat) {
  df_ice <- df[feature == feat & method == "ice", ]
  df_pdp <- df[feature == feat & method == "pdp" & real != "both", ]
  df_pdp_full <- df[feature == feat & method == "pdp", ]
  df_rug_feat <- df_rug[variable == feat, ]

  if (all(df_ice$feat_type == "numeric")) {
    df_rug_feat$gridpoint <- df_rug_feat$value
    p <- ggplot(mapping = aes(x = as.numeric(as.character(gridpoint)))) +
      geom_line(data = df_ice, aes(group = id, color = real, y = value), alpha = 0.5, linewidth = 0.3) +
      geom_line(aes(group = real, y = value), data = df_pdp_full, color = "gray25", linewidth = 1) +
      geom_line(aes(color = real, y = value), data = df_pdp, linewidth = 1, linetype = "dashed") +
      geom_rug(data = df_rug_feat, aes(color = real), sides = "b", alpha = 0.5) +
      labs(x = "Feature value", y = "Prediction", color = NULL) +
      theme(legend.position = "top")

    if (feat %in% c("capital_gain", "capital_loss")) {
      p <- p + scale_x_continuous(transform = "log2")
    }
  } else {
    df_rug_feat <- df_rug_feat[, .(count = .N / nrow(df_rug_feat)), by = c("real", "value")]
    p <- ggplot(df_ice, aes(y = gridpoint, x = value, fill = real)) +
      geom_bar(stat = "identity", data = df_rug_feat, aes(y = value, x = count, fill = real), alpha = 0.5, inherit.aes = FALSE) +
      labs(x = "Prediction", y = "Feature value", fill = NULL) +
      geom_boxplot() +
      theme(legend.position = "top")
  }

  ggsave(paste0("figures/Q2/Q2_adult_complete_", feat, ".pdf"), p, width = 8, height = 5)
})

res <- res_q2[dataset_name == "adult_complete" & method %in% c("pdp", "ale") & feature == "age"]
res$value <- ifelse(res$method == "pdp", res$value - 0.5, res$value)
ggplot(res) +
  geom_line(aes(x = as.numeric(as.character(gridpoint)), y = value, color = method, group = method), alpha = 0.5)



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

# Compare their sizes
rowid_test_vec <- ctree_dt[,rowid_test]

comp_dt <- data.table(rowid_test=rowid_test_vec,meandiff=rowMeans(abs(ctree_dt[,-1]-indep_dt[,-1])))
these_rowid <- comp_dt[order(-meandiff)][1:4,rowid_test]

# Multiple versions

plot_ctree <- plot_indep <- list()
for(i in seq_along(these_rowid)){
  plot_ctree[[i]] <- sv_force_shapviz_mod(ctree_dt[rowid_test==these_rowid[i],..features_cols],
                                          b=0.5,
                                          feature_vals_dt,
                                          row_id = 1,
                                          max_display=5,
                                          fill_colors = c("darkgreen","darkred"))+
    ggplot2::ggtitle(paste0("Ctree, test id = ",these_rowid[i]))+
    theme(plot.title = element_text(size=10))

  plot_indep[[i]] <- sv_force_shapviz_mod(indep_dt[rowid_test==these_rowid[i],..features_cols],
                                          b=0.5,
                                          feature_vals_dt,
                                          row_id = 1,
                                          max_display=5,
                                          fill_colors = c("darkgreen","darkred"))+
    ggplot2::ggtitle(paste0("Independence, test id = ",these_rowid[i]))+
    theme(plot.title = element_text(size=10))

}

library(patchwork)
pl_ctree <- (patchwork::wrap_plots(plot_ctree) +
  patchwork::plot_layout(ncol = 1L) + patchwork::plot_annotation(title="ctree"))
pl_indep <- (patchwork::wrap_plots(plot_indep) +
      patchwork::plot_layout(ncol = 1L) + patchwork::plot_annotation(title = "independence"))

p1 <- (pl_ctree | pl_indep) + patchwork::plot_annotation(
  title = "Comparison of ctree and independence approach",
  theme = theme(plot.title = element_text(hjust = 0.5)))

ggsave(paste0("figures/Q3/Q3_adult_complete_condshap_ctree_indep_comp_separate_test.pdf"), p1, scale = 1.1,width = 10, height = 8)


# Testing to put the forceplots for independence and ctree together

p2 <- sv_force_shapviz_mod2(ctree_dt[rowid_test==these_rowid[1],..features_cols],
                            indep_dt[rowid_test==these_rowid[1],..features_cols],
                            b=0.5,
                            feature_vals_dt,
                            row_id = 1,
                            max_display=5,
                            fill_colors = c("darkgreen","darkred"))+
  scale_y_discrete(breaks=c(1,2),
                   labels=c("ctree","independence"))

ggsave(paste0("figures/Q3/Q3_adult_complete_condshap_ctree_indep_comp_together_test.pdf"), p2 , width = 8, height = 4)






################################################################################
#                         RESEARCH QUESTION 4 (Q4)
#
#             TODO: Add description of the research question
################################################################################


res_ce_values <- fread("./results/Q4/ce_values_final.csv")
res_ce_measures <- fread("./results/Q4/ce_measures_final.csv")

if (!dir.exists("figures/Q4")) dir.create("figures/Q4",recursive = TRUE)
if (!dir.exists("tables/Q4")) dir.create("tables/Q4",recursive = TRUE)

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






