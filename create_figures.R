################################################################################
#                 Create all Figures for the manuscript
################################################################################
library(ggplot2)
library(patchwork)
library(geomtextpath)
library(rlang)
library(data.table)

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
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  