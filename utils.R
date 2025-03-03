################################################################################
#                         Utility functions
#
# This file contains heler functions for:
#       - Fitting Detection Models
#       - Data Preprocessing
#       - Get prediction functions
#       - Shapley/Shapvis helpers
#       - Plotting functions
#       - For CF tables
################################################################################


#-------------------------------------------------------------------------------
#                       Fitting Detection Models
#-------------------------------------------------------------------------------

# Fit detection model
# `data` is a list of data frames, each containing a real and synthetic dataset
# `model_name` gives the model type to fit
fit_model <- function(data, model_name = "ranger", max_runs = NULL,
                      n_threads = 15, time_limit = 160 * 30, n_parallel = 5,
                      init_points = n_parallel) {

  # Get names of the syn. runs
  run_names <- names(data)
  args <- strsplit(run_names, "--")

  res <- lapply(seq_along(data), function(i) {
    set.seed("2024")


    # Skip if not in these_runs
    if (!is.null(max_runs)) {
      if (as.numeric(args[[i]][3]) > max_runs) {
        return(NULL)
      }
    }

    # Shuffle data
    df <- data[[i]][sample(nrow(data[[i]])), ]
    df_train <- df[df$train == 1, -which(names(df) == "train")]
    df_test <- df[df$train == 0, -which(names(df) == "train")]
    df <- df[, -which(names(df) == "train")]

    # Ranger model -------------------------------------------------------------
    if (model_name == "ranger") {
      library(ranger)
      model <- ranger(real ~ ., data = df_train,
                      importance = 'impurity',
                      probability = TRUE,
                      classification = TRUE,
                      oob.error = TRUE)

      # Get predictions (as classes)
      y_pred_test <- apply(predict(model, data = df_test)$predictions, 1, which.max) - 1
      y_pred_train <- apply(predict(model, data = df_train)$predictions, 1, which.max) - 1

      # Model-specific metrics
      model_metric_name <- c("OOB error")
      model_metric_value <- c(model$prediction.error)

      # Save model
      dir <- paste0("./models/", model_name, "/")
      if (!dir.exists(dir)) dir.create(dir)
      saveRDS(model, paste0(dir, paste(args[[i]], collapse = "--"), ".rds"))

      # Logistic regression ------------------------------------------------------
    } else if (model_name == "logReg") {
      model <- glm(real ~ ., data = df_train, family = binomial)
      model <- stripGlmLR(model)

      # Check for missing factors
      if (any(sapply(df[, -which(names(df) == "real")], class) == "factor")) {
        factor_idx <- which(sapply(df[, -which(names(df) == "real")], class) == "factor")
        for (name in names(factor_idx)) {
          lev <- union(model$xlevels[[name]], levels(df[[name]]))
          lev <- lev[order(match(lev,levels(df[[name]])))]
          model$xlevels[[name]] <- lev
        }
      }

      # Get predictions (as classes)
      y_pred_test <- as.numeric(round(predict(model, newdata = df_test, type = "response")))
      y_pred_train <- as.numeric(round(predict(model, newdata = df_train, type = "response")))

      # Model-specific metrics
      model_metric_name <- c("AIC")
      model_metric_value <- c(model$aic)

      # Save model
      dir <- paste0("./models/", model_name, "/")
      if (!dir.exists(dir)) dir.create(dir)
      saveRDS(model, paste0(dir, paste(args[[i]], collapse = "--"), ".rds"))
      # XGBoost ------------------------------------------------------------------
    } else if (model_name == "xgboost") {
      library(xgboost)

      # Set hyperparameter bounds (only for xgboost)
      bounds <- list(
        eta = c(0.005, 0.5),
        max_depth = c(1L, 11L),
        min_child_weight = c(1, 60),
        subsample = c(0.1, 1),
        lambda = c(1, 10),
        alpha = c(1, 10)
      )

      # Encode datasets
      df_train_enc <- encode_cat_vars(df_train)
      df_test_enc <- encode_cat_vars(df_test)

      x <- as.matrix(df_train_enc[, -which(names(df_train_enc) == "real")])
      y <- as.numeric(df_train_enc$real) - 1

      best_params = tune_xgboost(x, y, bounds, n_threads = n_threads,
                                 initPoints = init_points,
                                 n_parallel = n_parallel, time_limit = time_limit)

      # Save the result of the tuning
      ggsave(paste0("./tmp/tuning_logs/", paste(c(args[[i]], "xgboost"), collapse = "--"), ".pdf"),
             width = 15, height = 10)

      model <- xgboost(params = best_params$params,
                       nrounds = best_params$nrounds,
                       data = x,
                       nthread = n_threads,
                       label = y,
                       objective = "binary:logistic",
                       verbose = 0)

      # Get predictions (as classes)
      y_pred_test <- as.numeric(round(predict(model,
                                              newdata = as.matrix(df_test_enc[, -which(names(df_test_enc) == "real")]))))
      y_pred_train <- as.numeric(round(predict(model,
                                               newdata = as.matrix(df_train_enc[, -which(names(df_train_enc) == "real")]))))

      # Model-specific metrics
      model_metric_name <- c()
      model_metric_value <- c()

      # Save model
      dir <- paste0("./models/", model_name, "/")
      if (!dir.exists(dir)) dir.create(dir)
      xgb.save(model, paste0(dir, paste(args[[i]], collapse = "--"), ".rds"))
    } else {
      stop("Model not implemented")
    }

    # Calculate metrics
    y_true_test <- as.numeric(df_test$real) - 1
    y_true_train <- as.numeric(df_train$real) - 1
    res_i <- data.frame(
      dataset = args[[i]][1],
      syn_name = args[[i]][2],
      run = as.numeric(args[[i]][3]),
      model_name = model_name,
      train = c(rep("train data", 3), rep("test data", 3),
                rep("model specific", length(model_metric_value))),
      metric = c("Accuracy", "Precision", "Recall",
                 "Accuracy", "Precision", "Recall",
                 model_metric_name),
      value = c(
        Metrics::accuracy(y_true_train, y_pred_train),
        Metrics::precision(y_true_train, y_pred_train),
        Metrics::recall(y_true_train, y_pred_train),
        Metrics::accuracy(y_true_test, y_pred_test),
        Metrics::precision(y_true_test, y_pred_test),
        Metrics::recall(y_true_test, y_pred_test),
        model_metric_value
      )
    )

    # Save res in logs
    saveRDS(res_i, paste0("./tmp/tuning_logs/", paste(c(args[[i]], as.character(model_name)), collapse = "--"), ".rds"))

    res_i
  })

  do.call(rbind, res)
}


# Hyperparameter tuning for XGBoost
tune_xgboost <- function(x, y, bounds,
                         n_threads = 20,
                         n_parallel = 5,
                         initPoints = 18,
                         n_folds = 5, time_limit = 60 * 30) {

  # Create folds
  folds <- lapply(seq_len(n_folds), function(i) {
    as.integer(seq(i, nrow(x), by = n_folds))
  })
  names(folds) <- paste0("fold", seq_len(n_folds))

  # Function must take the hyper-parameters as inputs
  obj_func <- function(eta, max_depth, min_child_weight, subsample, lambda, alpha) {
    param <- list(
      eta = eta,
      max_depth = max_depth,
      min_child_weight = min_child_weight,
      subsample = subsample,
      lambda = lambda,
      alpha = alpha,
      nthread = n_threads %/% n_parallel,

      # Classification problem
      objective = "binary:logistic")

    xgbcv <- xgb.cv(params = param,
                    data = x,
                    label = y,
                    nround = 10000,
                    folds = folds,
                    prediction = TRUE,
                    early_stopping_rounds = 5,
                    metrics = "auc",
                    maximize = TRUE,
                    verbose = FALSE)

    lst <- list(
      Score = max(xgbcv$evaluation_log$test_auc_mean),
      nrounds = xgbcv$best_iteration
    )

    return(lst)
  }

  # Set max to 30 mins
  cl <- parallelly::makeClusterPSOCK(n_parallel)
  registerDoParallel(cl)
  clusterExport(cl,c('folds', 'y', "x", "n_threads", "n_parallel"), envir = environment())
  clusterEvalQ(cl,expr= {
    library(xgboost)
  })

  bayes_out <- bayesOpt(FUN = obj_func, bounds = bounds,
                        initPoints = initPoints, iters.n = 30000,
                        otherHalting = list(timeLimit = time_limit),
                        iters.k = initPoints,
                        plotProgress = TRUE,
                        parallel = TRUE,
                        verbose = FALSE)
  stopCluster(cl)
  registerDoSEQ()

  # Get best hyperparameters
  params <- getBestPars(bayes_out)

  # Run cross validation
  xgbcv <- xgb.cv(params = params,
                  data = x,
                  label = y,
                  nthread = n_threads,
                  nround = 10000,
                  folds = folds,
                  prediction = TRUE,
                  early_stopping_rounds = 10,
                  maximize = TRUE,
                  metrics = "auc",
                  verbose = FALSE)

  list(params = params, nrounds = xgbcv$best_iteration)
}

#-------------------------------------------------------------------------------
#                           Data Preprocessing
#-------------------------------------------------------------------------------

# Load and combine real and synthetic data
load_data <- function(dataset_name, syn_name, test_split = 0.3) {
  path_real <- paste0("./data/", dataset_name, "/real/", dataset_name, ".csv")
  path_syn <- paste0("./data/", dataset_name, "/syn/", syn_name, "/")

  # Load real data and combine with synthetic data
  data_real <- cbind(read.csv(path_real), real = 1)
  # Replace all "." with "-"
  names(data_real) <- gsub("\\.", "_", names(data_real))

  data <- lapply(list.files(path_syn), function(pth) {
    # Set seed (in order to get the same split for train and test data and for
    # the different runs)
    set.seed(42)

    data_syn <- cbind(read.csv(paste0(path_syn, pth)), real = 0)
    # Replace all "." with "-"
    names(data_syn) <- gsub("\\.", "_", names(data_syn))

    df <- rbind(data_syn, data_real)

    # Transform outcome to factor
    df$real <- factor(df$real, levels = c(0, 1), labels = c("Synthetic", "Real"))

    # Set train and test split
    num_syn <- nrow(df[df$real == "Synthetic", ])
    num_real <- nrow(df[df$real == "Real", ]) # should be the same as num_syn
    df$train <- c(rbinom(num_syn, 1, prob = 1 - test_split),
                  rbinom(num_real, 1, prob = 1 - test_split))

    # Set factors for categorical data
    cat_idx <- which(sapply(df, class) == "character")
    for (idx in cat_idx) {
      df[, idx] <- as.factor(df[, idx])
    }

    df
  })
  names(data) <- gsub(".*_(\\d+)\\.csv$", paste(dataset_name, syn_name, "\\1", sep = "--"), list.files(path_syn))

  data
}


# Reduce size of glm model
# (see https://win-vector.com/2014/05/30/trimming-the-fat-from-glm-models-in-r/)
stripGlmLR = function(cm) {
  cm$y = c()
  cm$model = c()

  cm$residuals = c()
  cm$fitted.values = c()
  cm$effects = c()
  cm$linear.predictors = c()
  cm$prior.weights = c()
  cm$data = c()

  cm
}


to_categorical <- function(x, label = "") {
  if (is.factor(x)) {
    feat_names <- levels(x)
    y <- as.integer(x)
  } else if (is.character(x)) {
    feat_names <- unique(x)
    y <- match(x, feat_names)
  } else {
    feat_names <- unique(as.character(x))
    y <- match(x, feat_names)
  }

  if (length(feat_names) == 2) {
    feat_names <- feat_names[1]
    cat <- as.matrix(y - 1)
  } else {
    cat <- matrix(0, nrow = length(y), ncol = length(feat_names))
    cat[cbind(seq_along(y), y)] <- 1
  }

  colnames(cat) <- paste0(label, ": ", feat_names)
  cat
}

# onehot-encoding of categorical variables
encode_cat_vars <- function(df, exclude = c("real", "train")) {
  df <- as.data.frame(df)
  cat_idx <- which(sapply(df, class) == "factor")
  cat_idx <- cat_idx[!names(cat_idx) %in% exclude]
  if (length(cat_idx) > 0) {
    df_cat <- do.call("cbind", lapply(names(cat_idx), function(name) {
      to_categorical(df[[name]], name)
    }))
    if(length(cat_idx)!=ncol(df)){
      cbind(df_cat, df[, -cat_idx,drop = FALSE])
    } else {
      df_cat
    }
  } else {
    df
  }
}

#-------------------------------------------------------------------------------
#                         Get prediction functions
#-------------------------------------------------------------------------------

get_predict_fun <- function(model_name) {
  # Select predict function
  if (model_name == "xgboost") {
    pred_fun <- function(model, newdata) {
      class(model) = "xgb.Booster" # Required for shapr workaround
      # Encode data
      newdata <- as.matrix(encode_cat_vars(newdata))
      as.numeric(predict(model, newdata = newdata))
    }
  } else if (model_name == "logReg") {
    pred_fun <- function(model, newdata) {
      as.numeric(predict(model, newdata = newdata, type = "response"))
    }
  } else if (model_name == "ranger") {
    pred_fun <- function(model, newdata) {
      as.numeric(predict(model, data = newdata)$predictions[, 2])
    }
  }

  pred_fun
}


get_predict_fun_shapr <- function(model_name){
  # Ranger model -------------------------------------------------------------
  if (model_name == "ranger") {

    # Only get the probability of the positive class
    pred_fun <- function(model, newdata) {
      ranger:::predict.ranger(model, data = newdata, type = "response")$predictions[, 2]
    }

    # Logistic Regression model ------------------------------------------------
  } else if (model_name == "xgboost"){
    pred_fun <- function(model, newdata) {
      class(model) = "xgb.Booster"
      to_categorical <- function(x, label = "") {
        if (is.factor(x)) {
          feat_names <- levels(x)
          y <- as.integer(x)
        } else if (is.character(x)) {
          feat_names <- unique(x)
          y <- match(x, feat_names)
        } else {
          feat_names <- unique(as.character(x))
          y <- match(x, feat_names)
        }

        if (length(feat_names) == 2) {
          feat_names <- feat_names[1]
          cat <- as.matrix(y - 1)
        } else {
          cat <- matrix(0, nrow = length(y), ncol = length(feat_names))
          cat[cbind(seq_along(y), y)] <- 1
        }

        colnames(cat) <- paste0(label, ": ", feat_names)
        cat
      }

      # onehot-encoding of categorical variables
      encode_cat_vars <- function(df, exclude = c("real", "train")) {
        df <- as.data.frame(df)
        cat_idx <- which(sapply(df, class) == "factor")
        cat_idx <- cat_idx[!names(cat_idx) %in% exclude]
        if (length(cat_idx) > 0) {
          df_cat <- do.call("cbind", lapply(names(cat_idx), function(name) {
            to_categorical(df[[name]], name)
          }))
          if(length(cat_idx)!=ncol(df)){
            cbind(df_cat, df[, -cat_idx,drop = FALSE])
          } else {
            df_cat
          }
        } else {
          df
        }
      }

      # Encode data
      newdata <- as.matrix(encode_cat_vars(newdata))
      xgboost:::predict.xgb.Booster(model, newdata = newdata)
    }
  } else {
    pred_fun = function(model,newdata){
      predict(model,newdata,type="response")
    }
  }

  return(pred_fun)

}

#-------------------------------------------------------------------------------
#                         Shapley/Shapvis helpers
#-------------------------------------------------------------------------------

# Calculate variable importance including interactions
sv_vi <- function(object, abs = TRUE, idx = NULL) {
  ints <- object$S_inter
  
  if (!is.null(idx)) {
    ints <- ints[idx,,]
  }
  
  dimnames(ints)[[1]] <- 1:(dim(ints)[1])
  
  # Remove lower part of matrix
  for (i in 1:dim(ints)[1]) {
    ints[i,,][lower.tri(ints[i,,])] <- NA
  }
  
  dt <- as.data.table(ints)
  dt[, id := as.numeric(V1)]
  dt[, var := paste(V2, V3, sep = " - ")]
  dt[V2 == V3, var := V2]
  dt[, degree := 2]
  dt[V2 == V3, degree := 1]
  
  # Interactions x2 because of removed lower parts
  dt[degree == 2, value := 2*value]
  
  if (abs) {
    aggr <- dt[, list(value = mean(abs(value))), by = .(var, degree)]
  } else {
    aggr <- dt[, list(value = mean((value))), by = .(var, degree)]
  }
  
  aggr
}


# Necessary for waterfall plot (see below)
.make_dat <- function(object, format_feat, sep = " = ") {
  ints <- object$S_inter
  dimnames(ints)[[1]] <- 1:(dim(ints)[1])
  
  # Remove lower part of matrix
  for (i in 1:dim(ints)[1]) {
    ints[i,,][lower.tri(ints[i,,])] <- NA
  }
  
  dt <- as.data.table(ints)
  dt[, id := as.numeric(V1)]
  dt[, var := paste(V2, V3, sep = " - ")]
  dt[V2 == V3, var := V2]
  dt[, degree := 2]
  dt[V2 == V3, degree := 1]
  
  labels <- data.table(feat = colnames(object$X),
                       label = paste(colnames(object$X), format_feat(object$X), sep = sep))
  dt <- merge(merge(dt, labels, by.x = "V2", by.y = "feat", all.x = TRUE),
              labels, by.x = "V3", by.y = "feat", all.x = TRUE)
  dt[label.x == label.y, label.y := ""]
  dt[label.y != "", label.y := paste0(", ", label.y)]
  dt[, label := paste0(label.x, label.y)]
  #dt[, S := format_feat(value)]
  dt[, S := value]
  
  dt[, .(S, label)]
}

get_top_df <- function(df_intershap, top = 20) {
  df_interact_shap <- rbindlist(lapply(df_intershap, sv_vi))
  df_interact_mean <- df_interact_shap[, .(value = median(value)), by = c("var", "degree")]
  vi_top <- df_interact_mean[order(-abs(value))][1:top]
  df <- df_interact_shap[var %in% vi_top$var]
  df[, degree := factor(degree)]
  df[, var := factor(var, levels = rev(vi_top$var))]
  df$type <- "global TreeSHAP (interactions)"

  df
}


#-------------------------------------------------------------------------------
#                             Plotting functions
#-------------------------------------------------------------------------------

sv_force_shapviz_mod <- function(shap_dt,b,feature_vals_dt, row_id = 1L, max_display = 6L,
                                 fill_colors = c("#f7d13d", "#a52c60"),
                                 format_shap = getOption("shapviz.format_shap"),
                                 format_feat = getOption("shapviz.format_feat"),
                                 contrast = TRUE, bar_label_size = 3.2,
                                 show_annotation = TRUE, annotation_size = 3.2, ...) {
  stopifnot(
    "Exactly two fill colors must be passed" = length(fill_colors) == 2L,
    "format_shap must be a function" = is.function(format_shap),
    "format_feat must be a function" = is.function(format_feat)
  )
  # START manual editing

  #  object <- object[row_id, ]
  #  b <- get_baseline(object)
  #  dat <- .make_dat(object, format_feat = format_feat, sep = "=")
  #  if (ncol(object) > max_display) {
  #    dat <- .collapse(dat, max_display = max_display)
  #  }

  feature_cols <- names(feature_vals_dt)
  b <- b
  dat <- data.frame(S=unlist(shap_dt[row_id,]),
                    label = paste0(feature_cols, " = ",format_feat(unlist(feature_vals_dt[row_id,])))
  )
  if (ncol(shap_dt) > max_display) {
    dat <- shapviz:::.collapse(dat, max_display = max_display)
  }

  # END manual editing

  # Reorder rows and calculate order dependent columns
  .sorter <- function(y, p) {
    y <- y[order(abs(y$S)), ]
    y$to <- cumsum(y$S)
    y$from <- shapviz:::.lag(y$to, default = 0)
    hook <- y[nrow(y), "to"]
    vars <- c("to", "from")
    y[, vars] <- y[, vars] + p - hook
    y
  }
  dat$id <- "1"
  pred <- b + sum(dat$S)
  dat <- do.call(rbind, lapply(split(dat, dat$S >= 0), .sorter, p = pred))

  # Make a force plot
  b_pred <- c(b, pred)
  height <- grid::unit(0.17, "npc")

  p <- ggplot2::ggplot(
    dat,
    ggplot2::aes(
      xmin = from, xmax = to, y = id, fill = factor(S < 0, levels = c(FALSE, TRUE))
    )
  ) +
    gggenes::geom_gene_arrow(
      show.legend = FALSE,
      arrowhead_width = grid::unit(2, "mm"),
      arrow_body_height = height,
      arrowhead_height = height
    ) +
    ggrepel::geom_text_repel(
      ggplot2::aes(x = (from + to) / 2, y = as.numeric(id) + 0.08, label = label),
      size = bar_label_size,
      nudge_y = 0.3,
      segment.size = 0.1,
      segment.alpha = 0.5,
      direction = "both"
    ) +
    ggfittext::geom_fit_text(
      ggplot2::aes(label = paste0(ifelse(S > 0, "+", ""), format_shap(S))),
      show.legend = FALSE,
      contrast = contrast,
      ...
    ) +
    ggplot2::coord_cartesian(ylim = c(0.8, 1.2), clip = "off") +
    ggplot2::scale_x_continuous(expand = ggplot2::expansion(mult = 0.13)) +
    # scale_y_discrete(expand = expansion(add = c(0.1 + 0.5 * show_annotation, 0.6))) +
    ggplot2::scale_fill_manual(values = fill_colors, drop = FALSE) +
    ggplot2::theme_bw() +
    ggplot2::theme(
      aspect.ratio = 1 / 4,
      panel.border = ggplot2::element_blank(),
      panel.grid.minor = ggplot2::element_blank(),
      panel.grid.major.y = ggplot2::element_blank(),
      axis.line.x = ggplot2::element_line(),
      axis.ticks.y = ggplot2::element_blank(),
      axis.text.y = ggplot2::element_blank()
    ) +
    ggplot2::labs(y = ggplot2::element_blank(), x = "Prediction")

  if (show_annotation) {
    p <- p +
      ggplot2::annotate(
        "segment",
        x = b_pred,
        xend = b_pred,
        y = c(0.5, 0.75),
        yend = c(0.92, 1),
        linewidth = 0.3,
        linetype = 2
      ) +
      ggplot2::annotate(
        "text",
        x = b_pred,
        y = c(0.4, 0.65),
        label = paste0(c("E[C(x)]=", "C(x)="), format_shap(b_pred)),
        size = annotation_size
      )
  }
  p
}

sv_force_shapviz_mod3 <- function(shap_dt,b,feature_vals_dt, row_id = 1L, max_display = 6L,
                                 fill_colors = c("#f7d13d", "#a52c60"),
                                 format_shap = getOption("shapviz.format_shap"),
                                 format_feat = getOption("shapviz.format_feat"),
                                 contrast = TRUE, bar_label_size = 3.2,
                                 show_annotation = TRUE, annotation_size = 3.2, ...) {
  stopifnot(
    "Exactly two fill colors must be passed" = length(fill_colors) == 2L,
    "format_shap must be a function" = is.function(format_shap),
    "format_feat must be a function" = is.function(format_feat)
  )
  # START manual editing

  #  object <- object[row_id, ]
  #  b <- get_baseline(object)
  #  dat <- .make_dat(object, format_feat = format_feat, sep = "=")
  #  if (ncol(object) > max_display) {
  #    dat <- .collapse(dat, max_display = max_display)
  #  }

  feature_cols <- names(feature_vals_dt)
  b <- b
  dat <- data.frame(S=unlist(shap_dt[row_id,]),
                    label = paste0(feature_cols, " = ",format_feat(unlist(feature_vals_dt[row_id,])))
  )
  if (ncol(shap_dt) > max_display) {
    dat <- shapviz:::.collapse(dat, max_display = max_display)
  }

  # END manual editing

  # Reorder rows and calculate order dependent columns
  .sorter <- function(y, p) {
    y <- y[order(abs(y$S)), ]
    y$to <- cumsum(y$S)
    y$from <- shapviz:::.lag(y$to, default = 0)
    hook <- y[nrow(y), "to"]
    vars <- c("to", "from")
    y[, vars] <- y[, vars] + p - hook
    y
  }
  dat$id <- "1"
  pred <- b + sum(dat$S)
  dat <- do.call(rbind, lapply(split(dat, dat$S >= 0), .sorter, p = pred))

  # Make a force plot
  b_pred <- c(b, pred)
  height <- grid::unit(0.2, "npc")

  p <- ggplot2::ggplot(
    dat,
    ggplot2::aes(
      xmin = from, xmax = to, y = id, fill = factor(S < 0, levels = c(FALSE, TRUE))
    )
  ) +
    gggenes::geom_gene_arrow(
      show.legend = FALSE,
      arrowhead_width = grid::unit(2, "mm"),
      arrow_body_height = height,
      arrowhead_height = height
    ) +
    ggrepel::geom_text_repel(
      ggplot2::aes(x = (from + to) / 2, y = as.numeric(id) + 0.08, label = label),
      size = bar_label_size,
      nudge_y = 0.3,
      segment.size = 0.1,
      segment.alpha = 0.5,
      direction = "both"
    ) +
    ggfittext::geom_fit_text(
      ggplot2::aes(label = paste0(ifelse(S > 0, "+", ""), format_shap(S))),
      show.legend = FALSE,
      contrast = contrast
    ) +
    ggplot2::coord_cartesian(ylim = c(0.8, 1.2), clip = "off") +
    ggplot2::scale_x_continuous(expand = ggplot2::expansion(mult = 0.13)) +
    # scale_y_discrete(expand = expansion(add = c(0.1 + 0.5 * show_annotation, 0.6))) +
    ggplot2::scale_fill_manual(values = fill_colors, drop = FALSE) +
    ggplot2::theme_bw() +
    ggplot2::theme(
      aspect.ratio = 1 / 8,
      panel.border = ggplot2::element_blank(),
      panel.grid.minor = ggplot2::element_blank(),
      panel.grid.major.y = ggplot2::element_blank(),
      axis.line.x = ggplot2::element_line(),
      axis.ticks.y = ggplot2::element_blank(),
      axis.text.y = ggplot2::element_blank()
    ) +
    ggplot2::labs(y = ggplot2::element_blank(), x = "Prediction")

  if (show_annotation) {
    p <- p +
      ggplot2::annotate(
        "segment",
        x = b_pred,
        xend = b_pred,
        y = c(0.5, 0.75),
        yend = c(0.92, 1),
        linewidth = 0.3,
        linetype = 2
      ) +
      ggplot2::annotate(
        "text",
        x = b_pred,
        y = c(0.4, 0.65),
        label = paste0(c("E[C(x)]=", "C(x)="), format_shap(b_pred)),
        size = annotation_size
      )
  }
  p
}


sv_force_shapviz_mod2 <- function(shap_dt1,shap_dt2,b,feature_vals_dt, row_id = 1, max_display = 6L,
                                 fill_colors = c("#f7d13d", "#a52c60"),
                                 format_shap = getOption("shapviz.format_shap"),
                                 format_feat = getOption("shapviz.format_feat"),
                                 contrast = TRUE, bar_label_size = 3.2,
                                 show_annotation = TRUE, annotation_size = 3.2, ...) {
  stopifnot(
    "Exactly two fill colors must be passed" = length(fill_colors) == 2L,
    "format_shap must be a function" = is.function(format_shap),
    "format_feat must be a function" = is.function(format_feat)
  )
  # START manual editing

  #  object <- object[row_id, ]
  #  b <- get_baseline(object)
  #  dat <- .make_dat(object, format_feat = format_feat, sep = "=")
  #  if (ncol(object) > max_display) {
  #    dat <- .collapse(dat, max_display = max_display)
  #  }

  feature_cols <- names(feature_vals_dt)
  b <- b
  dat1 <- data.frame(S=unlist(shap_dt1[row_id,]),
                     label = paste0(feature_cols, " = ",format_feat(unlist(feature_vals_dt[row_id,])))
  )
  dat2 <- data.frame(S=unlist(shap_dt2[row_id,]),
                     label = paste0(feature_cols, " = ",format_feat(unlist(feature_vals_dt[row_id,])))
  )

  if (length(feature_cols) > max_display) {
    dat1 <- shapviz:::.collapse(dat1, max_display = max_display)
    dat2 <- shapviz:::.collapse(dat2, max_display = max_display)
  }



  # END manual editing

  # Reorder rows and calculate order dependent columns
  .sorter <- function(y, p) {
    y <- y[order(abs(y$S)), ]
    y$to <- cumsum(y$S)
    y$from <- shapviz:::.lag(y$to, default = 0)
    hook <- y[nrow(y), "to"]
    vars <- c("to", "from")
    y[, vars] <- y[, vars] + p - hook
    y
  }
  dat1$id <- "1"
  pred <- b + sum(dat1$S)
  dat1 <- do.call(rbind, lapply(split(dat1, dat1$S >= 0), .sorter, p = pred))

  dat2$id <- "2"
  pred <- b + sum(dat2$S)
  dat2 <- do.call(rbind, lapply(split(dat2, dat2$S >= 0), .sorter, p = pred))


  # Make a force plot
  b_pred <- c(b, pred)
  height <- grid::unit(0.17, "npc")


  dat <- rbind(dat1,dat2)
  p <- ggplot2::ggplot(
    dat,
    ggplot2::aes(
      xmin = from, xmax = to, y = id, fill = factor(S < 0, levels = c(FALSE, TRUE))
    )
  ) +
    gggenes::geom_gene_arrow(
      show.legend = FALSE,
      arrowhead_width = grid::unit(2, "mm"),
      arrow_body_height = height,
      arrowhead_height = height
    ) +
    ggrepel::geom_text_repel(
      ggplot2::aes(x = (from + to) / 2, y = as.numeric(id) + 0.08, label = label),
      size = bar_label_size,
      nudge_y = 0.3,
      segment.size = 0.1,
      segment.alpha = 0.5,
      direction = "both"
    ) +
    ggfittext::geom_fit_text(
      ggplot2::aes(label = paste0(ifelse(S > 0, "+", ""), format_shap(S))),
      show.legend = FALSE,
      contrast = contrast,
      ...
    ) +
    ggplot2::coord_cartesian(ylim = c(0.8, 1.2), clip = "off") +
    ggplot2::scale_x_continuous(expand = ggplot2::expansion(mult = 0.13)) +
    # scale_y_discrete(expand = expansion(add = c(0.1 + 0.5 * show_annotation, 0.6))) +
    ggplot2::scale_fill_manual(values = fill_colors, drop = FALSE) +
    ggplot2::theme_bw() +
    ggplot2::theme(
      aspect.ratio = 1 / 4,
      panel.border = ggplot2::element_blank(),
      panel.grid.minor = ggplot2::element_blank(),
      panel.grid.major.y = ggplot2::element_blank(),
      axis.line.x = ggplot2::element_line(),
      axis.ticks.y = ggplot2::element_blank()
    ) +
    ggplot2::labs(y = ggplot2::element_blank(), x = "Prediction")

  if (show_annotation) {
    p <- p +
      ggplot2::annotate(
        "segment",
        x = b_pred,
        xend = b_pred,
        y = c(0.5, 0.75),
        yend = c(0.92, 1),
        linewidth = 0.3,
        linetype = 2
      ) +
      ggplot2::annotate(
        "text",
        x = b_pred,
        y = c(0.4, 0.65),
        label = paste0(c("E[C(x)]=", "C(x)="), format_shap(b_pred)),
        size = annotation_size
      )
  }
  p
}

# Waterfall plot that shows interactions
plot_waterfall <- function(object, row_id = 1L, max_display = 10L,
                           order_fun = function(s) order(abs(s)),
                           fill_colors = c("#f7d13d", "#a52c60"),
                           format_shap = getOption("shapviz.format_shap"),
                           format_feat = getOption("shapviz.format_feat"),
                           contrast = TRUE, show_connection = TRUE,
                           show_annotation = TRUE, annotation_size = 3.2,
                           marg_int_colors = c("black","red"),
                           ...) {
  object <- object[row_id, ]
  b <- get_baseline(object)
  dat <- .make_dat(object, format_feat = format_feat, sep = " = ")
  if (nrow(dat) > max_display) {
    dat <- shapviz:::.collapse(dat, max_display = max_display)
  }
  m <- nrow(dat)

  # Add order dependent columns
  dat <- dat[order_fun(dat$S), ]
  dat$i <- seq_len(m)
  dat$to <- cumsum(dat$S) + b
  dat$from <- shapviz:::.lag(dat$to, default = b)

  marg_or_int <- ifelse(grepl(",",dat$label), "int", "marg")

  # Make a waterfall plot
  height <- grid::unit(1 / (1 + 2 * m), "npc")

  p <- ggplot2::ggplot(
    dat,
    ggplot2::aes(
      xmin = from,
      xmax = to,
      y = stats::reorder(label, i),
      fill = factor(to < from, levels = c(FALSE, TRUE))
    )
  ) +
    gggenes::geom_gene_arrow(
      show.legend = FALSE,
      arrowhead_width = grid::unit(2, "mm"),
      arrowhead_height = height,
      arrow_body_height = height
    ) +
    ggfittext::geom_fit_text(
      ggplot2::aes(label = paste0(ifelse(S > 0, "+", ""), format_shap(S))),
      show.legend = FALSE,
      contrast = contrast,
      ...
    ) +
    ggplot2::scale_fill_manual(values = fill_colors, drop = FALSE) +
    ggplot2::theme_bw() +
    ggplot2::theme(
      panel.border = ggplot2::element_blank(),
      panel.grid.minor = ggplot2::element_blank(),
      panel.grid.major.x = ggplot2::element_blank(),
      axis.line.x = ggplot2::element_line(),
      axis.ticks.y = ggplot2::element_blank()
    ) +
    ggplot2::labs(y = ggplot2::element_blank(), x = "Logodds prediction")

  if (show_connection) {
    p <- p +
      ggplot2::geom_segment(
        ggplot2::aes(x = to, xend = to, y = i, yend = shapviz:::.lag(i, lead = TRUE, default = m)),
        linewidth = 0.3,
        linetype = 2
      )
  }
  if (show_annotation) {
    full_range <- c(as.numeric(dat[m, "to"]), as.numeric(dat[1L, "from"]))
    p <- p +
      ggplot2::annotate(
        "segment",
        x = full_range,
        xend = full_range,
        y = c(m, 1),
        yend = c(m, 1) + m * c(0.075, -0.075) + 0.13 * c(1, -1),
        linewidth = 0.3,
        linetype = 2
      ) +
      ggplot2::annotate(
        "text",
        x = full_range,
        y = c(m, 1) + m * c(0.1, -0.1) + 0.15 * c(1, -1),
        label = paste0(c("logistic(C(x))=", "logistic(E[C(x)])="), format_shap(full_range), c("    ","")),
        size = annotation_size
      ) +
      ggplot2::scale_x_continuous(expand = ggplot2::expansion(mult = c(0.05, 0.12))) +
      ggplot2::scale_y_discrete(expand = ggplot2::expansion(add = 0.3, mult = 0.2)) +
      ggplot2::coord_cartesian(clip = "off")
  }
  p+theme(axis.text.y = element_text(color = ifelse(marg_or_int=="int", marg_int_colors[2], marg_int_colors[1])))
}


#-------------------------------------------------------------------------------
#                           Creating CF Tables
#-------------------------------------------------------------------------------


make_cf_table <- function(cf_table, dataset_name) {
  
  data <- fread(paste0("data/", dataset_name ,"/real/", dataset_name, ".csv"))
  
  cols <- names(data)
  # replace "-" with "_"
  cols <- gsub("-", "_", cols)
  
  num_cols <- cols[(data[, sapply(data, is.numeric)])]
  cat_cols <- setdiff(cols, num_cols)
  
  cf_tab <- cbind(cols, tab_final)
  
  # Create the flextable
  ft <- flextable(cf_tab)
  
  # Remove the header name for the first column
  ft <- set_header_labels(ft, cols = "")
  
  # Apply image and background color for matching categorical and numeric cells
  for (col in names(cf_tab)[3:6]) {  # Exclude "Original" column
    matching_rows_cat <- intersect(which(cols %in% cat_cols), which(cf_tab[[col]] != cf_tab$Original))
    matching_rows_num_higher <- intersect(which(cols %in% num_cols), which(cf_tab[[col]] > cf_tab$Original))
    matching_rows_num_lower <- intersect(which(cols %in% num_cols), which(cf_tab[[col]] < cf_tab$Original))
    
    for (row in seq_along(cols)) {
      
      if (row %in% matching_rows_cat) {
        bgcol = "lightblue"
        img = "tables/Q4/arrow_lr.png"
      } else if (row %in% matching_rows_num_higher) {
        bgcol = "lightgreen"
        img = "tables/Q4/arrow_up.png"
      } else if (row %in% matching_rows_num_lower) {
        bgcol = "indianred1"
        img = "tables/Q4/arrow_down.png"
      } else next
      
      ft <- compose(
        ft,
        j = col,
        i = row,
        value = as_paragraph(
          as_chunk(cf_tab[[col]][row]),  # Keep the original text
          " ",
          as_image(src = img, width = .23, height = .15)  # Add image
        )
      )
      
      # Apply background color
      ft <- bg(ft, j = col, i = row, bg = bgcol)
    }
  }
  
  # Apply light gray background to "Original" column
  ft <- bg(ft, j = "Original", bg = "lightgray")
  
  # Make first and last row bold
  ft <- bold(ft, part = "header", bold = TRUE)  # header
  ft <- bold(ft, j = 1, bold = TRUE)  # first col
  # Auto-fit for better appearance
  ft <- autofit(ft)
  
  # Print the flextable
  ft
  
}
