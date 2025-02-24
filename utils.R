################################################################################
#                         Utility functions
################################################################################


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


# TODO: Do we need this?? # MJ: Yes for workaround with paralallization with shapr
get_predict_fun_shapr <- function(model_name){
  # Ranger model -------------------------------------------------------------
  if (model_name == "ranger") {

    # Only get the probability of the positive class
    pred_fun <- function(model, newdata) {
      ranger:::predict.ranger(model, data = newdata, type = "response")$predictions[, 2]
    }

    # Logistic Regression model ------------------------------------------------
  } else if (model_name == "xgboost"){
    class(model)="blabla"
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

