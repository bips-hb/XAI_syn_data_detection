



# Additional computations for the paper






# adult_complete
dataset_name = "adult_complete"
model_name = "xgboost"
syn_name = "TabSyn"
run_model = 2



data <- load_data(dataset_name, syn_name)

data <- as.data.table(data[[run_model+1]])

data_test <- data[data$train == 0]

data_train <- data[data$train == 1]

# Very small correlation between age and education_num overall
(data_test[,.(.N,cor(age,education_num))])
(data_test[real=="Real",.(.N,cor(age,education_num))])
(data_test[real=="Synthetic",.(.N,cor(age,education_num))])

(data_train[,.(.N,cor(age,education_num))])
(data_train[real=="Real",.(.N,cor(age,education_num))])
(data_train[real=="Synthetic",.(.N,cor(age,education_num))])

# High correlation between age and education_num for the younger people
(data_test[real=="Real" & age<=20,.(.N,cor(age,education_num))])
(data_test[real=="Synthetic" & age<=20,.(.N,cor(age,education_num))])

(data_train[real=="Real" & age<=20,.(.N,cor(age,education_num))])
(data_train[real=="Synthetic" & age<=20,.(.N,cor(age,education_num))])

