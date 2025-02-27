

# Additional computations for the paper


res_ce_values <- fread("./results/Q4/ce_values_final.csv")
res_ce_measures <- fread("./results/Q4/ce_measures_final.csv")

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

