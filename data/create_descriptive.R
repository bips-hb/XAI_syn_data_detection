suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(ggpubr)
  library(foreach)
  library(doParallel)
})

# get datasetes

data_dir <- dir("data")
datasets <- data_dir[!data_dir %in% c("create_descriptive.R", "get_data.py", "datasets_overview.csv")]


# Overview table over all real datasets
create_overview_table <- function(dataset) {
  
  real <- fread(paste0("data/", dataset, "/real/", dataset, ".csv"))
  num_cols_count <- length(names(real)[sapply(real, is.numeric)])
  cat_cols <- names(real)[sapply(real, \(x) is.factor(x) | is.character(x))]
  cat_cols_count <- length(cat_cols)
  max_count_classes <- ifelse(cat_cols_count > 0,
                            max(sapply(real[, .SD, .SDcols = cat_cols], uniqueN)),
                            NA_integer_)
  
  overview <- data.table(
    dataset = dataset,
    rows = nrow(real),
    cols = ncol(real),
    num_cols = num_cols_count,
    cat_cols = cat_cols_count,
    max_classes = max_count_classes
  )
  
  return(overview)
}

overview_table <- foreach(dataset = datasets, .combine = rbind) %do% create_overview_table(dataset)
fwrite(overview_table, "data/datasets_overview.csv")


# just works for 1 synthetic dataset (the first) per synthesizer/real-dataset-combination
# if needed, median results and quantile ranges should be calculated over synthetic datasets and plotted

create_histograms <- function(dataset, synthesizer, cont_plot_type = 2) {
  
  syn_files <- dir(paste0("data/", dataset, "/syn/", synthesizer), full.names = T)
  
  real <- fread(paste0("data/", dataset, "/real/", dataset, ".csv"))
  num_cols <- names(real)[sapply(real, is.numeric)]
  cat_cols <- setdiff(names(real), num_cols)
  lvls <- lapply(real[, .SD, .SDcols = cat_cols], \(x) sort(unique(x)))
  
  real[, dataset := "Real"]
  
  syn_stacked <- foreach(syn_file_no = seq_along(syn_files), .combine = rbind) %do% {
    syn <- fread(syn_files[syn_file_no])
    syn[ , dataset := paste0("Syn_",syn_file_no)]
    syn[]
  }
  
  if(!is.null(syn_stacked)) {

    real[, ID := .I]
    syn_stacked[, ID := .I]
    suppressWarnings({
      if (length(num_cols) > 0){
        real_num_long <- melt(real[,.SD, .SDcols = c("ID", "dataset", num_cols)], id.vars = c("ID", "dataset"))[, variable := factor(variable, levels = num_cols)]
        num_long <- rbind(real_num_long, melt(syn_stacked[,.SD, .SDcols =  c("ID", "dataset", num_cols)], id.vars = c("ID", "dataset"))[, variable := factor(variable, levels = num_cols)])
      }
      if (length(cat_cols) > 0) {
        cat_long <- melt(rbind(real, syn_stacked)[,.SD, .SDcols = c("ID", "dataset", cat_cols)], id.vars = c("ID", "dataset"))[, variable := factor(variable, levels = cat_cols)]
      }
    })
    
    if (length(num_cols) > 0) {
    
      plot_num_hist <- ggplot(real_num_long, aes(x=value, fill=dataset)) + geom_histogram(bins=30, position="dodge") + facet_wrap(~variable, scales="free")
      #barplot_real_hist <- as.data.table(ggplot_build(plot_num)$data[[1]])[, .(variable = factor(num_cols[PANEL], levels = num_cols), x = x, y = y)][, data := "Real"][]
      bins_num <-  as.data.table(ggplot_build(plot_num_hist)$data[[1]])[, .(binwidth = xmax[1]-xmin[1], min_binleft = xmin[1]), by = PANEL][, variable := factor(num_cols[PANEL], levels = num_cols)][, -"PANEL"]
      
      num_long_ranges <- num_long[, .(dataset, value, min = min(value), max = max(value)), by = variable]
      num_long_bins <- num_long_ranges[bins_num, on = "variable"][, .(dataset, value, min_binleft_new = min - (min - min_binleft) %% binwidth, binwidth, max), by = .(variable)]
      num_long_bins0 <- num_long_bins[, .(binwidth = binwidth[1], x = seq(min_binleft_new[1], max[1], binwidth[1]) + binwidth[1]/2, y = 0), by = .(dataset, variable)]
      
      num_long_bins[, bin := (value - min_binleft_new)%/%binwidth * binwidth + min_binleft_new]
      barplot_data_num <- unique(num_long_bins[, .(binwidth, x = bin + binwidth/2, y = .N), by = .(dataset, variable, bin)])[, -"bin"]
      barplot_data_num <- unique(rbind(barplot_data_num, num_long_bins0), by = c("variable", "dataset", "x"))
      setorder(barplot_data_num, variable, x)
      barplot_data_num <- rbind(barplot_data_num[dataset == "Real"], 
                            barplot_data_num[dataset != "Real", .(dataset = synthesizer, binwidth, y = median(y), lo = quantile(y, 0.025), hi = quantile(y, 0.975)), by = .(variable, x)],
                            fill = T
      )[, dataset := factor(dataset, levels = c("Real", synthesizer))][]
      
      if (cont_plot_type == 1) {
        plot_num <- ggplot(barplot_data_num, aes(x=x, y=y,width= binwidth, fill=dataset)) + geom_bar(stat="identity", position="dodge") + 
          geom_errorbar(aes(ymin=lo, ymax=hi), position = "dodge") +
          theme(axis.title=element_blank()) + facet_wrap(~variable, scales="free") + 
          scale_fill_manual(values=c("dodgerblue4", "orange2"))
        
      } else if (cont_plot_type == 2) {
        plot_num <- ggplot(barplot_data_num, aes(x = x, y = y, col = dataset, fill = dataset)) + 
          geom_line(linewidth = 0.5) +
          geom_ribbon(aes(ymin = lo, ymax = hi), linetype=0, alpha = .5) + 
          theme(axis.title=element_blank()) + facet_wrap(~variable, scales="free") + 
          scale_fill_manual(values=c("dodgerblue4", "orange2")) +
          scale_color_manual(values=c("dodgerblue4", "orange2"))
        
      } else if (cont_plot_type == 3) {
        plot_num <- ggplot(barplot_data_num, aes(x = x, y = y,width= binwidth, col = dataset, fill = dataset)) + 
          geom_bar(barplot_data_num[dataset == "Real"],  mapping = aes(), stat="identity") +
          geom_line(barplot_data_num[dataset != "Real"],  mapping = aes(), linewidth = 0.5) +
          geom_ribbon(barplot_data_num[dataset != "Real"], mapping = aes(ymin = lo, ymax = hi), linetype=0, alpha = .6) + 
          theme(axis.title=element_blank()) + facet_wrap(~variable, scales="free") + 
          scale_fill_manual(values=c("dodgerblue4", "orange2")) +
          scale_color_manual(values=c("white", "orange2"))
      }
    
    }
    
    if (length(cat_cols) > 0) {
      
      cat_long_counts0 <- cat_long[, c(lvls[variable], y = 0) , by = .(dataset, variable)]
      setnames(cat_long_counts0, 3, "x")
      cat_long_counts <- cat_long[, .(y = .N), by = .(dataset, variable, value)]
      setnames(cat_long_counts, 3, "x")
      
      barplot_data_cat <- unique(rbind(cat_long_counts, cat_long_counts0), by = c("variable", "dataset", "x"))
      
      barplot_data_cat <- rbind(barplot_data_cat[dataset == "Real"], 
                                barplot_data_cat[dataset != "Real", .(dataset = synthesizer, y = median(y), lo = quantile(y, 0.025), hi = quantile(y, 0.975)), by = .(variable, x)],
                                fill = T
      )[, dataset := factor(dataset, levels = c("Real", synthesizer))][]
      
      plot_cat <- ggplot(barplot_data_cat, aes(x=x, y=y, fill=dataset)) + geom_bar(stat="identity", position="dodge") + 
        geom_errorbar(aes(ymin=lo, ymax=hi), position = position_dodge(0.9), width = 0.6) +
        theme(axis.title=element_blank(), axis.text.x = element_text(angle = 30, hjust = 1)) + facet_wrap(~variable, scales="free") + 
        scale_fill_manual(values=c("dodgerblue4", "orange2"))
      
    }  
    
    plot_rows_num <- 1 + (length(num_cols) - 1) %/% 3
    plot_rows_cat <- 1 + (length(cat_cols) - 1) %/% 3
    
    # combine both plots in one page with ggarange
    if (plot_rows_cat == 0) {
      plot_cat <- NULL
      heights <- c(1, 0)
    } else if (plot_rows_num == 0) {
      plot_num <- NULL
      heights <- c(0, 1)
    } else {
      heights <- c(1, plot_rows_cat/plot_rows_num)
    }
    
    plot <- ggarrange(plot_num, plot_cat, ncol=1, heights = heights, legend = "right", common.legend = TRUE)
    annotate_figure(plot, top = text_grob(paste0("Histograms", ": ", dataset, " - real/", synthesizer), face = "bold", size = 16))
  
    ggsave(paste0("data/", dataset, "/histograms/hist_", dataset, "_", synthesizer, ".pdf"), width = 32, height = 16, dpi = 30)
  }
}

create_cordiff <- function(dataset, synthesizers, agg = "median") {
  
  real <- fread(paste0("data/", dataset, "/real/", dataset, ".csv"))
  num_cols <- names(real)[sapply(real, is.numeric)]
  
  if(length(num_cols) == 0) {
    return()
  } else {
    cor_real <- cor(real[, .SD, .SDcols = num_cols])
    pdf(paste0("data/", dataset, "/correlations/corr_real_", dataset, ".pdf"), width = 15, height = 10, )
    cor_real_plot <- corrplot::corrplot(cor_real, title = paste0("Correlations: ", dataset), method = "color", type = "upper", tl.col = "black", tl.srt = 45, addCoef.col = "black", number.cex = 0.8, tl.cex = 1, cl.cex = 1, cl.length = 0.5, cl.ratio = 0.3, mar = c(0,0,2,0))
    dev.off()
    pdf(paste0("data/", dataset, "/correlations/corrdiff_", agg, "_", dataset, ".pdf"), width = 15, height = 10)
    par(mfrow=c(1 + (length(synthesizers) - 1) %/% 3,3), oma = c(0, 0, 2, 0))
    for (synthesizer in synthesizers) {
      syn_files <- dir(paste0("data/", dataset, "/syn/", synthesizer), full.names = T)
      
      if (length(syn_files) > 0) {
      
        syn_stacked <- foreach(syn_file_no = seq_along(syn_files), .combine = rbind) %do% {
          syn <- fread(syn_files[syn_file_no])
          syn[ , dataset := paste0("Syn_",syn_file_no)]
          syn[]
        }
        cor_syn <- syn_stacked[, .(cor = c(cor(.SD))), by = dataset , .SDcols = num_cols][, entry := 1:.N, by = dataset][]
        cor_syn <- cor_syn[, .(median = median(cor), lo = quantile(cor, 0.025), hi = quantile(cor, 0.975)), by = entry]
        cor_syn_median <- matrix(cor_syn[, median], nrow = length(num_cols), dimnames = list(num_cols, num_cols))
        cor_syn_lo <- matrix(cor_syn[, lo], nrow = length(num_cols), dimnames = list(num_cols, num_cols))
        cor_syn_hi <- matrix(cor_syn[, hi], nrow = length(num_cols), dimnames = list(num_cols, num_cols))
        
        if(agg == "median") {
          col_ <- NULL
          plotCI_ <- "n"
        } else if (agg == "CI") {
          col_ <- c("indianred2", "skyblue2")
          plotCI_ <- "square"
        }
        cordiff_plot <- corrplot::corrplot(cor_syn_median - cor_real, low = cor_syn_lo- cor_real, upp = cor_syn_hi- cor_real, title = paste0("real - ", synthesizer), method = "color", plotCI = plotCI_,
                                            col = col_, type = "upper", tl.col = "black", tl.srt = 45, addCoef.col = "black", number.cex = 0.5, tl.cex = 0.9, cl.cex = 0.9, cl.length = 3, cl.ratio = 0.15, mar = c(0,0,2,0))
      
        cordiff_plot
      }
    }
    mtext(paste0("Correlation differences (", agg , "): ", dataset), side = 3, line = 0.5, outer = T)
    dev.off()
    par(mfrow = c(1, 1))
  }
}

for (dataset in datasets) {
  synthesizers <- dir(paste0("data/", dataset, "/syn"))
  for (synthesizer in synthesizers) {
    create_histograms(dataset, synthesizer)
  }
  create_cordiff(dataset, synthesizers, "median")
  create_cordiff(dataset, synthesizers, "CI")
  
}

