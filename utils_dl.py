import json
import os

from tabsyn.process_dataset import run as tabsyn_process
from tabsyn.tabsyn.vae.main import run as tabsyn_vae
from tabsyn.tabsyn.main import run as tabsyn_diff
from tabsyn.tabsyn.sample import run as tabsyn_sample

from CTABGANPlus.model.ctabgan import CTABGAN

from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer
from sdv.single_table import CTGANSynthesizer

# TabSyn

def create_info_json(data, data_name, task_type, target_col_idx, train_share = 1):
    
    if (len(target_col_idx) >0 and task_type == ""):
        raise ValueError("task_type must be provided when target_col_idx is provided.")
    
    num_col_idx = data.columns.get_indexer(data.select_dtypes(exclude = object).columns).tolist()
    cat_col_idx = data.columns.get_indexer(data.select_dtypes(include = object).columns).tolist()
    
    num_col_idx = list(set(num_col_idx) - set(target_col_idx))
    cat_col_idx = list(set(cat_col_idx) - set(target_col_idx))
    num_col_idx.sort()
    cat_col_idx.sort()
    
    data_dict = {'name': data_name,
                'task_type': task_type,
                "header": 'infer',
                'column_names': None,
                "num_col_idx": num_col_idx,
                "cat_col_idx": cat_col_idx,
                "target_col_idx": target_col_idx,
                "file_type": 'csv',
                "data_path": 'tabsyn_tmp/data/' + data_name + "/" + data_name + ".csv",
                "test_path": None,
                "train_share": train_share} 
    
    if not os.path.exists("tabsyn_tmp/data/Info"):
        os.mkdir("tabsyn_tmp/data/Info")
    
    with open("tabsyn_tmp/data/Info/" + data_name + ".json", "w") as outfile: 
        json.dump(data_dict, outfile, indent = 4)

def synthesize_tabsyn(data, data_name = "", save_path = None, task_type = "", target_col_idx = [], device = "cuda:0", train_share = 1): 
    if not os.path.exists("tabsyn_tmp"):
        os.mkdir("tabsyn_tmp")
    if not os.path.exists("tabsyn_tmp/data"):
        os.mkdir("tabsyn_tmp/data")
    if not os.path.exists("tabsyn_tmp/data/" + data_name):
        os.mkdir("tabsyn_tmp/data/" + data_name)
    if not os.path.exists("tabsyn_tmp/data" + data_name + "/" + data_name + ".csv"):
        data.to_csv("tabsyn_tmp/data/" + data_name + "/" + data_name + ".csv", index = False)
        
    gpu = int(device.split(":")[1])
    
    create_info_json(data, data_name, task_type = task_type, target_col_idx = target_col_idx, train_share = train_share)
    tabsyn_process(dataname = data_name)
    tabsyn_vae(dataname = data_name, gpu = gpu)
    tabsyn_diff(dataname = data_name, gpu = gpu) 
    syn = tabsyn_sample(dataname = data_name, save_path= save_path, gpu = gpu)
    return(syn)


# CTAB-GAN+

def synthesize_ctabgan_plus(data, device = "cuda:0"):
    ctabgan = CTABGAN(data = data,
                    integer_columns= data.select_dtypes(include = int).columns.tolist(),
                    categorical_columns= data.select_dtypes(include = object).columns.tolist(),
                    problem_type= {None: None}, device = device)
    ctabgan.fit()
    return(ctabgan.generate_samples())

# TVAE

def synthesize_tvae(data, device = "cuda:0"): 
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    synthesizer = TVAESynthesizer(metadata,cuda = device)
    synthesizer.fit(data)
    return synthesizer.sample(len(data))

# CTGAN

def synthesize_ctgan(data, device = "cuda:0"): 
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    synthesizer = CTGANSynthesizer(metadata,cuda = device)
    synthesizer.fit(data)
    return synthesizer.sample(len(data))