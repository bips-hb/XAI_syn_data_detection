import os
import pandas as pd
from ucimlrepo import fetch_ucirepo 
import kagglehub

# create dataset folder function
def create_folder(dataset):
    if not os.path.exists(f'data/{dataset}'):
        os.makedirs(f'data/{dataset}/real')
        os.makedirs(f'data/{dataset}/syn')
        os.makedirs(f'data/{dataset}/histograms')
        os.makedirs(f'data/{dataset}/correlations')


### UCI

# save UCI data function
def save_uci_data(id, dataset_name):
   
    data_repo = fetch_ucirepo(id=id)
    target = data_repo.variables[data_repo.variables.role == "Target"]["name"].values[0]
    data = data_repo.data.original
    
    # special handling for adult
    if(id == 2):
        data['income'] = data['income'].str.replace(".", "")
        data = data.replace("?", "unknown")
        
        # for native-country, only keep top 30 categories, rest are set to 'other'
        top_29_countries = data['native-country'].value_counts().head(29).index
        data['native-country'] = data['native-country'].apply(lambda x: x if x in top_29_countries else 'other')
        
        # eclude education column
        data = data.drop(columns='education')
    
    #special handling for nursery
    elif(id == 76):
        # remove rows with 'recommend' in 'class' column
        data = data[data['class'] != 'recommend']
        
    
    data_cc = data.dropna(axis = 0, how = 'any').reset_index(drop=True)
    
    create_folder(dataset_name)
    data_cc.to_csv(f'data/{dataset_name}/real/{dataset_name}.csv', index=False)
    with open(f'data/{dataset_name}/real/target.txt', 'w') as f:
        f.write(target)

# dictionary with UCI datasets and their ids
uci_datasets = {
    'adult_complete': 2,
    'car_evaluation': 19,
    'statlog_landsat_satellite': 146,
    'nursery': 76,
    'magic_gamma_telescope': 159,
    'letter_recognition': 59,
    'chess_king_rook_vs_king': 23,
    'connect_4': 26
}

for dataset_name, id in uci_datasets.items():
    save_uci_data(id, dataset_name)


### kaggle

# save kaggle data function
def save_kaggle_data(url, download_filename, dataset_name, target):
        
        path = kagglehub.dataset_download(url)
        data = pd.read_csv(path + f'/{download_filename}.csv')
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')] 
        data_cc = data.dropna(axis = 0, how = 'any').reset_index(drop=True)
        
        create_folder(dataset_name)
        data_cc.to_csv(f'data/{dataset_name}/real/{dataset_name}.csv', index=False)
        with open(f'data/{dataset_name}/real/target.txt', 'w') as f:
            f.write(target)

kagge_datasets = {
    'diabetes': {"url": "mathchi/diabetes-data-set", "download_filename": "diabetes", "target": "Outcome"},
    'diamonds': {"url": "shivam2503/diamonds", "download_filename": "diamonds", "target": "price"},
    'diabetes_HI': {"url": "alexteboul/diabetes-health-indicators-dataset", "download_filename": "diabetes_012_health_indicators_BRFSS2015", "target": "Diabetes_012"}
}

for dataset_name, info in kagge_datasets.items():
    save_kaggle_data(info["url"], info["download_filename"], dataset_name, info["target"])