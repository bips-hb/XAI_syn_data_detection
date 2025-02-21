import argparse
import os
import numpy as np
import pandas as pd
import random
import torch

from utils_dl import synthesize_tvae, synthesize_ctgan, synthesize_ctabgan_plus, synthesize_tabsyn

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# set seeds
def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# define synthesiers dict
synthesizers = {
    "TVAE": synthesize_tvae,
    "CTGAN": synthesize_ctgan,
    "CTAB-GAN+": synthesize_ctabgan_plus,
    "TabSyn": synthesize_tabsyn
}

def synthesize(dataset = "", synthesizer = "", reps = 1, device = "cuda:0"):
    data = pd.read_csv("data/" + dataset + "/real/" + dataset + ".csv")
    
    # check if synthesizer folder exists
    if not os.path.exists("data/" + dataset + "/syn/" + synthesizer):
        os.makedirs("data/" + dataset + "/syn/" + synthesizer)
    
    synthesizer_fn_ = synthesizers[synthesizer]
    for rep in range(reps):
        print("Synthesizing dataset: " + dataset + " with synthesizer: " + synthesizer + " (rep: " + str(rep+1) + "/" + str(reps) + ")")
        # if synthetic file does not exists yet, synthesize
        if os.path.exists("data/" + dataset + "/syn/" + synthesizer + "/syn_" + dataset + "_" + synthesizer + "_" + str(rep+1) + ".csv"):
            print("Synthetic data already exists at: " + "data/" + dataset + "/syn/" + synthesizer + "/syn_" + dataset + "_" + synthesizer + "_" + str(rep+1) + ".csv")
        else:
            setSeed(rep)
            if synthesizer == "TabSyn":
                synthesizer_fn = lambda data, device: synthesizer_fn_(data, data_name = dataset + "_" + str(rep+1), device = device)
            else:
                synthesizer_fn = synthesizer_fn_
            syn = synthesizer_fn(data, device = device)
            syn.to_csv("data/" + dataset + "/syn/" + synthesizer + "/syn_" + dataset + "_" + synthesizer + "_" + str(rep+1) + ".csv", index = False)
            print("Synthetic data saved at: " + "data/" + dataset + "/syn/" + synthesizer + "/syn_" + dataset + "_" + synthesizer + "_" + str(rep+1) + ".csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create synthetic data')
    parser.add_argument('--dataset', type=str, default=None, help='Name of dataset.')
    parser.add_argument('--synthesizer', type=str, default=None, help='Name of synthesizer.')
    parser.add_argument('--reps', type=int, default=1, help='Number of synthetic datasets to generate.')
    parser.add_argument('--device', type=str, default="cuda:0", help='CUDA device.')
    
    args = parser.parse_args()

    synthesize(**vars(args))
