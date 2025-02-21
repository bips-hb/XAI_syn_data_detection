
# IML for Synthetic Data Detection

## Generative Models
TODO

* synthpop
* ARF
* TVAE / CTGAN
* CTAB-GAN+
* TabSyn
* LLM approaches: GReaT, TabuLa

## Fitting Detection Models

The script `fit_detection_models.R` automatically retrieves all datasets from 
the "data/" folder, along with the corresponding generated synthetic datasets. 
Note that multiple synthetic datasets may be generated for a single real 
dataset and one generative model, which are labeled with the suffix "_[Run].csv".

For each generated synthetic dataset, a probability Random Forest 
(`ranger`) and a logistic regression model are fitted (TODO: add more detection 
models). The trained models are then saved in the "detection_models/ranger" 
and "detection_models/logReg" folders. Each model is named following this 
pattern: [Datasetname]--[GenerativeModel]--[Run]. Additionally, the predictive 
performances of all models are saved in the csv file 
"detection_models/results.csv", and a plot of the performances is 
generated and stored as "detection_models/plot_results.pdf".

## IML Methods

* `run_PFI.R` finds all model classes and the corresponding models from the
"detection_models/" folder and applies the PFI method. The results are combined 
and saved in "iml/results/PFI/feature_importance.csv", with a visualization
generated and stored in "iml/results/PFI/feature_importance.pdf".

* `run_fastshap.R` calculates Shapley values (based on the `fastshap` R package) 
for all models and stores the results in "iml/results/fastshap/shap.csv". 
A visualization of the global feature-wise Shapley values is also generated.

* `run_feat_effects.R` creates Partial Dependence Plots (PDP) and Individual 
Conditional Expectation (ICE) plots for each model, saving them in the 
subfolders of "iml/results/effects/".

* `run_cf.R` calculates counterfactual explanations (based on the `mcceR` R package,
to be installed with `remotes::install_github("NorskRegnesentral/mcceR")` 
for all models and stores the results in "iml/results/cf/cf.csv". 
It provides a single counterfactual for a random sample of correctly classified *real* instances, to 
see how they can be turned into synthetic. 
A visualization of the how often the counterfactuals changes the different features for different models and 
synthesizers is also generated.
