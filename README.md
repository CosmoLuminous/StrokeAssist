# TODO
- [ ] Normalize Data
- [ ] Add to repository
- [ ] Run all experiments
- [ ] Verify results

# StrokeAssist
This is the official repository for the paper titled "***STROKE-ASSIST: a novel machine learning framework for stroke type identification in resource-constrained settings resilient to missing data.***"

______________________________________________________________________

## Experiment Details
* **Exp 1-5:** Refer to Methods section of the paper
* **Exp-6:** Performance calculation on prospective data
* **Exp-7:** Siriraj Score calculation on prospective data
* **Exp-8:** --excluded from the repository
* **Exp-9:** Leave One Year (LYO) out analysis for identification of domain/covariate shift in the data over the period of 10 years.
* **Exp-10:** SHAP analysis for identification of top attributes 
* **Exp-11:** Performance analysis on the interpretable models. 

## Requirements
This library has been developed on `python=3.7.15`. For rest of the requirements refer `requirements.txt`


## Run Commands

```shell
# Requirements installation
pip install -r requirements. txt

# To run all experiments one-by-one with all 69 attributes or provide multiple experiment numbers.

python -m src.run --exp_list 1-2-3-4-5 2>&1 | tee ./logs/exp.log

# For running experimentation with 20 attributes possibly causing target leakage removed run with --drop_all tag

python -m src.run --exp_list 1-2 --drop_all true 2>&1 | tee ./logs/exp_removed_all.log
```
> Note: Experimentation 3,4,5 require same set of data preparation, which may take time. 



## Arguments

```
optional arguments:
  -h, --help            show this help message and exit
  --data_dir            DATA_DIR             Path to dataset directory.
  --config_dir          CONFIG_DIR           Path to config files directory.
  --models_dir          MODELS_DIR           Path to export models directory.
  --results_dir         RESULTS_DIR          Path to export results directory.
  --exp_list            EXP_LIST             Provide comma separated list of experiment numbers to run e.g. 1,2,3
  --random_state        RANDOM_STATE         Random state for models and params initialization.
  --n_jobs              N_JOBS               Number of parallel jobs to run for models that support parallelization.
  --k_fold              K_FOLD               k-fold corss-validation.
  --run_best            RUN_BEST             Generate only results with best performing classifier for each experiment. If False then it will experiment with all the classifiers and regressors.
  --drop_all            DROP_ALL             drop all columns causing target leakage. If False then it will remove Total-NIHSS and Serum Homosystein
  --overwrite_models    OVERWRITE_MODELS     Overwrite previously saved models and do fresh imputation.
```

______________________________________________________________________

## Citation

```

```