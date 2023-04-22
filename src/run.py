import argparse
import os
from xmlrpc.client import boolean
from src.utils import *
from src.models import exp1, exp2, exp3, exp4, exp5, prospective, siriraj, shap, arvscore, lyo, interpretable
from src.dataset.preprocess import *


def get_parser():
    """
    Generate a parameter parser
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Stroke Assist")

    # path to .csv data files.
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to dataset directory.")

    # path to .csv config files.
    parser.add_argument("--config_dir", type=str, default="./config", help="Path to config files directory.")

    # path to .csv models directory.
    parser.add_argument("--models_dir", type=str, default="./models", help="Path to export models directory.")

    # path to .csv results directory.
    parser.add_argument("--results_dir", type=str, default="./results", help="Path to export results directory.")

    # experiments to run. default will run all experiments.
    parser.add_argument("--exp_list", type=str, default=None, 
    help="Provide comma separated list of experiment numbers to run e.g. 1,2,3")

    # random state for reproducible results.
    parser.add_argument("--random_state", type=int, default=51, help="Random state for models and params initialization.")

    # number of parallel jobs to run. default = -1 will utilize all the available cores
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs to run for models that support parallelization.")
    
    # k fold cross validation.
    parser.add_argument("--k_fold", type=int, default=10, help="k-fold corss-validation.")

    # generate only best model results for each experiment. 
    parser.add_argument("--run_best", type=bool, default=False, help="Generate only results with best performing classifier for each experiment. If False then it will experiment with all the classifiers and regressors.")

    # drop all columns causing target leakage
    parser.add_argument("--drop_all", type=bool, default=False, help="drop all columns causing target leakage. If False then it will remove Total-NIHSS and Serum Homosystein")

    # drop all columns causing target leakage
    parser.add_argument("--overwrite_models", type=bool, default=False, help="Overwrite previously saved models and do fresh imputation.")

    return parser

def run_experiment():
    
    # assert type(args.exp_list) == type(str), "data type of exp_list must be a list."
    # assert len(args.exp_list) > 0, "provide at least one experiment to run."
                        
    exp_list = list()
                        
    exp_list.extend(args.exp_list.split("-"))
    exp_list = [int(x) for x in exp_list if x != "-"]


    for exp in exp_list:
        assert exp in EXP_SET, f"Invalid experiment number: {exp}"
        
        if exp == 0:
            print("Data Preparation for Exp-3,4,5 begins")
            dataset = PREPROCESS(args)
            dataset.preprocess_data()
        
        if exp == 1:
            e1 = exp1.EXP1(args)
            e1.generate_results()
        
        elif exp == 2:
            e2 = exp2.EXP2(args)
            e2.generate_results()
        
        elif exp == 3:
            e3 = exp3.EXP3(args)
            e3.generate_results()
        
        elif exp == 4:
            e4 = exp4.EXP4(args)
            e4.generate_results()
        
        elif exp == 5:
            e5 = exp5.EXP5(args)
            e5.generate_results()
        
        elif exp == 6:
            e6 = prospective.PROSPECTIVE(args)
            e6.generate_results()

        elif exp == 7:
            e7 = siriraj.SIRIRAJ(args)
            e7.run()
            
        elif exp == 8:
            e8 = arvscore.SCORE(args)
            e8.generate_results()
            
        elif exp == 9:
            e9 = lyo.LYO(args)
            e9.generate_results()

        elif exp == 10:
            e10 = shap.SHAP(args)
            e10.generate_results()
            
        elif exp == 11:
            e11 = interpretable.INTERPRETABLE(args)
            e11.generate_results()
    


if __name__ == "__main__":
    #global variables
    TOTAL_EXP = 12
    EXP_SET = set(list(range(TOTAL_EXP)))

    #generate parser
    parser = get_parser()
    args = parser.parse_args()
    args.data_dir = os.path.relpath(args.data_dir)
    assert os.path.isdir(args.data_dir), f"invalid directory: {args.data_dir}"
    print(args)
    run_experiment()


