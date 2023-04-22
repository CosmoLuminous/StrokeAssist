import numpy as np
from collections import Counter, defaultdict
from time import time
import re
from glob import glob
from sklearn.base import clone as CLONE
import warnings
from sklearn.exceptions import ConvergenceWarning
import sklearn.metrics as metrics

# Imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from pickle import dump, load

# Models
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor, CatBoostClassifier

# Custom functions
from src.utils import *
from src.dataset.preprocess import *

class EXP3():
    def __init__(self, args) -> None:
        self.args = args
        self.data_dir = os.path.relpath(args.data_dir)
        self.results_dir = os.path.relpath(args.results_dir)
        self.models_dir = os.path.relpath(args.models_dir)
        self.config_dir = os.path.relpath(args.config_dir)
        self.drop_all = args.drop_all
        self.k_fold = args.k_fold
        self.n_jobs = args.n_jobs
        self.random_state = args.random_state
        self.overwrite_models = args.overwrite_models
        self.imputation = None
        self.classification = None
        self.pred_class_list = ["GT", "BC", "RFC", "ADBC", "GBC", "BRFC", "BBC", "DTC", "CBC"]

        #relative paths
        self.data_read_dir = os.path.join(args.data_dir, "temp/data_345")
        self.temp_dir = os.path.join(args.data_dir, "temp/exp3")
        self.exp_result_dir = os.path.join(self.results_dir, "exp3")
        self.exp_models_dir = os.path.join(self.models_dir, "exp3")
        if self.drop_all:
            self.data_read_dir = os.path.join(args.data_dir, "temp_removed_all/data_345")            
            self.temp_dir = os.path.join(args.data_dir, "temp_removed_all/exp3")
            self.exp_result_dir = os.path.join(self.results_dir, "removed_all", "exp3")
            self.exp_models_dir = os.path.join(self.models_dir, "removed_all", "exp3")
        self.data_csv = os.path.join(self.data_dir, "target_ordinal_encoded_data.csv")
        self.result_excel = os.path.join(self.exp_result_dir, "EXP3_PERFORMANCE.xlsx")
        self.config = os.path.join(self.config_dir, "selected_columns.xlsx")
        self.drop_cols = os.path.join(self.config_dir, "remove_cols.xlsx")
        
    def models_init(self) -> None:
        """
        Initialize ML Models for Imputation and Prediction.
        """
        classification_models = [
            BaggingClassifier(n_jobs = self.n_jobs, random_state = self.random_state),
            RandomForestClassifier(n_jobs = self.n_jobs, random_state = self.random_state),
            AdaBoostClassifier(random_state = self.random_state),
            GradientBoostingClassifier(random_state = self.random_state),
            BalancedRandomForestClassifier(n_jobs = self.n_jobs, random_state = self.random_state),
            BalancedBaggingClassifier(random_state = self.random_state, n_jobs=self.n_jobs),
            DecisionTreeClassifier(random_state = self.random_state), 
            CatBoostClassifier(silent = True, random_seed = self.random_state, thread_count=self.n_jobs)
            ]
        
        classification = defaultdict(list)
        classification["MODEL"].extend(classification_models)
        classification["ALIAS"].extend(["BC", "RFC", "ADBC", "GBC", "BRFC", "BBC", "DTC", "CBC"])
        classification["NAME"].extend([
            "BaggingClassifier",
            "RandomForestClassifier",
            "AdaBoostClassifier",
            "GradientBoostingClassifier",
            "BalancedRandomForestClassifier",
            "BalancedBaggingClassifier",
            "DecisionTreeClassifier",
            "CatBoostClassifier"])
        self.classification = classification


    def generate_results(self) -> None:

        # variables
        x_train_list = []
        x_test_list = []
        y_train_list = []
        y_test_list = []

        PREDICTIONS = dict()
        RESULTS = defaultdict(defaultdict)
        NAMES = []   
        cm_list = ["TP", "FN", "FP", "TN"]

        if not os.path.isdir(self.exp_result_dir):
            os.mkdir(self.exp_result_dir)

        if not os.path.isdir(self.temp_dir):
            os.mkdir(self.temp_dir)

        if not os.path.isdir(self.data_read_dir):
            os.mkdir(self.data_read_dir)

        if not os.path.isdir(self.exp_models_dir):
            os.mkdir(self.exp_models_dir)


        # check if dataset is prepared
        if len(glob(os.path.join(self.data_read_dir, "*.csv"))) > 30:
            print("Dataset already prepared. Skipping data generation step. If you feel there is any discrepancy with the results the please clear ./data/temp/exp_345 directory to do a fresh start.")
        else:
            dataset = PREPROCESS(self.args)
            dataset.preprocess_data()
            print("Dataset preparation complete. All data files have been save at {}".format(self.data_read_dir))

        
        # init models
        self.models_init()

        for i, train_path in enumerate(glob(os.path.join(self.data_read_dir, "*train.csv"))):
            t = train_path.split("/")[-1].replace("_train.csv", "").split("_")
            name = "_".join(t[1:])
            test_path = train_path.replace("_train.csv", "_test.csv")

            
            X_train = read_data(train_path)
            X_test = read_data(test_path)
            
            y_train = list(X_train["LABELS"])
            y_test = list(X_test["LABELS"])
            
            X_train.drop("LABELS", axis=1, inplace=True)
            X_test.drop("LABELS", axis=1, inplace=True)
        
            pred_key = train_path.split("/")[-1].replace("_train.csv", "")
            if pred_key not in PREDICTIONS:
                PREDICTIONS[pred_key] = np.zeros((len(y_test), len(self.classification["MODEL"])+1))
                PREDICTIONS[pred_key][:,0] = y_test

            for j, model in enumerate(self.classification["MODEL"]):
            
                
                clf_name = self.classification["NAME"][j]
                key = name + "_" + clf_name
                
                print("\nFOLD: {}, DATA_CLASSIFIER: {}".format(t[0], key))

                clf_dump_name = os.path.join(self.exp_models_dir, "{}_{}_clf.pkl".format(t[0], key))
                if os.path.isfile(clf_dump_name) and not self.overwrite_models:
                    print("Loading previously saved classification model at = {}".format(clf_dump_name))
                    clf = load(open(clf_dump_name, 'rb'))
                    pred = clf.predict(X_test)
                    
                else:
                    print("Training classifier.")
                    clf = CLONE(model, safe=True)         
                    clf.fit(X_train, y_train)
                    pred = clf.predict(X_test)
                    dump(clf, open(clf_dump_name, 'wb'))
                    print("Model saved at = {}".format(clf_dump_name))


                
                
                if key not in RESULTS:
                    RESULTS[key] = {m: 0 for m in cm_list}
                print(PREDICTIONS[pred_key].shape)    
                PREDICTIONS[pred_key][:,j+1] = list(pred)
                
                tp, fn, fp, tn = metrics.confusion_matrix(y_test, pred).ravel()
                    
                RESULTS[key]['TP'] += int(tp)
                RESULTS[key]['FN'] += int(fn)
                RESULTS[key]['FP'] += int(fp)
                RESULTS[key]['TN'] += int(tn)


                performance = evaluate_performance(RESULTS[key]['TP'], RESULTS[key]['FN'], RESULTS[key]['FP'], RESULTS[key]['TN'])
                RESULTS[key].update(performance)
                print("{}".format(RESULTS[key]))
                
                R = np.zeros([len(RESULTS.keys()), 12])      
                for t, val in enumerate(RESULTS.items()):
                    p, performance = val
                    R[t,:] = list(performance.values())
                    
                R = pd.DataFrame(R, index=RESULTS.keys(), columns=['TP','FN','FP','TN','WA', 'ACCU', 'SEN', 'SPE', 'GM', 'PRE', 'RECALL', 'F1'])
                
                R.to_excel(self.result_excel)
                
                for t, val in PREDICTIONS.items():
                    if t == pred_key:
                        path = os.path.join(self.temp_dir, "{}_predictions.csv".format(t))
                        val_df = pd.DataFrame(val,columns=self.pred_class_list).astype(int)
                        val_df.to_csv(path, index=False)
    




