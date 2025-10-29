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
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor, CatBoostClassifier
# New classifier imports
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

# Custom functions
from src.utils import *
from src.dataset.preprocess import *

class EXP5():
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
        # Updated prediction class list with new model aliases
        self.pred_class_list = ["GT", "BC", "RFC", "ADBC", "GBC", "BRFC", "BBC", "DTC", "CBC", "XGBC", "TNC"]

        #relative paths
        self.data_read_dir = os.path.join(args.data_dir, "temp/data_345")
        self.temp_dir = os.path.join(args.data_dir, "temp/exp5")
        self.exp_result_dir = os.path.join(self.results_dir, "exp5")
        self.exp_models_dir = os.path.join(self.models_dir, "exp5")
        if self.drop_all:
            self.data_read_dir = os.path.join(args.data_dir, "temp_removed_all/data_345")            
            self.temp_dir = os.path.join(args.data_dir, "temp_removed_all/exp5")
            self.exp_result_dir = os.path.join(self.results_dir, "removed_all", "exp5")
            self.exp_models_dir = os.path.join(self.models_dir, "removed_all", "exp5")
        self.data_csv = os.path.join(self.data_dir, "target_ordinal_encoded_data.csv")
        self.result_excel = os.path.join(self.exp_result_dir, "EXP5_PERFORMANCE.xlsx")
        self.config = os.path.join(self.config_dir, "selected_columns.xlsx")
        self.drop_cols = os.path.join(self.config_dir, "remove_cols.xlsx")
        self.numerical_cols = os.path.join(self.config_dir, "numerical_attributes.xlsx")
            
        
    def models_init(self) -> None:
        """
        Initialize ML Models for Imputation and Prediction.
        """
        # Added XGBoost and TabNet classifiers
        classification_models = [
            BaggingClassifier(n_jobs = self.n_jobs, random_state = self.random_state),
            RandomForestClassifier(n_jobs = self.n_jobs, random_state = self.random_state),
            AdaBoostClassifier(random_state = self.random_state),
            GradientBoostingClassifier(random_state = self.random_state),
            BalancedRandomForestClassifier(n_jobs = self.n_jobs, random_state = self.random_state),
            BalancedBaggingClassifier(random_state = self.random_state, n_jobs=self.n_jobs),
            DecisionTreeClassifier(random_state = self.random_state), 
            CatBoostClassifier(silent = True, random_seed = self.random_state, thread_count=self.n_jobs),
            XGBClassifier(random_state=self.random_state, n_jobs=self.n_jobs, use_label_encoder=False, eval_metric='logloss'),
            TabNetClassifier(seed=self.random_state, verbose=0)
            ]
        
        classification = defaultdict(list)
        classification["MODEL"].extend(classification_models)
        # Updated aliases for the new models
        classification["ALIAS"].extend(["BC", "RFC", "ADBC", "GBC", "BRFC", "BBC", "DTC", "CBC", "XGBC", "TNC"])
        # Updated names for the new models
        classification["NAME"].extend([
            "BaggingClassifier",
            "RandomForestClassifier",
            "AdaBoostClassifier",
            "GradientBoostingClassifier",
            "BalancedRandomForestClassifier",
            "BalancedBaggingClassifier",
            "DecisionTreeClassifier",
            "CatBoostClassifier",
            "XGBoostClassifier",
            "TabNetClassifier"])
        self.classification = classification


    def generate_results(self) -> None:

        PREDICTIONS = dict()
        PROB_PREDICTIONS = dict()
        RESULTS = defaultdict(defaultdict)
        NAMES = []   
        cm_list = ["TP", "FN", "FP", "TN"]

        if not os.path.isdir(self.exp_result_dir):
            os.makedirs(self.exp_result_dir)

        if not os.path.isdir(self.temp_dir):
            os.makedirs(self.temp_dir)

        if not os.path.isdir(self.data_read_dir):
            os.makedirs(self.data_read_dir)        
        
        if not os.path.isdir(self.exp_models_dir):
            os.makedirs(self.exp_models_dir)

        numerical_cols_df = read_data(self.numerical_cols)

        # check if dataset is prepared
        if len(glob(os.path.join(self.data_read_dir, "*.csv"))) > 30:
            print("Dataset already prepared. Skipping data generation step. If you feel there is any discrepancy with the results the please clear ./data/temp/exp_345 directory to do a fresh start.")
        else:
            dataset = PREPROCESS(self.args)
            dataset.preprocess_data()
            print(f"Dataset preparation complete. All data files have been save at {self.data_read_dir}")

        
        # init models
        self.models_init()
        
        # Using sorted to ensure train and test files match up correctly
        train_files = sorted(glob(os.path.join(self.data_read_dir, "*train.csv")))

        for i, train_path in enumerate(train_files):
            # Correctly handle path splitting on both Windows and Unix-like systems
            path_parts = train_path.replace("\\", "/").split("/")
            file_id_part = path_parts[-1].replace("_train.csv", "")
            fold_num = file_id_part.split("_")[0]
            name = "_".join(file_id_part.split("_")[1:])
            
            test_path = train_path.replace("_train.csv", "_test.csv")

            
            X_train_df = read_data(train_path)
            X_test_df = read_data(test_path)
            
            y_train = list(X_train_df["LABELS"])
            y_test = list(X_test_df["LABELS"])
            
            X_train_df.drop("LABELS", axis=1, inplace=True)
            X_test_df.drop("LABELS", axis=1, inplace=True)


            numerical_cols = []
            categorical_cols = []            

            for c in list(X_train_df.columns):
                if c in list(numerical_cols_df["attribute_name"]):
                    numerical_cols.append(c)
                else:
                    categorical_cols.append(c)
            
            # ONE HOT ENCODING
            temp_data = pd.concat([X_train_df, X_test_df], axis=0)
            temp_ohe = pd.get_dummies(temp_data, columns=categorical_cols)
            X_train_ohe = temp_ohe[:len(X_train_df)]
            X_test_ohe = temp_ohe[len(X_train_df):]

            # NOTE: This block seems redundant but is kept to preserve original logic.
            # BOTH OHE and LE
            X_train_ohe.loc[:,list(temp_ohe.columns)] = temp_ohe[:len(X_train_df)]
            X_test_ohe.loc[:,list(temp_ohe.columns)] = temp_ohe[len(X_train_df):]
            print(X_train_ohe.shape, X_test_ohe.shape)

            assert len(X_train_ohe) == len(y_train) and len(X_test_ohe) == len(y_test)

            # Convert data to NumPy arrays for wider compatibility (especially for TabNet)
            X_train = X_train_ohe.to_numpy()
            X_test = X_test_ohe.to_numpy()
            y_train_np = np.array(y_train)
            y_test_np = np.array(y_test)
            y_train_tabnet = y_train_np.reshape(-1, 1) # Reshape y_train for TabNet
                    
            pred_key = file_id_part
            if pred_key not in PREDICTIONS:
                PREDICTIONS[pred_key] = np.zeros((len(y_test), len(self.classification["MODEL"])+1))
                PREDICTIONS[pred_key][:,0] = y_test_np

                PROB_PREDICTIONS[pred_key] = np.zeros((len(y_test), len(self.classification["MODEL"])+1))
                PROB_PREDICTIONS[pred_key][:,0] = y_test_np

            for j, model in enumerate(self.classification["MODEL"]):
                
                clf_alias = self.classification["ALIAS"][j] # Use alias for shorter key
                clf_name = self.classification["NAME"][j] # Use full name for messages
                key = name + "_" + clf_name
                
                print(f"\nFOLD: {fold_num}, DATA_CLASSIFIER: {key}")
                clf_dump_name = os.path.join(self.exp_models_dir, f"{fold_num}_{key}_clf.pkl")
                if os.path.isfile(clf_dump_name) and not self.overwrite_models:
                    print(f"Loading previously saved classification model at = {clf_dump_name}")
                    clf = load(open(clf_dump_name, 'rb'))
                    pred = clf.predict(X_test)
                    prob_pred = clf.predict_proba(X_test)
                    
                else:
                    print("Training classifier.")
                    clf = CLONE(model, safe=True)
                    if clf_name == "XGBoostClassifier":
                        clf.fit(X_train, y_train_np - 1)
                    else:            
                        clf.fit(X_train, y_train_np)

                    prob_pred = clf.predict_proba(X_test)
                    pred = clf.predict(X_test)
                    dump(clf, open(clf_dump_name, 'wb'))
                    print(f"Model saved at = {clf_dump_name}")
                
                if clf_name == "XGBoostClassifier":
                        pred = pred + 1

                if key not in RESULTS:
                    RESULTS[key] = {m: 0 for m in cm_list}
                    
                PREDICTIONS[pred_key][:,j+1] = list(pred)
                PROB_PREDICTIONS[pred_key][:,j+1] = list(np.round(prob_pred[:,0], 2))
                
                tp, fn, fp, tn = metrics.confusion_matrix(y_test_np, pred).ravel()
                    
                RESULTS[key]['TP'] += int(tp)
                RESULTS[key]['FN'] += int(fn)
                RESULTS[key]['FP'] += int(fp)
                RESULTS[key]['TN'] += int(tn)


                performance = evaluate_performance(RESULTS[key]['TP'], RESULTS[key]['FN'], RESULTS[key]['FP'], RESULTS[key]['TN'])
                RESULTS[key].update(performance)
                print(f"{RESULTS[key]}")
            
                # This block for saving results should be outside the inner loop. But kept inside to save on every iteration.
                R = np.zeros([len(RESULTS.keys()), 12])      
                for t_idx, (p_key, performance) in enumerate(RESULTS.items()):
                    R[t_idx,:] = list(performance.values())
                    
                R = pd.DataFrame(R, index=RESULTS.keys(), columns=['TP','FN','FP','TN','WA', 'ACCU', 'SEN', 'SPE', 'GM', 'PRE', 'RECALL', 'F1'])
                
                R.to_excel(self.result_excel)
                
                for t_key, val in PREDICTIONS.items():
                    if t_key == pred_key:
                        path = os.path.join(self.temp_dir, f"{t_key}_predictions.csv")
                        val_df = pd.DataFrame(val,columns=self.pred_class_list).astype(int)
                        val_df.to_csv(path, index=False)

                for t_key, val in PROB_PREDICTIONS.items():
                    if t_key == pred_key:
                        path = os.path.join(self.temp_dir, f"{t_key}_prob_lable1_predictions.csv")
                        val_df = pd.DataFrame(val,columns=self.pred_class_list)
                        val_df.to_csv(path, index=False)