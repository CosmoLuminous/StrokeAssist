import numpy as np
from collections import Counter, defaultdict
from time import time
import re
from glob import glob
from sklearn.base import clone as CLONE
import sklearn.metrics as metrics

# Imputation
from sklearn.impute import IterativeImputer

# Models
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
from catboost import CatBoostClassifier

# Custom functions
from src.utils import *
from src.dataset.preprocess import *
from src.dataset.preproc_lyo import *




class LYO():
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
        self.imputation = None
        self.classification = None
        self.pred_class_list = ["GT", "BC", "RFC", "ADBC", "GBC", "BRFC", "BBC", "DTC", "CBC"]

        #relative paths
        if self.drop_all:

            self.data_read_dir = os.path.join(args.data_dir, "temp_removed_all/exp6_prospective/lyo_imputed")        
            self.temp_dir = os.path.join(args.data_dir, "temp_removed_all/exp6_prospective/lyo")  
            self.exp_result_dir = os.path.join(self.results_dir, "removed_all", "exp6_prospective", "lyo")

        else:
            self.data_read_dir = os.path.join(args.data_dir, "temp/exp6_prospective/lyo_imputed")        
            self.temp_dir = os.path.join(args.data_dir, "temp/exp6_prospective/lyo")
            self.exp_result_dir = os.path.join(self.results_dir, "exp6_prospective", "lyo")
            
#         self.result_excel1 = os.path.join(self.exp_result_dir, "EXP6_PERFORMANCE_E1.xlsx")
#         self.result_excel2 = os.path.join(self.exp_result_dir, "EXP6_PERFORMANCE_E2.xlsx")
        self.result_prospective_excel4 = os.path.join(self.exp_result_dir, "LYO_PROSPECTIVE_PERFORMANCE_E4.xlsx")
        self.result_test_excel4 = os.path.join(self.exp_result_dir, "LYO_TEST_PERFORMANCE_E4.xlsx")
#         self.result_excel4 = os.path.join(self.exp_result_dir, "EXP6_PERFORMANCE_E4.xlsx")
#         self.result_excel5 = os.path.join(self.exp_result_dir, "EXP6_PERFORMANCE_E5.xlsx")
        
        self.prospective_data = os.path.join(self.data_dir, "AIIMS-PROSPECTIVE.xlsx")
        self.config = os.path.join(self.config_dir, "selected_columns.xlsx")
        self.drop_cols = os.path.join(self.config_dir, "remove_cols.xlsx")
        self.numerical_cols = os.path.join(self.config_dir, "numerical_attributes.xlsx")
        
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

    def evaluate_exp4_models(self) -> None:        

#         PREDICTIONS = dict()
        RESULTS_PROSPECTIVE = defaultdict(defaultdict)
        RESULTS_TEST = defaultdict(defaultdict)        
        EXTRA_INFO_TEST = defaultdict(defaultdict)
        
        NAMES = []   
        cm_list = ["TP", "FN", "FP", "TN"]

        
        if not os.path.isdir(self.temp_dir):
            os.mkdir(self.temp_dir)
        
        numerical_cols_df = read_data(self.numerical_cols)

        # init models
        self.models_init()
        train_path_list = glob(os.path.join(self.data_read_dir, "*train.csv"))
        train_path_list.sort()
        yrs = set()
        
        for pth in train_path_list:
            yrs.add(pth.split("/")[-1].split("_")[0])
        
        print("######--------------- READING DATA FROM", self.data_read_dir, len(self.data_read_dir))        
        print("######--------------- TOTAL YEARS =", yrs)
        
        for i, train_path in enumerate(train_path_list):
            t = train_path.split("/")[-1].replace("_train.csv", "").split("_")
            name = "_".join(t)
            year = t[0]
            prospective_path = train_path.replace("_train.csv", "_prospective.csv")
            test_path = train_path.replace("_train.csv", "_test.csv")
            
            X_train = read_data(train_path)
            X_prospective = read_data(prospective_path)
            X_test = read_data(test_path)
            
            y_train = list(X_train["LABELS"])
            y_prospective = list(X_prospective["LABELS"])
            y_test = list(X_test["LABELS"])
            
            X_train.drop("LABELS", axis=1, inplace=True)
            X_prospective.drop("LABELS", axis=1, inplace=True)
            X_test.drop("LABELS", axis=1, inplace=True)

            numerical_cols = []
            categorical_cols = []            

            for c in list(X_train.columns):
                if c in list(numerical_cols_df["attribute_name"]):
                    numerical_cols.append(c)
                else:
                    categorical_cols.append(c)
            
            # ONE HOT ENCODING
            temp_data = pd.concat([X_train, X_test, X_prospective], axis=0)
            temp_ohe = pd.get_dummies(temp_data, columns=categorical_cols)
            X_train = temp_ohe[:len(X_train)]
            X_test = temp_ohe[len(X_train):len(X_train)+len(X_test)]
            X_prospective = temp_ohe[len(X_train)+len(X_test):]

            assert len(X_train) == len(y_train) and len(X_prospective) == len(y_prospective)
#             pred_key = train_path.split("/")[-1].replace("_train.csv", "")
#             if pred_key not in PREDICTIONS:
#                 PREDICTIONS[pred_key] = np.zeros((len(y_prospective), len(self.classification["MODEL"])+1))
#                 PREDICTIONS[pred_key][:,0] = y_prospective

            for j, model in enumerate(self.classification["MODEL"]):
            
                clf = CLONE(model, safe=True)
                clf_name = self.classification["NAME"][j]
                key = name + "_" + clf_name
                
                print("\nEXP4 FOLD: {}, DATA_CLASSIFIER: {}".format(year, key))
                
                clf.fit(X_train, y_train)
                pred_prospective = clf.predict(X_prospective)
                pred_test = clf.predict(X_test)
                
                if key not in RESULTS_PROSPECTIVE:
                    RESULTS_PROSPECTIVE[key] = {m: 0 for m in cm_list}
                    RESULTS_TEST[key] = {m: 0 for m in cm_list}
#                 print(PREDICTIONS[pred_key].shape)    
#                 PREDICTIONS[pred_key][:,j+1] = list(pred)
                tp, fn, fp, tn = metrics.confusion_matrix(y_prospective, pred_prospective).ravel()
                    
                RESULTS_PROSPECTIVE[key]['TP'] += int(tp)
                RESULTS_PROSPECTIVE[key]['FN'] += int(fn)
                RESULTS_PROSPECTIVE[key]['FP'] += int(fp)
                RESULTS_PROSPECTIVE[key]['TN'] += int(tn)
                
                try:
                    tp, fn, fp, tn = metrics.confusion_matrix(y_test, pred_test).ravel()
                except:
                    if y_test[0] == 1:
                        tp, fn, fp, tn = (1,0,0,0)
                    else:
                        tp, fn, fp, tn = (0,0,0,1)
                    
                RESULTS_TEST[key]['TP'] += int(tp)
                RESULTS_TEST[key]['FN'] += int(fn)
                RESULTS_TEST[key]['FP'] += int(fp)
                RESULTS_TEST[key]['TN'] += int(tn)
                EXTRA_INFO_TEST[key]['YEAR'] = year
                count_test = Counter(np.array(y_test).astype(int))
                EXTRA_INFO_TEST[key]['IS'] = count_test[1]
                    


                performance = evaluate_performance(RESULTS_PROSPECTIVE[key]['TP'], RESULTS_PROSPECTIVE[key]['FN'], RESULTS_PROSPECTIVE[key]['FP'], RESULTS_PROSPECTIVE[key]['TN'])
                RESULTS_PROSPECTIVE[key].update(performance)
                print("{}".format(RESULTS_PROSPECTIVE[key]))
                
                performance = evaluate_performance(RESULTS_TEST[key]['TP'], RESULTS_TEST[key]['FN'], RESULTS_TEST[key]['FP'], RESULTS_TEST[key]['TN'])
                RESULTS_TEST[key].update(performance)
                print("{}".format(RESULTS_TEST[key]))
                
                
                years_list = []
                R = np.zeros([len(RESULTS_PROSPECTIVE.keys()), 12])      
                for t, val in enumerate(RESULTS_PROSPECTIVE.items()):
                    p, performance = val
                    R[t,:] = list(performance.values())
                    years_list.append(int(EXTRA_INFO_TEST[p]["YEAR"]))
                    
                R = pd.DataFrame(R, index=RESULTS_PROSPECTIVE.keys(), columns=['TP','FN','FP','TN','WA', 'ACCU', 'SEN', 'SPE', 'GM', 'PRE', 'RECALL', 'F1'])
                R.loc[:, "YEAR"] = years_list
                R.loc[:, "IMP_CLF_KEY"] = [clf_key[5:] for clf_key in list(RESULTS_PROSPECTIVE.keys()) ]
                R.to_excel(self.result_prospective_excel4)
                
                R = np.zeros([len(RESULTS_TEST.keys()), 12])
                years_list = []
                is_list = []
                for t, val in enumerate(RESULTS_TEST.items()):
                    p, performance = val
                    R[t,:] = list(performance.values())
                    years_list.append(int(EXTRA_INFO_TEST[p]["YEAR"]))
                    is_list.append(int(EXTRA_INFO_TEST[p]["IS"]))
                    
                R = pd.DataFrame(R, index=RESULTS_TEST.keys(), columns=['TP','FN','FP','TN','WA', 'ACCU', 'SEN', 'SPE', 'GM', 'PRE', 'RECALL', 'F1'])
                R.loc[:, "YEAR"] = years_list
                R.loc[:, "IS-COUNT"] = is_list
                R.loc[:, "IMP_CLF_KEY"] = [clf_key[5:] for clf_key in list(RESULTS_TEST.keys()) ]
                R.to_excel(self.result_test_excel4)
                
    
        return

    def generate_results(self) -> None:

        if not os.path.isdir(self.data_read_dir):
            os.mkdir(self.data_read_dir)            

        if not os.path.isdir(self.exp_result_dir):
            os.mkdir(self.exp_result_dir)

        if not os.path.isdir(self.temp_dir):
            os.mkdir(self.temp_dir)
        
        # check if dataset is prepared
        if len(glob(os.path.join(self.data_read_dir, "*.csv"))) > 100:
            print("Dataset already prepared. Skipping data generation step. If you feel there is any discrepancy with the results the please clear ./data/temp/exp_345 directory to do a fresh start.")
        else:
            preproc = PREPROCESS_LYO(self.args)
            preproc.preprocess_data()
            print("Dataset preparation complete. All data files have been save at {}".format(self.data_read_dir))
        

        self.evaluate_exp4_models()
        # Doing Leave One Year analysis only for the best performing experiment i.e. EXP4
#         self.evaluate_exp3_models()
#         self.evaluate_exp5_models()

        return




