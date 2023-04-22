import numpy as np
from collections import Counter, defaultdict
from time import time
import re
from glob import glob
from sklearn.base import clone as CLONE
import sklearn.metrics as metrics

# Imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from pickle import dump, load

# Models
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
from sklearn.linear_model import LinearRegression
from catboost import CatBoostClassifier

# Custom functions
from src.utils import *
from src.dataset.preprocess import *




class PROSPECTIVE():
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
        if self.drop_all:

            self.data_read_dir = os.path.join(args.data_dir, "temp_removed_all/data_345")
            self.temp_dir = os.path.join(args.data_dir, "temp_removed_all/exp6_prospective")
            self.temp_dir1 = os.path.join(args.data_dir, "temp_removed_all/exp6_prospective/e1")
            self.temp_dir2 = os.path.join(args.data_dir, "temp_removed_all/exp6_prospective/e2")
            self.temp_dir3 = os.path.join(args.data_dir, "temp_removed_all/exp6_prospective/e3")
            self.temp_dir4 = os.path.join(args.data_dir, "temp_removed_all/exp6_prospective/e4")
            self.temp_dir5 = os.path.join(args.data_dir, "temp_removed_all/exp6_prospective/e5")
            self.exp_result_dir = os.path.join(self.results_dir, "removed_all", "exp6_prospective")
            
            self.exp_models_dir1 = os.path.join(args.models_dir, "removed_all/exp1")
            self.exp_models_dir2 = os.path.join(args.models_dir, "removed_all/exp2")
            self.exp_models_dir3 = os.path.join(args.models_dir, "removed_all/exp3")
            self.exp_models_dir4 = os.path.join(args.models_dir, "removed_all/exp4")
            self.exp_models_dir5 = os.path.join(args.models_dir, "removed_all/exp5")

        else:
            self.data_read_dir = os.path.join(args.data_dir, "temp/data_345")        
            self.temp_dir = os.path.join(args.data_dir, "temp/exp6_prospective")     
            self.temp_dir1 = os.path.join(args.data_dir, "temp/exp6_prospective/e1")
            self.temp_dir2 = os.path.join(args.data_dir, "temp/exp6_prospective/e2")
            self.temp_dir3 = os.path.join(args.data_dir, "temp/exp6_prospective/e3")
            self.temp_dir4 = os.path.join(args.data_dir, "temp/exp6_prospective/e4")
            self.temp_dir5 = os.path.join(args.data_dir, "temp/exp6_prospective/e5")
            self.exp_result_dir = os.path.join(self.results_dir, "exp6_prospective")
            
            self.exp_models_dir1 = os.path.join(args.models_dir, "exp1")
            self.exp_models_dir2 = os.path.join(args.models_dir, "exp2")
            self.exp_models_dir3 = os.path.join(args.models_dir, "exp3")
            self.exp_models_dir4 = os.path.join(args.models_dir, "exp4")
            self.exp_models_dir5 = os.path.join(args.models_dir, "exp5")
            
        self.result_excel1 = os.path.join(self.exp_result_dir, "EXP6_PERFORMANCE_E1.xlsx")
        self.result_excel2 = os.path.join(self.exp_result_dir, "EXP6_PERFORMANCE_E2.xlsx")
        self.result_excel3 = os.path.join(self.exp_result_dir, "EXP6_PERFORMANCE_E3.xlsx")
        self.result_excel4 = os.path.join(self.exp_result_dir, "EXP6_PERFORMANCE_E4.xlsx")
        self.result_excel5 = os.path.join(self.exp_result_dir, "EXP6_PERFORMANCE_E5.xlsx")
        
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

    def evaluate_exp3_models(self) -> None:        

        PREDICTIONS = dict()
        RESULTS = defaultdict(defaultdict)
        NAMES = []   
        cm_list = ["TP", "FN", "FP", "TN"]

        
        if not os.path.isdir(self.temp_dir3):
            os.mkdir(self.temp_dir3)
            
        # init models
        self.models_init()

        for i, train_path in enumerate(glob(os.path.join(self.data_read_dir, "*train.csv"))):
            t = train_path.split("/")[-1].replace("_train.csv", "").split("_")
            name = "_".join(t[1:])
            prospective_path = train_path.replace("_train.csv", "_prospective.csv")

            
            X_train = read_data(train_path)
            X_prospective = read_data(prospective_path)
            
            y_train = list(X_train["LABELS"])
            y_prospective = list(X_prospective["LABELS"])
            
            X_train.drop("LABELS", axis=1, inplace=True)
            X_prospective.drop("LABELS", axis=1, inplace=True)
        
            pred_key = train_path.split("/")[-1].replace("_train.csv", "")
            if pred_key not in PREDICTIONS:
                PREDICTIONS[pred_key] = np.zeros((len(y_prospective), len(self.classification["MODEL"])+1))
                PREDICTIONS[pred_key][:,0] = y_prospective

            for j, model in enumerate(self.classification["MODEL"]):
            
                
                clf_name = self.classification["NAME"][j]
                key = name + "_" + clf_name
                
                print("\nEXP3 FOLD: {}, DATA_CLASSIFIER: {}".format(t[0], key))
                
                clf_dump_name = os.path.join(self.exp_models_dir3, "{}_{}_clf.pkl".format(t[0], key))
                if os.path.isfile(clf_dump_name) and not self.overwrite_models:
                    print("Loading previously saved classification model at = {}".format(clf_dump_name))
                    clf = load(open(clf_dump_name, 'rb'))
                    pred = clf.predict(X_prospective)
                    
                else:
                    print("Training classifier.")
                    clf = CLONE(model, safe=True)         
                    clf.fit(X_train, y_train)
                    pred = clf.predict(X_prospective)
                    # dump(clf, open(clf_dump_name, 'wb'))
                    # print("Model saved at = {}".format(clf_dump_name))
                

                if key not in RESULTS:
                    RESULTS[key] = {m: 0 for m in cm_list}
                print(PREDICTIONS[pred_key].shape)    
                PREDICTIONS[pred_key][:,j+1] = list(pred)
                
                tp, fn, fp, tn = metrics.confusion_matrix(y_prospective, pred).ravel()
                    
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
                
                R.to_excel(self.result_excel3)
                
                for t, val in PREDICTIONS.items():
                    if t == pred_key:
                        path = os.path.join(self.temp_dir3, "{}_predictions.csv".format(t))
                        val_df = pd.DataFrame(val,columns=self.pred_class_list).astype(int)
                        val_df.to_csv(path, index=False)
    
        return

    def evaluate_exp4_models(self) -> None:        

        PREDICTIONS = dict()
        RESULTS = defaultdict(defaultdict)
        NAMES = []   
        cm_list = ["TP", "FN", "FP", "TN"]

        
        if not os.path.isdir(self.temp_dir4):
            os.mkdir(self.temp_dir4)
            
        # init models
        self.models_init()

        numerical_cols_df = read_data(self.numerical_cols)

        for i, train_path in enumerate(glob(os.path.join(self.data_read_dir, "*train.csv"))):
            t = train_path.split("/")[-1].replace("_train.csv", "").split("_")
            name = "_".join(t[1:])
            prospective_path = train_path.replace("_train.csv", "_prospective.csv")

            
            X_train = read_data(train_path)
            X_prospective = read_data(prospective_path)
            
            y_train = list(X_train["LABELS"])
            y_prospective = list(X_prospective["LABELS"])
            
            X_train.drop("LABELS", axis=1, inplace=True)
            X_prospective.drop("LABELS", axis=1, inplace=True)


            numerical_cols = []
            categorical_cols = []            

            for c in list(X_train.columns):
                if c in list(numerical_cols_df["attribute_name"]):
                    numerical_cols.append(c)
                else:
                    categorical_cols.append(c)
            
            # ONE HOT ENCODING
            temp_data = pd.concat([X_train, X_prospective], axis=0)
            temp_ohe = pd.get_dummies(temp_data, columns=categorical_cols)
            X_train = temp_ohe[:len(X_train)]
            X_prospective = temp_ohe[len(X_train):]

            assert len(X_train) == len(y_train) and len(X_prospective) == len(y_prospective)

                    
            pred_key = train_path.split("/")[-1].replace("_train.csv", "")
            if pred_key not in PREDICTIONS:
                PREDICTIONS[pred_key] = np.zeros((len(y_prospective), len(self.classification["MODEL"])+1))
                PREDICTIONS[pred_key][:,0] = y_prospective

            for j, model in enumerate(self.classification["MODEL"]):
            
                
                clf_name = self.classification["NAME"][j]
                key = name + "_" + clf_name
                
                print("\nEXP4 FOLD: {}, DATA_CLASSIFIER: {}".format(t[0], key))
                
                clf_dump_name = os.path.join(self.exp_models_dir4, "{}_{}_clf.pkl".format(t[0], key))
                try:
                    if os.path.isfile(clf_dump_name) and not self.overwrite_models:
                        print("Loading previously saved classification model at = {}".format(clf_dump_name))
                        
                        clf = load(open(clf_dump_name, 'rb'))
                        pred = clf.predict(X_prospective)
                        
                    else:
                        print("Training classifier.")
                        clf = CLONE(model, safe=True)         
                        clf.fit(X_train, y_train)
                        pred = clf.predict(X_prospective)
                        # dump(clf, open(clf_dump_name, 'wb'))
                        # print("Model saved at = {}".format(clf_dump_name))
                except Exception as e:
                        print("ERROR OCCURED: Training classifier.")
                        with open("./logs/exp6_error.txt","a") as f:
                            f.writelines(["\n", "\n","EXP4", key, "\n", str(e)])
                        clf = CLONE(model, safe=True)         
                        clf.fit(X_train, y_train)
                        pred = clf.predict(X_prospective)
                
                if key not in RESULTS:
                    RESULTS[key] = {m: 0 for m in cm_list}
                    
                PREDICTIONS[pred_key][:,j+1] = list(pred)
                
                tp, fn, fp, tn = metrics.confusion_matrix(y_prospective, pred).ravel()
                    
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
                
                R.to_excel(self.result_excel4)
                
                for t, val in PREDICTIONS.items():
                    if t == pred_key:
                        path = os.path.join(self.temp_dir4, "{}_predictions.csv".format(t))
                        val_df = pd.DataFrame(val,columns=self.pred_class_list).astype(int)
                        val_df.to_csv(path, index=False)
    

        return

    def evaluate_exp5_models(self) -> None:        

        PREDICTIONS = dict()
        RESULTS = defaultdict(defaultdict)
        NAMES = []   
        cm_list = ["TP", "FN", "FP", "TN"]

        
        if not os.path.isdir(self.temp_dir5):
            os.mkdir(self.temp_dir5)
            
        # init models
        self.models_init()

        numerical_cols_df = read_data(self.numerical_cols)
        
        for i, train_path in enumerate(glob(os.path.join(self.data_read_dir, "*train.csv"))):
            t = train_path.split("/")[-1].replace("_train.csv", "").split("_")
            name = "_".join(t[1:])
            prospective_path = train_path.replace("_train.csv", "_prospective.csv")

            
            X_train = read_data(train_path)
            X_prospective = read_data(prospective_path)
            
            y_train = list(X_train["LABELS"])
            y_prospective = list(X_prospective["LABELS"])
            
            X_train.drop("LABELS", axis=1, inplace=True)
            X_prospective.drop("LABELS", axis=1, inplace=True)


            numerical_cols = []
            categorical_cols = []            

            for c in list(X_train.columns):
                if c in list(numerical_cols_df["attribute_name"]):
                    numerical_cols.append(c)
                else:
                    categorical_cols.append(c)
            
            # ONE HOT ENCODING
            temp_data = pd.concat([X_train, X_prospective], axis=0)
            temp_ohe = pd.get_dummies(temp_data, columns=categorical_cols)
            X_train = temp_ohe[:len(X_train)]
            X_prospective = temp_ohe[len(X_train):]

            #BOTH OHE and LE
            X_train.loc[:,list(temp_ohe.columns)] = temp_ohe[:len(X_train)]
            X_prospective.loc[:,list(temp_ohe.columns)] = temp_ohe[len(X_train):]
            print(X_train.shape, X_prospective.shape)

            assert len(X_train) == len(y_train) and len(X_prospective) == len(y_prospective)

                    
            pred_key = train_path.split("/")[-1].replace("_train.csv", "")
            if pred_key not in PREDICTIONS:
                PREDICTIONS[pred_key] = np.zeros((len(y_prospective), len(self.classification["MODEL"])+1))
                PREDICTIONS[pred_key][:,0] = y_prospective

            for j, model in enumerate(self.classification["MODEL"]):
            
                clf_name = self.classification["NAME"][j]
                key = name + "_" + clf_name
                
                print("\nEXP5 FOLD: {}, DATA_CLASSIFIER: {}".format(t[0], key))

                clf_dump_name = os.path.join(self.exp_models_dir5, "{}_{}_clf.pkl".format(t[0], key))
                try:
                    if os.path.isfile(clf_dump_name) and not self.overwrite_models:
                        print("Loading previously saved classification model at = {}".format(clf_dump_name))
                        print(X_train.shape, X_prospective.shape)
                        for col in X_train.columns:
                            if col not in list(X_prospective.columns):
                                print(col)
                        clf = load(open(clf_dump_name, 'rb'))
                        pred = clf.predict(X_prospective)
                        
                    else:
                        print("Training classifier.")
                        clf = CLONE(model, safe=True)         
                        clf.fit(X_train, y_train)
                        pred = clf.predict(X_prospective)
                        # dump(clf, open(clf_dump_name, 'wb'))
                        # print("Model saved at = {}".format(clf_dump_name))
                except Exception as e:
                        print("ERROR OCCURED: Training classifier.")
                        with open("./logs/exp6_error.txt","a") as f:
                            f.writelines(["\n", "\n", "EXP5", key, "\n", str(e)])
                        clf = CLONE(model, safe=True)         
                        clf.fit(X_train, y_train)
                        pred = clf.predict(X_prospective)
                
                if key not in RESULTS:
                    RESULTS[key] = {m: 0 for m in cm_list}
                    
                PREDICTIONS[pred_key][:,j+1] = list(pred)
                
                tp, fn, fp, tn = metrics.confusion_matrix(y_prospective, pred).ravel()
                    
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
                
                R.to_excel(self.result_excel5)
                
                for t, val in PREDICTIONS.items():
                    if t == pred_key:
                        path = os.path.join(self.temp_dir5, "{}_predictions.csv".format(t))
                        val_df = pd.DataFrame(val,columns=self.pred_class_list).astype(int)
                        val_df.to_csv(path, index=False)

        
        
        return

    def generate_results(self) -> None:

        if not os.path.isdir(self.data_read_dir):
            os.mkdir(self.data_read_dir)            

        if not os.path.isdir(self.exp_result_dir):
            os.mkdir(self.exp_result_dir)

        if not os.path.isdir(self.temp_dir):
            os.mkdir(self.temp_dir)

        # self.evaluate_exp3_models()
        self.evaluate_exp4_models()
        self.evaluate_exp5_models()

        return




