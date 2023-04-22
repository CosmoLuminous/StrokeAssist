import numpy as np
from collections import Counter, defaultdict
import sys
from glob import glob
from sklearn.base import clone as CLONE
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler

# Imputation
from pickle import dump, load

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
import six
sys.modules['sklearn.externals.six'] = six
from id3 import Id3Estimator
from sklearn.svm import  SVC

# Custom functions
from src.utils import *
from src.dataset.preprocess import *
from src.dataset.preproc_lyo import *


class INTERPRETABLE():
    def __init__(self, args) -> None:
        self.data_dir = os.path.relpath(args.data_dir)
        self.results_dir = os.path.relpath(args.results_dir)
        self.models_dir = os.path.relpath(args.models_dir)
        self.config_dir = os.path.relpath(args.config_dir)
        self.drop_all = args.drop_all
        self.k_fold = args.k_fold
        self.n_jobs = args.n_jobs
        self.random_state = args.random_state
        self.overwrite_models = args.overwrite_models
        self.best_data = "ADBR_BC" #Best performing data as per catboostclassifier
        self.best_exp = "4" #Best performing exp as per catboostclassifier
        self.pred_class_list = ["GT", "LR", "DTC-CART", "DTC-ID3", "SVC-LINEAR", "SVC-RBF", "DTC-CART-H3"] 

        #relative paths
        self.data_read_dir = os.path.join(args.data_dir, "temp/data_345")
        self.temp_dir = os.path.join(args.data_dir, "temp/exp11_interpretable")
        self.exp_result_dir = os.path.join(self.results_dir, "exp11_interpretable")
        self.exp_models_dir = os.path.join(self.models_dir, "exp11_interpretable")
        if self.drop_all:
            self.data_read_dir = os.path.join(args.data_dir, "temp_removed_all/data_345")
            self.temp_dir = os.path.join(args.data_dir, "temp_removed_all/exp11_interpretable")
            self.exp_result_dir = os.path.join(self.results_dir, "removed_all","exp11_interpretable")
            self.exp_models_dir = os.path.join(self.models_dir, "removed_all", "exp11_interpretable")

        self.data_csv = os.path.join(self.data_dir, "target_encoded_data.csv")
        self.result_excel_retro = os.path.join(self.exp_result_dir, "INTERPRETABLE_RETRO.xlsx")
        self.result_excel_prospective = os.path.join(self.exp_result_dir, "INTERPRETABLE_PROSPECTIVE.xlsx")
        self.config = os.path.join(self.config_dir, "selected_columns.xlsx")
        self.drop_cols = os.path.join(self.config_dir, "remove_cols.xlsx")
        self.drop_cols_all = os.path.join(self.config_dir, "remove_cols_all.xlsx")
        self.numerical_cols = os.path.join(self.config_dir, "numerical_attributes.xlsx")

    def models_init(self) -> None:
        """
        Initialize ML Models for Imputation and Prediction.
        """
        classification_models = [
            LogisticRegression(solver='liblinear', random_state=self.random_state, max_iter=5000),
            DecisionTreeClassifier(random_state=self.random_state), 
            Id3Estimator(),
            SVC(random_state=self.random_state,kernel="linear"),
            SVC(random_state=self.random_state,kernel="rbf"),
            DecisionTreeClassifier(random_state=self.random_state, max_depth=3)
            ]
        
        classification = defaultdict(list)
        classification["MODEL"].extend(classification_models)
        classification["ALIAS"].extend(["LR", "DTC-CART", "DTC-ID3", "SVC-LINEAR", "SVC-RBF", "DTC-CART-H3"])
        classification["NAME"].extend([
            "LogisticRegression",
            "DecisionTree-CART",
            "DecisionTree-ID3",
            "SupportVectorClassifier-linear",
            "SupportVectorClassifier-rbf",
            "DecisionTree-CART-H3"])
        self.classification = classification

    def generate_results(self):

        if not os.path.isdir(self.exp_result_dir):
            os.mkdir(self.exp_result_dir)

        if not os.path.isdir(self.temp_dir):
            os.mkdir(self.temp_dir)
        
        if not os.path.isdir(self.exp_models_dir):
            os.mkdir(self.exp_models_dir)


        TO_NORMALIZE = [True, False, False, True, True, False]
        RESULTS_PROSPECTIVE = defaultdict(defaultdict)
        RESULTS_TEST = defaultdict(defaultdict)        
        EXTRA_INFO_TEST = defaultdict(defaultdict)
        NAMES = []
        cm_list = ["TP", "FN", "FP", "TN"]
        
        numerical_cols_df = read_data(self.numerical_cols)
        # init models
        self.models_init()
        train_path_list = glob(os.path.join(self.data_read_dir, "*"+self.best_data+"_train.csv"))
        train_path_list.sort()

        for i, train_path in enumerate(train_path_list):
            t = train_path.split("/")[-1].replace("_train.csv", "").split("_")
            name = "_".join(t[1:])
            
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
            

            for j, model in enumerate(self.classification["MODEL"]):

                X_train_ = X_train.copy()
                X_test_ = X_test.copy()
                X_prospective_ = X_prospective.copy()
                
                if TO_NORMALIZE[j]:
                    sc = StandardScaler()
                    X_train_ = sc.fit_transform(X_train_) # normalizing the features
                    X_test_ = sc.fit_transform(X_test_)
                    X_prospective_ = sc.fit_transform(X_prospective_)

                clf_name = self.classification["NAME"][j]
                key = name + "_" + clf_name
                
                print("\nInterpretable Models. FOLD: {}, DATA_CLASSIFIER: {}".format(i, key))
                clf_dump_name = os.path.join(self.exp_models_dir, "{}_{}_clf.pkl".format(i, key))
                if os.path.isfile(clf_dump_name) and not self.overwrite_models:
                    print("Loading previously saved classification model at = {}".format(clf_dump_name))
                    clf = load(open(clf_dump_name, 'rb'))
                    pred_prospective = clf.predict(X_prospective_)
                    pred_test = clf.predict(X_test_)
                    
                else:
                    print("Training classifier.")
                    clf = CLONE(model, safe=True)            
                    clf.fit(X_train_, y_train)
                    pred_prospective = clf.predict(X_prospective_)
                    pred_test = clf.predict(X_test_)
                    dump(clf, open(clf_dump_name, 'wb'))
                    print("Model saved at = {}".format(clf_dump_name))

                if key not in RESULTS_PROSPECTIVE:
                    RESULTS_PROSPECTIVE[key] = {m: 0 for m in cm_list}
                    RESULTS_TEST[key] = {m: 0 for m in cm_list}

                tp, fn, fp, tn = metrics.confusion_matrix(y_prospective, pred_prospective).ravel()
                    
                RESULTS_PROSPECTIVE[key]['TP'] += int(tp)
                RESULTS_PROSPECTIVE[key]['FN'] += int(fn)
                RESULTS_PROSPECTIVE[key]['FP'] += int(fp)
                RESULTS_PROSPECTIVE[key]['TN'] += int(tn)
                
                
                tp, fn, fp, tn = metrics.confusion_matrix(y_test, pred_test).ravel()                
                    
                RESULTS_TEST[key]['TP'] += int(tp)
                RESULTS_TEST[key]['FN'] += int(fn)
                RESULTS_TEST[key]['FP'] += int(fp)
                RESULTS_TEST[key]['TN'] += int(tn)

                performance = evaluate_performance(RESULTS_PROSPECTIVE[key]['TP'], RESULTS_PROSPECTIVE[key]['FN'], RESULTS_PROSPECTIVE[key]['FP'], RESULTS_PROSPECTIVE[key]['TN'])
                RESULTS_PROSPECTIVE[key].update(performance)
                print("{}".format(RESULTS_PROSPECTIVE[key]))
                
                performance = evaluate_performance(RESULTS_TEST[key]['TP'], RESULTS_TEST[key]['FN'], RESULTS_TEST[key]['FP'], RESULTS_TEST[key]['TN'])
                RESULTS_TEST[key].update(performance)
                print("{}".format(RESULTS_TEST[key]))

                R = np.zeros([len(RESULTS_PROSPECTIVE.keys()), 12])      
                for t, val in enumerate(RESULTS_PROSPECTIVE.items()):
                    p, performance = val
                    R[t,:] = list(performance.values())

                R = pd.DataFrame(R, index=RESULTS_PROSPECTIVE.keys(), columns=['TP','FN','FP','TN','WA', 'ACCU', 'SEN', 'SPE', 'GM', 'PRE', 'RECALL', 'F1'])
                R.to_excel(self.result_excel_prospective)

                R = np.zeros([len(RESULTS_TEST.keys()), 12])
                for t, val in enumerate(RESULTS_TEST.items()):
                    p, performance = val
                    R[t,:] = list(performance.values())
                R = pd.DataFrame(R, index=RESULTS_TEST.keys(), columns=['TP','FN','FP','TN','WA', 'ACCU', 'SEN', 'SPE', 'GM', 'PRE', 'RECALL', 'F1'])
                R.to_excel(self.result_excel_retro)

