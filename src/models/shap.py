import numpy as np
from collections import Counter, defaultdict
from sklearn.base import clone as CLONE
import sklearn.metrics as metrics
import json
from glob import glob
import matplotlib.pyplot as plt

# Imputation
from pickle import dump, load

# Models
from catboost import  CatBoostClassifier, Pool, EShapCalcType, EFeaturesSelectionAlgorithm
from imblearn.ensemble import BalancedRandomForestClassifier

# Custom functions
from src.utils import *

class SHAP():
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
        self.n_features = 69
        self.pred_class_list = ["GT"]

        #relative paths
        self.data_read_dir = os.path.join(args.data_dir, "temp/data_345")
        self.temp_dir = os.path.join(args.data_dir, "temp/exp10_shap")
        self.exp_result_dir = os.path.join(self.results_dir, "exp10_shap")
        self.exp_models_dir = os.path.join(self.models_dir, "exp10_shap")
        if self.drop_all:
            self.data_read_dir = os.path.join(args.data_dir, "temp_removed_all/data_345")
            self.temp_dir = os.path.join(args.data_dir, "temp_removed_all/exp10_shap")
            self.exp_result_dir = os.path.join(self.results_dir, "removed_all","exp10_shap")
            self.exp_models_dir = os.path.join(self.models_dir, "removed_all", "exp10_shap")
            self.n_features = 49

        self.data_csv = os.path.join(self.data_dir, "target_encoded_data.csv")
        self.result_excel = os.path.join(self.exp_result_dir, "SHAP_SELECTED_FT_PERFORMANCE_BRFC.xlsx")
        self.result_json = os.path.join(self.exp_result_dir, "SHAP_FT_SELECTION.json")
        self.result_n_vs_wa_pdf = os.path.join(self.exp_result_dir, "N_FT_VS_WA.pdf")
        self.result_n_vs_wa_svg = os.path.join(self.exp_result_dir, "N_FT_VS_WA.svg")
        self.config = os.path.join(self.config_dir, "selected_columns.xlsx")
        self.drop_cols = os.path.join(self.config_dir, "remove_cols.xlsx")
        self.drop_cols_all = os.path.join(self.config_dir, "remove_cols_all.xlsx")
        self.numerical_cols = os.path.join(self.config_dir, "numerical_attributes.xlsx")
 
    def dataset(self):
        x_train_list = []
        x_test_list = []
        x_test_correct_list = []
        y_train_list = []
        y_test_list = []
        y_test_correct_list = []

        for i, train_path in enumerate(glob(os.path.join(self.data_read_dir, "*"+self.best_data+"_train.csv"))):
            print("Reading Data:", train_path)
            fold = train_path.split("/")[-1].split("_")[0]
            test_path = train_path.replace("train", "test")
            exp_pred_path = os.path.join(self.data_read_dir.replace("data_345", "exp"+self.best_exp), "_".join([fold, self.best_data, "predictions.csv"]))

            X_train = read_data(train_path)
            X_test = read_data(test_path)
            exp_predictions = read_data(exp_pred_path)

            y_train = X_train["LABELS"]
            y_test = X_test["LABELS"]
            
            
            print("FOLD: {}, Pred Split: {}".format(fold, Counter(exp_predictions["CBC"] == y_test)))
            X_test_correct_pred = X_test.loc[exp_predictions["CBC"] == y_test, :]
            y_test_correct_pred = X_test_correct_pred["LABELS"]

            X_train.drop(["LABELS"], axis=1, inplace=True)
            X_test.drop(["LABELS"], axis=1, inplace=True)
            X_test_correct_pred.drop(["LABELS"], axis=1, inplace=True)

            x_train_list.append(X_train)
            x_test_list.append(X_test)
            x_test_correct_list.append(X_test)

            y_train_list.append(y_train)
            y_test_list.append(y_test)
            y_test_correct_list.append(y_test)
        
        return x_train_list, x_test_list, x_test_correct_list, y_train_list, y_test_list, y_test_correct_list

    def generate_plots(self):
        ft_var = read_data(self.result_excel)
        ft_var = ft_var.sort_values("Unnamed: 0")
        fig = plt.figure(1) 
        
        wa = np.zeros(len(ft_var["WA"])+1)
        wa[1:] = list(ft_var["WA"])
        x_axis = list(range(self.n_features+1))
        
        # x_axis = [1, 5, 10, 14, 19, 24, 29, 34, 39, 45, 50, 55, 60, 65, len(ft_var["WA"])-2]
        x_axis = list(range(1, self.n_features+1, 3))
        wa = wa[x_axis]

        sen = np.zeros(len(ft_var["SEN"])+1)
        sen[1:] = list(ft_var["SEN"])
        sen = sen[x_axis] 

        spe = np.zeros(len(ft_var["SPE"])+1)
        spe[1:] = list(ft_var["SPE"])
        spe = spe[x_axis]

        plt.plot(x_axis, spe, c="tab:orange", label="Specificity",linestyle="dashed", marker=".",)
        plt.plot(x_axis, sen, c="tab:green", label="Sensitivity",linestyle="dotted", marker=".",)
        plt.plot(x_axis, wa, c="crimson", label="Weighted Accuracy", marker="o",)
        plt.xlabel("Number of Top Attributes ($N_{top}$)")
        plt.ylabel("Peformance Metrics (%)")
        plt.legend()
        # plt.show()
        fig.savefig(self.result_n_vs_wa_svg, dpi= 300, pad_inches=0, format="svg")
        fig.savefig(self.result_n_vs_wa_pdf, dpi= 300, pad_inches=0)
        
        return

    def run_shap(self):
        if not os.path.isdir(self.exp_result_dir):
            os.mkdir(self.exp_result_dir)

        if not os.path.isdir(self.temp_dir):
            os.mkdir(self.temp_dir)
        
        if not os.path.isdir(self.exp_models_dir):
            os.mkdir(self.exp_models_dir)

        x_train_list, x_test_list, x_test_correct_list, y_train_list, y_test_list, y_test_correct_list = self.dataset()

        
        PREDICTIONS = dict()
        RESULTS = defaultdict(defaultdict)
        NAMES = []   
        FEATURES = dict()
        cm_list = ["TP", "FN", "FP", "TN"]
        for i in range(1,self.n_features+1):
            self.pred_class_list.append("N_{}".format(i))
            FEATURES[i] = dict()
        
        for i in range(len(x_train_list)):
            X_train = x_train_list[i]
            X_test = x_test_list[i]
            X_test_correct_pred = x_test_correct_list[i]
            y_train = y_train_list[i]
            y_test = y_test_list[i]
            y_test_correct_pred = y_test_correct_list[i]
            print(X_train.shape, X_test.shape, X_test_correct_pred.shape)

            f_name = list(range(0, len(X_train.columns)))
            train_pool = Pool(X_train.to_numpy(), y_train.to_numpy(), feature_names=list(X_train.columns))
            test_pool = Pool(X_test_correct_pred.to_numpy(), y_test_correct_pred.to_numpy(), feature_names=list(X_train.columns))

            pred_key = "{}_{}".format(i, self.best_data)
            if pred_key not in PREDICTIONS:
                PREDICTIONS[pred_key] = np.zeros((len(y_test), self.n_features+1))
                PREDICTIONS[pred_key][:,0] = y_test
            
            for k in range(self.n_features,0,-1):
                print("\n\n FOLD: {}, N_Features: {}".format(i, k))

                shap_model_dump_name = os.path.join(self.exp_models_dir, "{}_{}_shap_model.pkl".format(i, k))
                if os.path.isfile(shap_model_dump_name) and not self.overwrite_models:
                    print("Loading previously saved classification model at = {}".format(shap_model_dump_name))
                    summary = load(open(shap_model_dump_name, 'rb'))
                    
                else:
                    print("Preforming new SHAP analysis.")
                    shap_model = CLONE(CatBoostClassifier(iterations=1000, silent = True, random_seed = self.random_state, thread_count=-1), safe=True)
                    summary = shap_model.select_features(
                                    train_pool,
                                    eval_set=test_pool,
                                    features_for_select=f_name,
                                    num_features_to_select=k,
                                    steps=5,
                                    algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
                                    shap_calc_type=EShapCalcType.Regular,
                                    train_final_model=False,
                                    logging_level='Silent',
                                    plot=False)
                    dump(summary, open(shap_model_dump_name, 'wb'))
                    print("Model saved at = {}".format(shap_model_dump_name))

                selected_features = summary['selected_features_names']

                if i not in FEATURES[k]:
                    FEATURES[k][i] = selected_features
                
                with open(self.result_json, "w") as f:
                    json.dump(FEATURES, f)   
                
                clf_dump_name = os.path.join(self.exp_models_dir, "{}_{}_clf.pkl".format(i, k))
                if os.path.isfile(clf_dump_name) and not self.overwrite_models:
                    print("Loading previously saved classification model at = {}".format(clf_dump_name))
                    clf = load(open(clf_dump_name, 'rb'))
                    pred = clf.predict(X_test)
                    
                else:
                    print("Training classifier.")
                    clf = CLONE(BalancedRandomForestClassifier(n_jobs = self.n_jobs, random_state = self.random_state), safe=True)        
                    clf.fit(X_train, y_train)
                    pred = clf.predict(X_test)
                    dump(clf, open(clf_dump_name, 'wb'))
                    print("Model saved at = {}".format(clf_dump_name))



                
                clf.fit(X_train[selected_features], y_train)
                pred = clf.predict(X_test[selected_features])
                
                PREDICTIONS[pred_key][:,k] = list(pred)
        
                tp, fn, fp, tn = metrics.confusion_matrix(y_test, pred).ravel()
                key = k    
                if key not in RESULTS:
                    RESULTS[key] = {m: 0 for m in cm_list}
                    
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

        
        return

    def generate_results(self):
        
        self.run_shap()
        self.generate_plots()
