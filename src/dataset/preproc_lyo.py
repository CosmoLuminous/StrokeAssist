import numpy as np
from collections import Counter, defaultdict
from sklearn.base import clone as CLONE
import sklearn.metrics as metrics
from glob import glob

# Imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from pickle import dump, load

# Models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor, CatBoostClassifier

# Custom functions
from src.utils import *

class PREPROCESS_LYO():
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
        self.regression = None
        self.classification = None        
        self.pred_class_list = ["GT", "BC", "RFC", "ADBC", "GBC", "BRFC", "BBC", "DTC", "CBC"]

        #relative paths
        self.temp_dir = os.path.join(args.data_dir, "temp/exp6_prospective/lyo")
        self.imputed_dir = os.path.join(args.data_dir, "temp/exp6_prospective/lyo_imputed")
        self.exp_models_dir = os.path.join(self.models_dir, "lyo_imputed")
        if self.drop_all:
            self.temp_dir = os.path.join(args.data_dir, "temp_removed_all/exp6_prospective/lyo")
            self.imputed_dir = os.path.join(args.data_dir, "temp_removed_all/exp6_prospective/lyo_imputed")
            self.exp_models_dir = os.path.join(self.models_dir, "removed_all", "lyo_imputed")
            
        self.retro_data = os.path.join(self.data_dir, "target_ordinal_encoded_retro_data_with_date.csv")        
        self.prospective_data = os.path.join(self.data_dir, "target_ordinal_encoded_prospective_data.csv")
        self.config = os.path.join(self.config_dir, "selected_columns.xlsx")
        self.drop_cols = os.path.join(self.config_dir, "remove_cols.xlsx")
        self.drop_cols_all = os.path.join(self.config_dir, "remove_cols_all.xlsx")
        self.numerical_cols = os.path.join(self.config_dir, "numerical_attributes.xlsx")
        

    def models_init(self) -> None:
        """
        Initialize ML Models for regression and Prediction.
        """
        regression_models = [
            LinearRegression(),
            AdaBoostRegressor(random_state = self.random_state),
            RandomForestRegressor(random_state = self.random_state,  n_jobs = self.n_jobs),
            GradientBoostingRegressor(random_state = self.random_state),
            CatBoostRegressor(silent = True, random_seed = self.random_state, thread_count=self.n_jobs)
            ]
        regression = defaultdict(list)
        regression["MODEL"].extend(regression_models)
        regression["ALIAS"].extend([ "LR", "ADBR", "RFR", "GBR", "CBR" ])
        regression["NAME"].extend([
            "LinearRegression",
            "AdaBoostRegressor",
            "RandomForestRegressor",
            "GradientBoostingRegressor",
            "CatBoostRegressor"
            ])
        self.regression = regression

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


    def filter_labels(self, df):
        TARGET_LABELS = [1,2]
        arr = []
        labels_list = []
        df['Diagnosis - stroke type - coded'] = df['Diagnosis - stroke type - coded'].astype(int)
        print(Counter(df['Diagnosis - stroke type - coded']))

        return list(df['Diagnosis - stroke type - coded'] <= 2), labels_list
    
    
    def lyo_split(self):
        print("Generating leave one-year out train test split.")
        df = read_data(self.retro_data)
        grouped_df= df.groupby(["ONSET_YEAR"],sort = True)
        
        for idx, g in enumerate(grouped_df):
            key, group = g
            
            year_df = pd.DataFrame()
            
            for key_, group_ in grouped_df:
                
                if key != key_:
                    year_df = year_df.append(group_, ignore_index=True)
                
            train_csv = os.path.join(self.temp_dir, key+"_train.csv")
            test_csv = os.path.join(self.temp_dir, key+"_test.csv")
            if key != "NOT_AVAILABLE":
                year_df.to_csv(train_csv, index=None)
                group.to_csv(test_csv, index=None)
                print("Year = {}, Train Size = {}, Test Size = {}".format(key, len(year_df), len(group)))
    
    
    
    def preprocess_data(self) -> None:
        
        self.lyo_split()
        
        # variables
        x_train_list = []
        x_test_list = []
        y_train_list = []
        y_test_list = []
        year_list = []
        
        numerical_cols = []
        categorical_cols = []

        if not os.path.isdir(self.temp_dir):
            os.mkdir(self.temp_dir)
        if not os.path.isdir(self.imputed_dir):
            os.mkdir(self.imputed_dir)
        if not os.path.isdir(self.exp_models_dir):
            os.mkdir(self.exp_models_dir)
    
        # init models
        self.models_init()
        
        config = read_data(self.config)
        
        if self.drop_all:
            drop_cols = read_data(self.drop_cols_all)
        else:
            drop_cols = read_data(self.drop_cols)
        numerical_cols_df = read_data(self.numerical_cols)
        
        prospective_df = read_data(self.prospective_data)
        arr, labels_list = self.filter_labels(prospective_df)
        prospective_df = prospective_df[arr]
        prospective_df.reset_index(inplace=True)        
        y_prospective = prospective_df["Diagnosis - stroke type - coded"]
        
        train_data_list = glob(os.path.join(self.temp_dir, "*train.csv"))
        drop_list = list(drop_cols["attribute_name"])
        
        for ind, train_path in enumerate(train_data_list):
            test_path = train_path.replace("train", "test")
            train_df = read_data(train_path)
            test_df = read_data(test_path)
            year_list.append(train_path.split("/")[-1].split("_")[0])
            
            train_df.drop(drop_list, axis=1, inplace=True)
            train_df.fillna(np.nan, inplace=True)
            test_df.drop(drop_list, axis=1, inplace=True)
            test_df.fillna(np.nan, inplace=True)
    
            cols_list = list(train_df.columns)
            cols_list.remove("Diagnosis - stroke type - coded")            
            cols_list.remove("ONSET_YEAR")
            
            X_train = train_df.loc[:, cols_list]
            y_train = train_df.loc[:, "Diagnosis - stroke type - coded"].astype(int)
            
            X_test = test_df.loc[:, cols_list]
            y_test = test_df.loc[:, "Diagnosis - stroke type - coded"].astype(int)
            
            
            x_train_list.append(X_train)
            x_test_list.append(X_test)

            y_train_list.append(y_train)
            y_test_list.append(y_test)
        
        
        for c in list(x_train_list[0].columns):
            if c in list(numerical_cols_df["attribute_name"]):
                numerical_cols.append(c)
            else:
                categorical_cols.append(c)
        print("TOTAL COLUMNS = {}, NUM COLUMNS = {}, CAT COLUMNS = {}".format(len(list(x_train_list[0].columns)), len(numerical_cols), len(categorical_cols)))
    
        for i, data in enumerate(zip(year_list, x_train_list, y_train_list, x_test_list, y_test_list)):
        # data
            year, X_train, y_train, X_test, y_test = data
            X_train_cat_dict = {}
            X_train_num_dict = {}
            X_test_cat_dict = {}
            X_test_num_dict = {}
            Prospective_cat_dict = {}
            Prospective_num_dict = {}
            
            X_train_cat = X_train[categorical_cols]
            X_train_num = X_train[numerical_cols]
            X_test_cat = X_test[categorical_cols]
            X_test_num = X_test[numerical_cols]
            Prospective_cat = prospective_df[categorical_cols]
            Prospective_num = prospective_df[numerical_cols]

            print("\nFOLD: {}".format(i+1))
            print("NUMERICAL ATTRIBUTES")
            for j, base in enumerate(self.regression["MODEL"]):
                try:
                    imputer_alias = self.regression['ALIAS'][j]
                    imputer_name = self.regression['NAME'][j]

                    

                    print("\nREGRESSOR: {}".format(imputer_name))

                    imputer_dump_name = os.path.join(self.exp_models_dir, "{}_{}_imputer.pkl".format(year, imputer_alias))
                    if os.path.isfile(imputer_dump_name) and not self.overwrite_models:
                        print("Loading previously saved imputation model at = {}".format(imputer_dump_name))
                        imputer = load(open(imputer_dump_name, 'rb'))
                        X_train_num_ = pd.DataFrame(imputer.transform(X_train_num), columns=numerical_cols)
                        X_test_num_ = pd.DataFrame(imputer.transform(X_test_num), columns=numerical_cols)
                        Prospective_num_ = pd.DataFrame(imputer.transform(Prospective_num), columns=numerical_cols)
                    else:
                        print("Performing fresh imputation.")
                        base_est = CLONE(base, safe=True)
                        imputer = IterativeImputer(estimator=base_est, random_state=self.random_state, initial_strategy="mean",
                                        max_iter=20, verbose=2)
                        X_train_num_ = pd.DataFrame(imputer.fit_transform(X_train_num), columns=numerical_cols)
                        X_test_num_ = pd.DataFrame(imputer.transform(X_test_num), columns=numerical_cols)
                        Prospective_num_ = pd.DataFrame(imputer.transform(Prospective_num), columns=numerical_cols)
                        dump(imputer, open(imputer_dump_name, 'wb'))
                        print("Model saved at = {}".format(imputer_dump_name))


                    X_train_num_dict[imputer_alias] = X_train_num_
                    X_test_num_dict[imputer_alias] = X_test_num_
                    Prospective_num_dict[imputer_alias] = Prospective_num_

                except Exception as e:
                    with open("./lyo_preproc_err.txt","a") as f:
                        f.writelines(["\n", "\n", imputer_name, "\n", str(e)])
                    print(imputer_name, str(e))

            print("CATEGORICAL ATTRIBUTES")    
            for j, base in enumerate(self.classification["MODEL"]):
                try:
                    imputer_alias = self.classification['ALIAS'][j]
                    imputer_name = self.classification['NAME'][j]

                    print("\nCLASSIFIER: {}".format(imputer_name))
                    
                    imputer_dump_name = os.path.join(self.exp_models_dir, "{}_{}_imputer.pkl".format(year, imputer_alias))
                    if os.path.isfile(imputer_dump_name) and not self.overwrite_models:
                        print("Loading previously saved imputation model at = {}".format(imputer_dump_name))
                        imputer = load(open(imputer_dump_name, 'rb'))
                        
                        X_train_cat_ = pd.DataFrame(imputer.transform(X_train_cat), columns=categorical_cols)
                        X_test_cat_ = pd.DataFrame(imputer.transform(X_test_cat), columns=categorical_cols)
                        Prospective_cat_ = pd.DataFrame(imputer.transform(Prospective_cat), columns=categorical_cols)
                    else:
                        print("Performing fresh imputation.")
                        base_est = CLONE(base, safe=True)
                        imputer = IterativeImputer(estimator=base_est, random_state=self.random_state, initial_strategy="most_frequent",
                                            max_iter=20, verbose=2)
                        X_train_cat_ = pd.DataFrame(imputer.fit_transform(X_train_cat), columns=categorical_cols)
                        X_test_cat_ = pd.DataFrame(imputer.transform(X_test_cat), columns=categorical_cols)
                        Prospective_cat_ = pd.DataFrame(imputer.transform(Prospective_cat), columns=categorical_cols)
                        dump(imputer, open(imputer_dump_name, 'wb'))
                        print("Model saved at = {}".format(imputer_dump_name))


                    X_train_cat_dict[imputer_alias] = X_train_cat_
                    X_test_cat_dict[imputer_alias] = X_test_cat_                    
                    Prospective_cat_dict[imputer_alias] = Prospective_cat_

                except Exception as e:
                    with open("./lyo_preproc_err.txt","a") as f:
                        f.writelines(["\n", "\n", imputer_name, "\n", str(e)])
                    print(imputer_name, str(e))


            print("SAVING TRAIN DATA...!")
            for reg, data_num in X_train_num_dict.items():
                for clf, data_cat in X_train_cat_dict.items():            
                    key = "{}_{}_{}_train.csv".format(year, reg, clf)
                    print(key)
                    data_merged = pd.concat([data_num, data_cat], axis=1)
                    data_merged.loc[:,"LABELS"] = list(y_train.to_numpy())
                    file_name = os.path.join(self.imputed_dir, key)
                    data_merged.to_csv(file_name, index=False)
                    
                    
            print("SAVING TEST DATA...!")                      
            for reg, data_num in X_test_num_dict.items():
                for clf, data_cat in X_test_cat_dict.items():
                    key =  "{}_{}_{}_test.csv".format(year, reg, clf)
                    print(key)
                    data_merged = pd.concat([data_num, data_cat], axis=1)
                    data_merged.loc[:,"LABELS"] = list(y_test.to_numpy())
                    file_name = os.path.join(self.imputed_dir, key)
                    data_merged.to_csv(file_name, index=False)
                    

            
            print("SAVING PROSPECTIVE DATA...!")
            for reg, data_num in Prospective_num_dict.items():
                for clf, data_cat in Prospective_cat_dict.items():            
                    key = "{}_{}_{}_prospective.csv".format(year, reg, clf)
                    print(key)
                    data_merged = pd.concat([data_num, data_cat], axis=1)
                    data_merged.loc[:,"LABELS"] = list(y_prospective.to_numpy())
                    file_name = os.path.join(self.imputed_dir, key)
                    data_merged.to_csv(file_name, index=False)