import numpy as np
from collections import Counter

# Custom functions
from src.utils import *

class SIRIRAJ():
    def __init__(self, args) -> None:
        self.data_dir = os.path.relpath(args.data_dir)
        self.results_dir = os.path.relpath(args.results_dir)
        self.models_dir = os.path.relpath(args.models_dir)
        self.config_dir = os.path.relpath(args.config_dir)
        self.k_fold = args.k_fold
        self.n_jobs = args.n_jobs
        self.random_state = args.random_state
        self.regression = None
        self.classification = None
        self.attributes = ["Level of Consciousness", "Vomiting", "Headache within 2Hrs", 
        "Blood pressure on admission/arrival to hospital", "Diabetes mellitus (DM)", 
        "Atheroma Marker - Angina", "Atheroma Marker - Intermittent claudication", "Diagnosis - stroke type - coded"]        

        #relative paths
        self.temp_dir = os.path.join(args.data_dir, "temp/exp7_siriraj")
        self.prospective_data = os.path.join(self.data_dir, "AIIMS-PROSPECTIVE.xlsx")
        self.exp_result_dir = os.path.join(self.results_dir, "exp7_siriraj")
        self.result_excel = os.path.join(self.exp_result_dir, "EXP7_PERFORMANCE.xlsx")
        self.config = os.path.join(self.config_dir, "selected_columns.xlsx")
        self.drop_cols = os.path.join(self.config_dir, "remove_cols.xlsx")
        self.numerical_cols = os.path.join(self.config_dir, "numerical_attributes.xlsx")

    def filter_labels(self, df):
        TARGET_LABELS = [1,2]
        arr = []
        labels_list = []
        for i, val in df.iterrows():        
            lables = str(val['Diagnosis - stroke type - coded']).split(";")        
            flag = False
            l = 0
            for v in lables:
                if int(v) in TARGET_LABELS:
                    flag = True
                    l = int(v)
            if flag:
                labels_list.append(l)
            arr.append(flag)
        return arr, labels_list

    def process_bp(self, df):
        high = []
        low = []

        for i, value in enumerate(df):
            value = str(value)
            if "/" in value:
                v = value.split("/")
                high.append(int(v[0]))
                low.append(int(v[1]))
            else:
                high.append(value)
                low.append(value)
        return low

    def atheroma_markers(self, df):

        df.loc[np.logical_or(np.logical_or(df.loc[:,"Atheroma Marker - Intermittent claudication"], df.loc[:,"Atheroma Marker - Angina"]), df.loc[:,"Diabetes mellitus (DM)"]), "Atheroma-Markers"] = 1
        df.loc[df.loc[:, "Atheroma-Markers"] != 1 ,"Atheroma-Markers"] = 0
        
        return df
        
    def calc_siriraj(self, df):

        df.loc[:, "Siriraj-Score"] = 2.5 * df["Level of Consciousness"] + 2 * df["Vomiting"] + 2 * df["Headache within 2Hrs"] + 0.1 * df["BP-Diastolic"] - 3 * df["Atheroma-Markers"] - 12

        return df


    def calc_performance(self, gt, pred):

        RESULTS = {"TP": 0, "FN": 0, "FP": 0, "TN":0, "EQ": 0}

        for i, val in enumerate(zip(gt, pred)):
            g, p = val
            if p != 3:
                if g == p:
                    if g == 1:                        
                        RESULTS["TP"] += 1
                    elif g == 2:                        
                        RESULTS["TN"] += 1
                else:                    
                    if g == 1:                        
                        RESULTS["FN"] += 1
                    elif g == 2:                        
                        RESULTS["FP"] += 1
            else:
                RESULTS["EQ"] += 1                                    
                if g == 1:                        
                    RESULTS["FN"] += 1
                elif g == 2:                        
                    RESULTS["FP"] += 1

        return RESULTS


    def run(self):

        if not os.path.isdir(self.temp_dir):
            os.mkdir(self.temp_dir)

        if not os.path.isdir(self.exp_result_dir):
            os.mkdir(self.exp_result_dir)
        data_df = read_data(self.prospective_data)
        arr, labels_list = self.filter_labels(data_df)
        labels_filtered_data_df = data_df[arr]

        labels_filtered_data_df.reset_index(inplace=True)


        filtered_data = labels_filtered_data_df.loc[:, self.attributes]
        filtered_data.loc[:, "BP-Diastolic"] = self.process_bp(filtered_data["Blood pressure on admission/arrival to hospital"])
        filtered_data.loc[filtered_data.loc[:, "Vomiting"] != 1 ,"Vomiting"] = 0        
        filtered_data.loc[filtered_data.loc[:, "Headache within 2Hrs"] != 1 ,"Headache within 2Hrs"] = 0        
        filtered_data.loc[filtered_data.loc[:, "Diabetes mellitus (DM)"] != 1 ,"Diabetes mellitus (DM)"] = 0        
        filtered_data.loc[filtered_data.loc[:, "Atheroma Marker - Angina"] != 1 ,"Atheroma Marker - Angina"] = 0        
        filtered_data.loc[filtered_data.loc[:, "Atheroma Marker - Intermittent claudication"] != 1 ,"Atheroma Marker - Intermittent claudication"] = 0        


        filtered_data = self.atheroma_markers(filtered_data)

        
        class_ratio = dict(Counter(filtered_data[ "Diagnosis - stroke type - coded"]))
        
        print("Processed dataframe shape =", filtered_data.shape)
        print("Original Class Distribution Ischemic Stroke = {}, Hemorrhagic Stroke = {}".format(class_ratio[1], class_ratio[2]))
        

        siriraj_df = self.calc_siriraj(filtered_data)        
        siriraj_df.loc[:, "Pred-Class"] = 3    
        siriraj_df.loc[filtered_data.loc[:, "Siriraj-Score"] <= -1 ,"Pred-Class"] = 1
        siriraj_df.loc[filtered_data.loc[:, "Siriraj-Score"] >= 1, "Pred-Class"] = 2  

        siriraj_df.to_excel(os.path.join(self.temp_dir, "SIRIRAJ_SCORE.xlsx"))

        RESULTS = self.calc_performance(list(filtered_data[ "Diagnosis - stroke type - coded"]), list(siriraj_df["Pred-Class"]))
        R = np.zeros([1, 15])
        R[0,0] = class_ratio[1]
        R[0,1] = class_ratio[2]

        R[0, 2:7] = list(RESULTS.values())
        performance = evaluate_performance(RESULTS['TP'], RESULTS['FN'], RESULTS['FP'], RESULTS['TN'])
        R[0, 7:] = list(performance.values())
        R = pd.DataFrame(R, columns=["GT-IS", "GT-HS", 'TP','FN','FP','TN', 'EQ','WA', 'ACCU', 'SEN', 'SPE', 'GM', 'PRE', 'RECALL', 'F1'])

        R.to_excel(self.result_excel)
        print("SIRIRAJ SCORE RESULTS ON PROSPECTIVE DATA")
        print(R)





        
