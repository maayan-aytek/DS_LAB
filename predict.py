import argparse
import os
import pickle
import pandas as pd
import numpy as np


def concat_all_patients(file_path):
    directory = os.listdir(file_path)
    all_df = pd.DataFrame()
    for filename in directory:
        with open(os.path.join(file_path, filename), 'r') as f:
            df = pd.read_csv(f, sep='|')
            df['patient id'] = filename[:-4]
            all_df = all_df.append(df, ignore_index=True)

    return all_df


def outliers_perc_of_col(target, df):
    df_not_null = df[target].dropna()
    Q1 = np.percentile(df_not_null, 25)
    Q3 = np.percentile(df_not_null, 75)
    IQR = Q3 - Q1
    ul = Q3 + 1.5 * IQR
    ll = Q1 - 1.5 * IQR
    outliers = df_not_null[(df_not_null > ul) | (df_not_null < ll)]
    outliers_per = round((len(outliers)/len(df_not_null))*100,2)
    return outliers_per


def pre_process(df):
    avg_dict = {'HR': 84.51555463407765, 'O2Sat': 97.14622407039397, 'Temp': 36.86225627343898, 'SBP': 123.54822621152614, 'MAP': 82.65786949525612, 'DBP': 63.72853564554743, 'Resp': 18.678977055200004, 'EtCO2': 33.062411655478535, 'BaseExcess': 0.00883608211932965, 'HCO3': 24.365009506000906, 'FiO2': 0.5097275467997768, 'pH': 7.391309780879508, 'PaCO2': 40.39648897356055, 'SaO2': 95.66168781719271, 'AST': 82.02189444557456, 'BUN': 22.50209223298439, 'Alkalinephos': 81.38623051204615, 'Calcium': 8.183274536957347, 'Chloride': 105.61275108099791, 'Creatinine': 1.4145731949030103, 'Bilirubin_direct': 0.5523103363742105, 'Glucose': 130.67136929444348, 'Lactate': 1.8869094674031528, 'Magnesium': 2.036425339425519, 'Phosphate': 3.4337188544812514, 'Potassium': 4.072790630656006, 'Bilirubin_total': 1.1490866284861856, 'TroponinI': 1.4645463199127902, 'Hct': 31.479664069142153, 'Hgb': 10.514291170502974, 'PTT': 34.909836989688145, 'WBC': 11.032792273668512, 'Fibrinogen': 262.5899049921501, 'Platelets': 201.84890309355782, 'Age': 62.03636171833029, 'Gender': 0.5554242884191091, 'Unit1': 0.6919664512494719, 'Unit2': 0.30803354875052813, 'HospAdmTime': -55.049495190928475, 'ICULOS': 26.57108506632033, 'SepsisLabel': 0.017580233777207504, 'patient id': 10052.137260654805}
    all_null_list = []
    patients_size = df.groupby("patient id").size()
    patients_size = patients_size[patients_size >= 36]
    full_hours_df = df[df['patient id'].isin(patients_size.index)].reset_index(drop=True)
    # backward and forward fill
    df_processed = df.groupby('patient id').apply(lambda x: x.bfill().ffill()).reset_index(drop=True)
    outlier_perc_dict = {}
    # Filling the rest of null values based on median/mean/mode according to the feature's outliers
    for col in df_processed.columns:
        if col != "patient id":
            if df_processed[col].isnull().all():
                df_processed[col] = avg_dict[col]
                full_hours_df[col] = avg_dict[col]
                all_null_list.append(col)
                continue
            outlier_perc_dict[col] = [outliers_perc_of_col(col, df_processed)]
    outliers_perc_df = pd.DataFrame(outlier_perc_dict).T
    outliers_perc_df.columns = ["Outliers %"]
    outliers_perc_df = outliers_perc_df.sort_values(by=['Outliers %'], ascending=False)
    columns = df_processed.columns
    for col in columns:
        if col != "patient id" and col not in all_null_list:
            median = full_hours_df[col].median()
            df_processed[col] = df_processed[col].fillna(median)
            avg = full_hours_df[col].mean()
            mode = full_hours_df[col].mode()[0]
            outlier_perc = outliers_perc_df.loc[col, 'Outliers %']
            if df_processed[col].nunique() == 2:
                df_processed[col] = df_processed[col].fillna(mode)
            else:
                if outlier_perc <= 5:
                    df_processed[col] = df_processed[col].fillna(avg)
                elif outlier_perc > 5:
                    df_processed[col] = df_processed[col].fillna(median)

    # data transformations
    condition_temp = (df_processed['Temp'] > 38) | (df_processed['Temp'] < 36)
    condition_HR = (df_processed['HR'] > 90)
    condition_Resp = (df_processed['Resp'] > 20) | (df_processed['PaCO2'] < 32)
    condition_wbc = (df_processed['WBC'] * 10 ** 3 > 12000) | (df_processed['WBC'] * 10 ** 3 < 4000)
    df_processed['SIRS'] = np.where(condition_temp, 1, 0) + \
                           np.where(condition_HR, 1, 0) + np.where(condition_Resp, 1, 0) + \
                           np.where(condition_wbc, 1, 0)
    df_processed['SIRS'] = np.where(df_processed['SIRS'] >= 2, 1, 0)

    df_processed["BUN_Creatinine_ratio"] = df_processed["BUN"] / df_processed["Creatinine"]
    df_processed["qSOFA"] = np.where((df_processed['SBP'] >= 100) & (df_processed['Resp'] >= 22), 1, 0)

    # Feature selection
    cols = ["ICULOS", "Lactate", "Temp", "SIRS", "PaCO2", "Unit1", "FiO2", "AST", "pH", "Hgb", "WBC", "BUN", "Calcium",
            "BaseExcess",
            "HospAdmTime", "patient id", "Hct", "HCO3", "HR", "MAP", "SaO2", "Alkalinephos", "Magnesium", "Potassium",
            "Bilirubin_total", "Phosphate", "O2Sat", "BUN_Creatinine_ratio", "PTT", "SBP", "SepsisLabel"]
    df_processed = df_processed[cols]
    # Aggregations - remain only one record for each patient
    groups = df_processed.groupby('patient id')
    # Initialize an empty DataFrame to store the results
    results = pd.DataFrame(columns=df_processed.columns)
    # Loop through the groups
    for name, group in groups:
        # Check if there is at least one row with label 1
        if (group['SepsisLabel'] == 1).any():
            # If there is, select the first row with label 1
            row = group[group['SepsisLabel'] == 1].iloc[0]
        else:
            # If there is no row with label 1, select the last row
            row = group.iloc[-1]

        # Add the selected row to the results DataFrame
        results.loc[name] = row
    ids = results["patient id"]
    to_model_df = results.apply(pd.to_numeric, errors='coerce')
    to_model_df["patient id"] = ids
    X = to_model_df.drop(['SepsisLabel','patient id'], axis=1)
    y = to_model_df["SepsisLabel"]

    return X, y, to_model_df


def predict(X_test, y_test, test_df):
    model = pickle.load(open("model.pkl", "rb"))
    y_pred = model.predict(X_test)
    test_df["prediction"] = y_pred
    test_df = test_df.rename(columns={"patient id": "id"})
    output = test_df[["id","prediction"]]
    output.to_csv("prediction.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='File path prediction')
    parser.add_argument('filepath', type=str, help='Path to the file for prediction')
    args = parser.parse_args()
    df = concat_all_patients(args.filepath)
    X_test, y_test, test_df = pre_process(df)
    predict(X_test, y_test, test_df)

