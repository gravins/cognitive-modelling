from model import *
from tools import *
from evaluation_func import CONJ_DICT
from scipy.stats import zscore

if __name__ == '__main__':
    # Load datasets
    df_train = get_dataframe_from_excel("expt_data.xls", skip_from_head=15, out_col=[3], nrows=16, usecols="B:E")
    df_test = get_dataframe_from_excel("expt_data.xls", skip_from_head=35, out_col=[3, 4], nrows=30, usecols="B:F")

    # Remove word category from the target column
    df_train["y"] = df_train.apply(lambda row: row["y"].replace("category ", "") if "category" in row["y"] else row["y"].replace("categories ", ""), axis=1)
    df_test["y_0"] = df_test.apply(lambda row: row["y_0"].replace("category ", "") if "category" in row["y_0"] else row["y_0"].replace("categories ", ""), axis=1)

    x_train = df_train[df_train.columns[:3]]
    y_train = df_train[df_train.columns[-1]]
    x_test = df_test[df_test.columns[:3]]
    queries = df_test["y_0"]
    user_results = df_test[df_test.columns[-1:]]

    #############################
    #       Distance Model      #
    #############################

    # Create the distance model and fit
    dist_mod = DistanceModel(CONJ_DICT["avg"])
    dist_mod.fit(x_train, y_train)

    print("prediction over training set -- distance model")
    pred = dist_mod.predict(x_train, y_train)
    save_results(pred, y_train.values.flatten())

    print("prediction over test set -- distance model")
    pred = dist_mod.predict(x_test, queries)
    # Save zscore results
    save_results(zscore(pred), zscore(user_results.values.flatten()), path="distance_model_avg.csv")
