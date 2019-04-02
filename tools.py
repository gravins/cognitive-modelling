import pandas as pd


def get_dataframe_from_excel(path, index=None, skip_from_head=None, out_col=None, nrows=None, usecols=None, col_name_row=-1):
    """
    Function that load DataFrame from excel file.
    :param path: path of the file to read
    :param index: index column used as index in the dataframe
    :param skip_from_head: number of rows to skip from the head of the csv
    :param out_col: list of columns that identify the target
    :param nrows: number of rows of file to read
    :param usecols: subset of the columns to use
    :param col_name_row: int , row index of the columns' names.
                        col_name_row == -1 means that no labels exist for the columns, so they will be created
    :return: DataFrame from excel file
    """

    df = pd.read_excel(path, index_col=index, skiprows=skip_from_head, header=col_name_row, usecols=usecols, nrows=nrows)

    if col_name_row < 0:
        # Create column labels
        col_name = ["x_" + str(i) for i in range(len(df.columns))]
        if out_col is not None:
            if len(out_col) == 1:
                col_name[out_col[0]] = "y"
            else:
                for count, i in enumerate(out_col):
                    col_name[i] = "y_" + str(count)
        df.columns = col_name

    return df


def save_results(y_pred, y_true, path=None):
    """
    Save results into csv and plot the scatter plot with y_true over y-axis and y_pred over x-axis if path is not None
    and print the resulting csv in a pretty way.
    :param y_pred: model prediction
    :param y_true: real target
    :param path: path of csv file
    """
    df = pd.DataFrame({"Prediction": y_pred, "True Value": y_true})
    print(df)
    if path:
        df.to_csv(path, float_format='%g')
        p = df.plot.scatter("Prediction", "True Value")
        fig = p.get_figure()
        path = path.replace("csv", "png")
        fig.savefig("plot_"+path)
