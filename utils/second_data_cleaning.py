import pandas as pd


def transform_categorical_feature(df, column_name, column_prefix=""):

    df1 = df[column_name].str.get_dummies()
    if column_prefix != "":
        df1.columns = ['is_type_' + col for col in df1.columns]

    new_df = pd.concat([df, df1], axis=1)

    return new_df
