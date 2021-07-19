import pandas as pd


# initial cleaning
def clean_df(df):
    # removing any row that contains empty area and price information because it's a key feature
    # also removing rows with NaN in facade_count because there are a lot
    df = df.dropna(subset=['area', 'facade_count', 'price'])

    # removes about 1500 duplicates that are same in these 4 features
    df = df.drop_duplicates(subset=['location', 'price', 'area', 'building_condition'])

    # initial cleaning on the columns that have a lot of missing values
    # and the columns are not really necessary for estates for sale such as "furnished"
    df = df.drop(columns=['terrace_area', 'garden_area', 'terrace_area', 'land_surface', 'furnished'])

    return df


def remove_outliers(df):
    # interquartile range in order to calculate outliers
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    # uncomment for printing the outliers
    # print(df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))

    df_outliers_removed = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    # explanation: https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba

    return df_outliers_removed

def transform_categorical_feature(df, column_name, column_prefix=""):
    # transforming textual data to categorical features
    df1 = df[column_name].str.get_dummies()
    if column_prefix != "":
        df1.columns = ['is_type_' + col for col in df1.columns]

    new_df = pd.concat([df, df1], axis=1)

    # we don't need transformed column anymore
    new_df = new_df.drop(columns=[column_name])

    return new_df
