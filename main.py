import pandas as pd

from utils.data_cleaners import clean_df, remove_outliers, transform_categorical_feature


df = pd.read_csv("assets/houses.csv")

df_cleaned = clean_df(df)
df_outliers_removed = remove_outliers(df_cleaned)
df_features_transformed = transform_categorical_feature(df_outliers_removed, "building_condition")

df_features_transformed