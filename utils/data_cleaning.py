# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# %%
df = pd.read_csv('../assets/houses.csv')
df


# %%
df.count()


# %%
df.shape


# %%
# removing any row that contains empty area information because it's a key feature
# also removing rows with NaN in facade_count because there are a lot
df = df.dropna(subset=['area', 'facade_count'])


# %%
df.shape


# %%
# removes about 1500 duplicates that are same in these 4 features
df = df.drop_duplicates(subset=['location', 'price', 'area', 'building_condition'])


# %%
df.shape


# %%
df.count()


# %%
# initial cleaning on the columns that have a lot of missing values
df = df.drop(columns=['terrace_area', 'garden_area', 'terrace_area', 'land_surface'])


# %%
# these columns are not really necessary for estates for sale
df = df.drop(columns=['furnished'])


# %%
# basic correlation plot
# plt.figure(figsize = (16,5))
# sns.heatmap(data=df.corr(), annot=True, cmap='BuPu')


# %%
# checking if we have still NaN (except building_condition)
df.count()


# %%
# interquartile range in order to calculate outliers

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# %%
# showing the outliers
print(df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))


# %%
# explanation: https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba

df_outliers_removed = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
df_outliers_removed.shape


# %%



