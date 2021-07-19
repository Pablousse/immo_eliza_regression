# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


df = pd.read_csv('../assets/houses.csv')

# removes about 1500 duplicates
df = df.drop_duplicates(subset=['location', 'price', 'area', 'building_condition'])

# removing any row that contains empty area information because it's a key feature
df = df.dropna(subset=['area'])

# initial cleaning on the columns that have a lot of missing values
df = df.drop(columns=['terrace_area', 'garden_area', 'terrace_area'])

plt.figure(figsize = (16,5))

sns.heatmap(data=df.corr(), annot=True, cmap='BuPu')





