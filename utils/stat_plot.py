import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("assets/houses.csv")

# Showing pairplots
new_df = df[["price", "area", "room_number"]]
y_column = ["price"]
x_column = list(new_df.columns)
x_column.remove(x_column[0])
g = sns.pairplot(new_df, x_vars=x_column, y_vars=y_column)
g.fig.set_size_inches(10, 10)

# Plotting correlation matrix
corr = df.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(corr, annot=True, cmap="coolwarm", square=True, fmt=".2f")
plt.show()
