import pandas as pd


df = pd.read_csv("assets/houses.csv")
new_df = df[["price", "area", "kitchen_equipped", "room_number", "furnished"]]

df = transform_categorical_feature(df, "type", "is_type_")


# sns.pairplot(new_df)
# plt.show()

# X = df
# y = X.pop("price")
# discrete_features = ["type", "subtype", "location", "kitchen_equipped", "furnished", "fireplace", "terrace", "garden", "swimming_pool", "building_condition"]




# corr = df.corr()

# top_corr_features = corr.index
# plt.figure(figsize=(20,20))
# #plot heat map
# g=sns.heatmap(corr,annot=True,cmap="coolwarm", square=True, fmt=".2f")
# plt.show()
