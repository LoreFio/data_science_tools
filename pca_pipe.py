import pandas as pd
from sklearn.decomposition import PCA


if __name__ == '__main__':
    train_x_cleaned = pd.read_csv("train_x.csv")
    valid_x_cleaned = pd.read_csv("valid_x.csv")
    train_y = pd.read_csv("train_y")
    valid_y = pd.read_csv("valid_y")

    for col in train_x_cleaned.columns:
        if train_x_cleaned[col].abs().max() > 250:
            print(train_x_cleaned[col].describe(percentiles=[0.1, 0.9]))

    n_components = 50
    reducer = PCA(n_components=n_components, random_state=None)
    train_reduced = reducer.fit_transform(X=train_x_cleaned, y=train_y)
    valid_reduced = reducer.transform(X=valid_x_cleaned)
    print("explained_variance_ratio_", reducer.explained_variance_ratio_)
