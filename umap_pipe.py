import pandas as pd
import umap


if __name__ == '__main__':
    train_x_cleaned = pd.read_csv("train_x.csv")
    valid_x_cleaned = pd.read_csv("valid_x.csv")
    train_y = pd.read_csv("train_y")
    valid_y = pd.read_csv("valid_y")

    n_neighbors = 100
    n_components = 50
    min_dist = 0.3
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist, metric='euclidean',
                        low_memory=True)

    train_reduced = reducer.fit_transform(X=train_x_cleaned, y=train_y)
    valid_reduced = reducer.transform(X=valid_x_cleaned)