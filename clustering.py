import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer


def multiple_kmeans(data, n_clusters_range, metric_str='euclidean'):
    """Performs several KMeans and return the best results according to silhouette score

    :param data: data on which operations are performed
    :type data: pd.DataFrame
    :param n_clusters_range: iterable with all n_cluster parameters to be used for KMeans
    :type n_clusters_range: iterable
    :param metric_str: metric to be used in the KMeans. Default is 'euclidean'
    :type metric_str: str
    :return best_kmeans: best KMeans
    :rtype best_kmeans: sklearn.cluster.KMeans
    :return best_result: best clustering result
    :rtype best_result: sklearn.cluster.KMeans
    :return best_sil: best silhouette score
    :rtype best_sil: float
    """
    best_kmeans = None
    best_result = None
    best_sil = 0

    for n_clusters in n_clusters_range:
        cur_kmeans = KMeans(n_clusters=n_clusters, random_state=0, precompute_distances=True, n_jobs=-1)
        cur_result = cur_kmeans.fit_predict(data)
        cur_score = silhouette_score(data, cur_result, metric=metric_str)
        if cur_score > best_sil:
            best_kmeans = cur_kmeans
            best_result = cur_result
            best_sil = cur_score
    return best_kmeans, best_result, best_sil


def visualize_silhouettes(data, km_model, path_suffix=''):
    """Visualize clustering silhouettes and save it to a figure

    :param data: data on which clustering is performed
    :type data: pd.DataFrame
    :param km_model: KMeans model to evaluate
    :type km_model: sklearn.cluster.KMeans
    :param path_suffix: suffix to be used in path saving. Default is ''
    :type path_suffix: str
    """
    fig = plt.figure(figsize=(20, 40))
    model = SilhouetteVisualizer(km_model)
    model.fit(data)
    model.show()
    fig.savefig(os.path.join("./", path_suffix + "k_m_silhouettes_{}.png".format(km_model.n_clusters)))
    plt.close()


if __name__ == '__main__':
    import pandas as pd
    import os
    from src.utils.cronometer import Cronometer
    from src.utils.file_helper import load_data
    os.chdir("../")

    crono = Cronometer()

    n_neighbors = 50
    n_components = 10
    min_dist = 0.3

    train_reduced_df = pd.read_csv("../data/reduced/train_full_{}_{}_{}.csv".format(
        n_components, n_neighbors, min_dist))
    valid_reduced_df = pd.read_csv("../data/reduced/valid_full_{}_{}_{}.csv".format(
        n_components, n_neighbors, min_dist))
    crono.lap("loading")

    best_kmeans, best_result, best_sil = multiple_kmeans(train_reduced_df, n_clusters_range=[2, 4, 6, 8, 10, 20])
    visualize_silhouettes(train_reduced_df, best_kmeans)
    crono.lap("kmeans")

    """
    tsne_visualization(data=train_reduced_df, labels=best_result, n_clusters=best_kmeans.n_clusters, n_components=3)
    crono.lap("TSNE")

    reduced_all = train_reduced_df.append(valid_reduced_df, ignore_index=True)
    train_label = [1] * len(train_reduced_df) + [0] * len(valid_reduced_df)
    tsne_visualization(data=reduced_all, labels=train_label, n_clusters=2, n_components=3, path_suffix='train_valid')
    crono.lap("train vs valid")

    train_y = load_data("train_y")  # .loc[:MAX_ROWS]

    class_label = train_y.append(load_data("valid_y"), ignore_index=True).x.to_list()
    tsne_visualization(data=reduced_all, labels=class_label, n_clusters=2, n_components=3, path_suffix='classification')
    crono.lap("classification")
    """