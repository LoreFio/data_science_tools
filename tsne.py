import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def tsne_visualization(data, labels, n_clusters, n_components=3, path_suffix=''):
    """Computes TSNE visualization in 2D or 3D

    :param data: data on which operations are performed
    :type data: pd.DataFrame
    :param labels: iterable containing labels of points, must have same length than data
    :type labels: iterable
    :param n_clusters: number of clusters
    :type n_clusters: int
    :param n_components: number of components of the TSNE
    :type n_components: int
    :param path_suffix: suffix to be used in path saving. Default is ''
    :type path_suffix: str
    """
    data_tsne = TSNE(n_components=n_components).fit_transform(data)
    path = os.path.join("./", "{}tsne_{}_c_{}.png".format(path_suffix, n_components, n_clusters))

    if n_components == 2:
        fig = plt.figure(figsize=(20, 40))
        plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap='viridis', marker='o', s=30)
        plt.xlabel('TSNE_2 1')
        plt.ylabel('TSNE_2 2')
        fig.savefig(path)
    else:
        scatter3d(data=data_tsne, path=path, labels=labels)


def scatter3d(data, path, labels):
    """Plot a 3D scatter plot and saves it

    :param data: data to show
    :type data: pd.DataFrame
    :param path: path to save image
    :type path: str
    :param labels: iterable containing labels of points, must have same length than data
    :type labels: iterable
    """
    fig = plt.figure(figsize=(20, 40))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis', marker='o', s=30)
    ax.set_xlabel('TSNE_3 1')
    ax.set_ylabel('TSNE_3 2')
    ax.set_zlabel('TSNE_3 3')
    fig.savefig(path)
