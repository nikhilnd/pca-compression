from numba import jit
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import scanpy
import pynndescent
from joblib import Parallel, delayed
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import operator
from pyod.models.ecod import ECOD
from matplotlib import pyplot as plt

@jit(nopython=True)
def get_distance_matrix(X):
    """
    Compute the distance matrix for a set of points.
    :param X: numpy array of shape (n_samples, n_features)
    :return: numpy array of shape (n_samples, n_samples)
    """
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.linalg.norm(X[i] - X[j])
    return D

def get_distance_matrix_seq(X):
    """
    Compute the distance matrix for a set of points. Non jit version.
    :param X: numpy array of shape (n_samples, n_features)
    :return: numpy array of shape (n_samples, n_samples)
    """
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.linalg.norm(X[i] - X[j])
    return D

def get_average_compression(C, cluster_sizes, k): 
    """
    Compute the average intracluster and intercluster compression for a set of points.
    :param C: numpy array of pairwise compression ratios (n_samples, n_samples). Must be grouped by cluster.
    :param cluster_sizes: array of cluster sizes. Order must match grouping of C.
    :param k: number of clusters 
    """

    # Create empty arrays of size k to store intra/inter cluster compression for each cluster
    avg_intracluster_compression = np.zeros(k)
    avg_intercluster_compression = np.zeros(k)

    for i in range(k):

        # Index of first point in cluster i
        before = sum(cluster_sizes[:i])

        t1 = range(before, before + cluster_sizes[i])
        t2 = range(C.shape[0])
        # Set of points (indices) not in cluster i
        t3 = set(t2).difference(set(t1))

        for j in range(before, before + cluster_sizes[i]): 
            for l in range(before, before + cluster_sizes[i]):
                if j == l: 
                    continue 
                # Sum up all intracluster compression ratios for cluster i
                avg_intracluster_compression[i] += C[j, l]

            for l in list(t3):
                # Sum up all intercluster compression ratios for cluster i
                avg_intercluster_compression[i] += C[j, l]

    for i in range(k):
        # Divide by number of pairs of points in cluster i
        avg_intracluster_compression[i] /= (cluster_sizes[i] * (cluster_sizes[i] - 1))

        # Divide by number of pairs of points s.t. one point is in cluster i and the other is not
        avg_intercluster_compression[i] /= (cluster_sizes[i] * (len(C) - cluster_sizes[i]))
    
    return avg_intercluster_compression, avg_intracluster_compression

def get_compression_matrix(data, cluster_sizes, pca_dim, reduce_dim=False, seq=False):
    """
    Return the compression matrix. 
    :param data: numpy array of shape (n_samples, n_features)
    :param cluster_sizes: array of cluster sizes (including outlier "cluster")
    data must be sorted by cluster and align with the order of cluster_sizes
    """

    pre_data = data
    # For very large datasets, reduce dimensionality to speed up computation
    if reduce_dim: 
        red_dim_n = min(1000, data.shape[0])
        pca = PCA(n_components=red_dim_n)
        pre_data = pca.fit_transform(data)
    
    # Compute pre-PCA distance matrix
    if seq: 
        D_pre = get_distance_matrix_seq(pre_data)
    else:
        D_pre = get_distance_matrix(pre_data)

    pca = PCA(n_components=pca_dim)
    post_data = pca.fit_transform(data)
    # Compute post-PCA distance matrix
    if seq: 
        D = get_distance_matrix_seq(post_data)
    else:
        D = get_distance_matrix(post_data)

    # Compute pairwise compression ratios
    C = D_pre / D

    return C

def get_compressibility(data, cluster_sizes, pca_dim, reduce_dim=False, seq=False): 
    """
    Return the compressibility matrix, avg inter, and avg intra cluster compressibility. 
    :param data: numpy array of shape (n_samples, n_features)
    :param cluster_sizes: array of cluster sizes (including outlier "cluster")
    data must be sorted by cluster and align with the order of cluster_sizes
    """

    C = get_compression_matrix(data, cluster_sizes, pca_dim, reduce_dim=reduce_dim, seq=seq)

    # Compute average intercluster and intracluster compression ratios
    avg_intercluster_compression, avg_intracluster_compression = get_average_compression(C, cluster_sizes, len(cluster_sizes))

    return C, avg_intercluster_compression, avg_intracluster_compression


def parse_h5ad(anndf, cluster_label_obs): 
    """
    Return the data, cluster_sizes, and cluster labels from an AnnData object.
    :param anndf: AnnData object
    :param cluster_label_obs: name of the cluster label in the AnnData object
    """
    # Get data and cluster labels
    data = anndf.to_df()
    cluster_labels = anndf.obs[cluster_label_obs].values

    # Sort data by cluster label
    combined = [(cluster_labels.codes[i], data.iloc[i]) for i in range(len(data))]
    combined.sort(key=lambda x: x[0])

    # Separate data
    data = np.array([x[1] for x in combined])

    # Get cluster sizes
    cluster_sizes = [0] * len(cluster_labels.categories)
    for i in range(len(combined)): 
        cluster_sizes[combined[i][0]] += 1

    # Separate cluster labels
    cluster_labels = np.array([x[0] for x in combined])

    return data, cluster_sizes, cluster_labels

def initiate(**kwargs):

    """
    Initialize the data, cluster sizes, and cluster labels for a dataset from a list of dataset names and paths. 
    :param kwargs: 
    dsname: list of dataset names
    dspath: list of dataset paths
    """

    # Get args from kwargs
    fix_ch= int(kwargs.get('fix_ch', -1))
    dsname = kwargs.get('dsname', [])
    dspath = kwargs.get('dspath', [])

    if(fix_ch==-1):
        fix_ch=int(input([(i,dsname[i]) for i in range(len(dsname))]))

    # Read and log normalize data
    sce_data = scanpy.read_h5ad(dspath[fix_ch])
    data, cs, labels = parse_h5ad(sce_data, 'phenoid')
    data= np.log1p(data)

    return data, cs, labels

def kmeans_nmi_ari(data, n_clusters, labels): 
    """
    Compute the NMI and ARI of a kmeans clustering on a dataset.
    :param data: numpy array of shape (n_samples, n_features)
    :param n_clusters: number of clusters
    :param labels: cluster labels
    """
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto').fit(data)
    return adjusted_rand_score(labels, kmeans.labels_), adjusted_mutual_info_score(labels, kmeans.labels_)

def pca_kmeans(data, n_clusters, labels, pca_dim): 
    
    """
    Compute the NMI and ARI of a kmeans clustering on a dataset after PCA.
    :param data: numpy array of shape (n_samples, n_features)
    :param n_clusters: number of clusters
    :param labels: cluster labels
    """

    data_pca = PCA(n_components=pca_dim, random_state=1).fit_transform(data)
    # Perform kmeans clustering on PCA'd data; repeat 5 times and take median
    arr = Parallel(n_jobs=5)(delayed(kmeans_nmi_ari)(data_pca, n_clusters, labels) for i in range(5))
    ari = np.median([x[0] for x in arr])
    nmi = np.median([x[1] for x in arr])
    return ari, nmi

def mad(X):

    """
    Compute the mean absolute deviation of the greatest and least 25% of X 
    :param X: data to compute variance for
    :param power: power
    """

    x1 = sorted(X)
    x1 = x1[:len(X)//4] + x1[-len(X)//4:] 
    X = x1
    sum = 0
    avg = np.mean(X)
    for i in range(len(X)): 
        sum += abs(X[i] - avg)
    sum /= len(X)
    return sum

def variance_list(C): 

    """
    Compute the variance of compressibility for each point in C
    :param C: compressibility matrix 
    :return: Array of (variance, index in data) sorted by variance in asscending order
    """

    np.nan_to_num(C, copy=False, nan=0.0)
    comp_var = [0] * len(C)
    for i in range(len(C)): 
        comp_var[i] = np.var(C[i], where=C[i] != 0)
    combined_var = [(comp_var[i], i) for i in range(len(C))]
    combined_var.sort(key=lambda x: x[0])
    return combined_var

def remove_pca_kmeans(data, cluster_sizes, labels, pca_dim, removal_rate=0.2, reduce_dim=False, C=None, method='compression'):
    
    n_clusters = len(cluster_sizes)

    combined = None

    if method == 'compression': 
        if C is None:
            C, _, __ = get_compressibility(data, cluster_sizes, pca_dim, reduce_dim=reduce_dim)
        combined = variance_list(C)

    # print(data)
    # print(combined)
    num_to_remove = int(removal_rate * sum(cluster_sizes))
    mask = np.ones(len(data), dtype=bool)
    mask[[combined[i][1] for i in range(num_to_remove)]] = False
    data_removed = data[mask, :]
    labels_removed = labels[mask]

    # data_removed = np.delete(data, [combined[i][1] for i in range(num_to_remove)], axis=0)
    # labels_removed = np.delete(labels, [combined[i][1] for i in range(num_to_remove)], axis=0)
    # print(data_removed)

    data_removed_pca = PCA(n_components=pca_dim, random_state=1).fit_transform(data_removed)
    arr = Parallel(n_jobs=5)(delayed(kmeans_nmi_ari)(data_removed_pca, n_clusters, labels_removed) for i in range(5))
    # print(arr)
    ari_removed = np.median([x[0] for x in arr])
    nmi_removed = np.median([x[1] for x in arr])
    return ari_removed, nmi_removed

def improvement_graph(datasets,improvement,title,metric="NMI"): 
    x = np.arange(len(datasets))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in improvement.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title(title)
    ax.set_ylabel(f'{metric} Improvement')
    ax.set_xticks(x + width, datasets, rotation=45)
    ax.legend(loc='upper left', ncols=3)
    min_val = min([y for x in improvement.values() for y in x])
    max_val = max([y for x in improvement.values() for y in x])
    ax.set_ylim(min(-0.005, min_val - 0.005), max(0.23, max_val + 0.005))
    # ax.set_ylim(-0.005, 0.23)

    plt.show()