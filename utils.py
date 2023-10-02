from numba import jit
import numpy as np
from sklearn.decomposition import PCA
import scanpy as sc

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
    :param C: numpy array of pairwise compression ratios (n_samples, n_samples)
    :param cluster_sizes: array of cluster sizes
    :param k: number of clusters (including outlier "cluster")
    """
    avg_intracluster_compression = np.zeros(k)
    avg_intercluster_compression = np.zeros(k)
    for i in range(k):
        before = sum(cluster_sizes[:i])

        for j in range(before, before + cluster_sizes[i]): 
            for l in range(before, before + cluster_sizes[i]):
                if j == l: 
                    continue 
                avg_intracluster_compression[i] += C[j, l]
        
            t1 = range(before, before + cluster_sizes[i])
            t2 = range(C.shape[0])
            t3 = set(t2).difference(set(t1))

            for l in list(t3):
                avg_intercluster_compression[i] += C[j, l]

    for i in range(k):
        avg_intracluster_compression[i] /= (cluster_sizes[i] * (cluster_sizes[i] - 1))
        avg_intercluster_compression[i] /= (cluster_sizes[i] * (len(C) - cluster_sizes[i]))
    
    return avg_intercluster_compression, avg_intracluster_compression

def sample_spherical(npoints, ndim=3, r_min=1, r_max=2):
    """
    Sample points on an ndim unit sphere.
    :param npoints: number of points to sample
    :param ndim: dimension of the sphere
    :param r_min: minimum radius
    :param r_max: maximum radius
    :return: numpy array of shape (npoints, ndim)
    """
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    # vec *= np.random.randint(r_min, r_max)
    vec *= np.random.uniform(r_min, r_max)
    return vec.T

def sample_normal(npoints, ndim=3, mu=0, sigma=1):
    """
    Sample points from a normal distribution.
    :param npoints: number of points to sample
    :param ndim: dimension of the space
    :param mu: mean of the distribution
    :param sigma: standard deviation of the distribution
    :return: numpy array of shape (npoints, ndim)
    """
    return np.random.normal(mu, sigma, (npoints, ndim))

def compression_grouping(C, cluster_sizes): 
    """
    Return flattened, sorted array of all pairwise compression ratios grouped by same vs. different cluster.
    :param C: numpy array of pairwise compression ratios (n_samples, n_samples)
    :param cluster_sizes: array of cluster sizes 
    """

    size = (len(C) * (len(C) - 1)) / 2
    res = np.zeros((int(size), 2))
    idx = 0 
    curr_cluster = 0 
    curr_cluster_start = 0
    curr_cluster_end = cluster_sizes[0] - 1

    for i in range(len(C)): 
        if i > curr_cluster_end:
            curr_cluster += 1
            curr_cluster_start = curr_cluster_end + 1
            curr_cluster_end = curr_cluster_start + cluster_sizes[curr_cluster] - 1
        
        for j in range(i + 1, len(C)):
            diff_clusters = curr_cluster + 1 if j <= curr_cluster_end else 0
            res[idx] = [C[i, j], diff_clusters]
            idx += 1
   
    return res[np.argsort(res[:, 0])]

def get_compressibility(data, cluster_sizes, pca_dim, reduce_dim=False): 
    """
    Return the compressibility matrix, avg inter, and avg intra cluster compressibility. 
    :param data: numpy array of shape (n_samples, n_features)
    :param cluster_sizes: array of cluster sizes (including outlier "cluster")
    data must be sorted by cluster and align with the order of cluster_sizes
    """

    pre_data = data
    if reduce_dim: 
        pca = PCA(n_components=1000)
        pre_data = pca.fit_transform(data)
    
    D_pre = get_distance_matrix(pre_data)

    pca = PCA(n_components=pca_dim)
    post_data = pca.fit_transform(data)
    D = get_distance_matrix(post_data)

    C = D_pre / D

    avg_intercluster_compression, avg_intracluster_compression = get_average_compression(C, cluster_sizes, len(cluster_sizes))

    return C, avg_intercluster_compression, avg_intracluster_compression

def parse_h5ad(anndf, cluster_label_obs): 
    """
    Return the data, cluster_sizes, and cluster labels from an AnnData object.
    :param anndf: AnnData object
    :param cluster_label_obs: name of the cluster label in the AnnData object
    """
    data = anndf.to_df()
    cluster_labels = anndf.obs[cluster_label_obs].values

    combined = [(cluster_labels.codes[i], data.iloc[i]) for i in range(len(data))]
    combined.sort(key=lambda x: x[0])

    data = np.array([x[1] for x in combined])

    cluster_sizes = [0] * len(cluster_labels.categories)
    for i in range(len(combined)): 
        cluster_sizes[combined[i][0]] += 1

    cluster_labels = np.array([x[0] for x in combined])

    return data, cluster_sizes, cluster_labels