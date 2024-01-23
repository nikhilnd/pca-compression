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

    :return: average intercluster compression, average intracluster compression
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

def get_compression_matrix(data, pca_dim, reduce_dim=False, seq=False):
    """
    Return the compression matrix. 
    :param data: numpy array of shape (n_samples, n_features)
    data must be sorted by cluster and align with the order of cluster_sizes

    :return: compression matrix
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

    :return: compressibility matrix, avg intercluster compressibility, avg intracluster compressibility
    """

    C = get_compression_matrix(data, pca_dim, reduce_dim=reduce_dim, seq=seq)

    # Compute average intercluster and intracluster compression ratios
    avg_intercluster_compression, avg_intracluster_compression = get_average_compression(C, cluster_sizes, len(cluster_sizes))

    return C, avg_intercluster_compression, avg_intracluster_compression


def parse_h5ad(anndf, cluster_label_obs): 
    """
    Return the data, cluster_sizes, and cluster labels from an AnnData object.
    :param anndf: AnnData object
    :param cluster_label_obs: name of the cluster label in the AnnData object

    :return: data, cluster_sizes, cluster_labels
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

    :return: data, cluster_sizes, cluster_labels
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

    :return: ARI and NMI of kmeans clustering on data
    """
    kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(data)
    return adjusted_rand_score(labels, kmeans.labels_), adjusted_mutual_info_score(labels, kmeans.labels_)

def pca_kmeans(data, n_clusters, labels, pca_dim): 
    
    """
    Compute the NMI and ARI of a kmeans clustering on a dataset after PCA.
    :param data: numpy array of shape (n_samples, n_features)
    :param n_clusters: number of clusters
    :param labels: cluster labels
    :param pca_dim: dimensionality of PCA

    :return: ARI and NMI of kmeans clustering on data after PCA
    """

    data_pca = PCA(n_components=pca_dim, random_state=1).fit_transform(data)
    
    # Perform kmeans clustering on post-PCA data; repeat 5 times and take max
    arr = Parallel(n_jobs=5)(delayed(kmeans_nmi_ari)(data_pca, n_clusters, labels) for i in range(5))
    ari = np.max([x[0] for x in arr])
    nmi = np.max([x[1] for x in arr])

    return ari, nmi

def mad(X):

    """
    Compute the mean absolute deviation of the greatest and least 25% of X 
    :param X: data to compute variance for
    :param power: power
    :return: mean absolute deviation of the greatest and least 25% of X
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

def variance_list(C, reverse=False): 

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
    combined_var.sort(key=lambda x: x[0], reverse=reverse)
    return combined_var

def remove_pca_kmeans(data, cluster_sizes, labels, pca_dim, removal_rate=0.2, reduce_dim=False, C=None, method='compression', pca_before_removal=True):
    
    """
    Compute the NMI and ARI of a kmeans clustering on a dataset after removing likely outlier points and performing PCA.

    :param data: numpy array of shape (n_samples, n_features), must be sorted by cluster
    :param cluster_sizes: array of cluster sizes
    :param labels: cluster labels, must be sorted by cluster
    :param pca_dim: dimensionality of PCA
    :param removal_rate: percentage of points to remove
    :param reduce_dim: whether to reduce dimensionality before computing compressibility
    :param C: compressibility matrix, if already computed
    :param method: method of outlier removal. Options are 'compression', 'lof', 'ecod', and 'dist

    :return: ARI and NMI of kmeans clustering on data after outlier removal
    """

    n_clusters = len(cluster_sizes)

    data_pca = PCA(n_components=pca_dim, random_state=1).fit_transform(data)
    data = data_pca.copy() 

    # Array of (outlier metric, index in data) sorted by outlier metric in descending order (i.e. most likely outlier first)
    combined = None

    # Variance of compressibility
    if method == 'compression': 
        if C is None:
            C = get_compression_matrix(data, pca_dim, reduce_dim=reduce_dim, seq=True)
        combined = variance_list(C)
    # Local outlier factor
    elif method == 'lof': 
        combined = lof(data, 10)
    # ECOD
    elif method == 'ecod':
        combined = ecod(data)
    # Variance of post-PCA distances
    elif method == 'dist': 
        data_pca = PCA(n_components=pca_dim, random_state=1).fit_transform(data)
        combined = variance_list(get_distance_matrix(data_pca), True)
    # KNN distance
    elif method == 'knn': 
        combined = KNN_dist(data,20)
    else:
        raise ValueError('Invalid outlier removal method')

    # Remove points from data and labels
    num_to_remove = int(removal_rate * sum(cluster_sizes))
    mask = np.ones(len(data), dtype=bool)
    mask[[int(combined[i][1]) for i in range(num_to_remove)]] = False
    data_removed = data[mask, :]
    labels_removed = labels[mask]

    # Perform kmeans clustering on post-PCA data; repeat 5 times and take max

    # data_removed_pca = PCA(n_components=pca_dim, random_state=1).fit_transform(data_removed)
    # arr = Parallel(n_jobs=5)(delayed(kmeans_nmi_ari)(data_removed_pca, n_clusters, labels_removed) for i in range(5))
    arr = Parallel(n_jobs=5)(delayed(kmeans_nmi_ari)(data_removed, n_clusters, labels_removed) for i in range(5))

    ari_removed = np.max([x[0] for x in arr])
    nmi_removed = np.max([x[1] for x in arr])

    return ari_removed, nmi_removed

def improvement_graph(datasets,improvement,title,ylabel): 

    """
    Plot the improvement of a metric for a set of datasets.

    :param datasets: list of dataset names
    :param improvement: dictionary of improvement values for each dataset
    :param title: title of graph
    :param metric: metric to plot improvement of, either 'NMI' or 'ARI'
    """

    x = np.arange(len(datasets))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in improvement.items():
        # Use the offset to move the bar
        offset = width * multiplier

        # Plot the bar
        ax.bar(x + offset, measurement, width, label=attribute)
        multiplier += 1

    # Add text for labels, title and custom x-axis tick labels, etc.
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x + width, datasets, rotation=45)
    ax.legend(loc='upper left', ncols=3)
    min_val = min([y for x in improvement.values() for y in x])
    max_val = max([y for x in improvement.values() for y in x])
    ax.set_ylim(min(-0.005, min_val - (abs(min_val) * 0.15)), max(0.23, max_val + (abs(max_val) * 0.15)))

    plt.show()

def lof(PX, ng=20):

    """
    Calculate the LOF score for each point in PX and return the points sorted by LOF score.

    :param PX: numpy array of shape (n_samples, n_features)
    :param ng: number of neighbors to use for LOF
    :return: array of (LOF score, index in PX) sorted by LOF score in ascending order
    """   

    n=PX.shape[0]

    # # Calculate LOF scores
    # clf = LocalOutlierFactor(n_neighbors=ng)
    # clf.fit_predict(PX)
    # X_scores=clf.negative_outlier_factor_

    # # Combine LOF scores with indices
    # xaxis=[i for i in range(n)]
    # LOF_order=np.zeros((n,2))
    # LOF_order[:,0]=X_scores
    # LOF_order[:,1]=xaxis 

    # # Sort by LOF score
    # LOF_order=sorted(LOF_order, key=lambda x: x[0],reverse=False) 
    # LOF_order=np.array(LOF_order)

    # Calculate LOF scores
    clf = LocalOutlierFactor(n_neighbors=ng)
    clf.fit_predict(PX)
    X_scores=np.abs(np.array(clf.negative_outlier_factor_) + 1)

    # Combine LOF scores with indices
    xaxis=[i for i in range(n)]
    LOF_order=np.zeros((n,2))
    LOF_order[:,0]=X_scores
    LOF_order[:,1]=xaxis 

    # Sort by LOF score
    LOF_order=sorted(LOF_order, key=lambda x: x[0],reverse=True) 
    LOF_order=np.array(LOF_order)

    return LOF_order

def ecod(PX):

    """
    Calculate the ECOD score for each point in PX and return the points sorted by ECOD score.

    :param PX: numpy array of shape (n_samples, n_features)
    :return: array of (ECOD score, index in PX) sorted by ECOD score in ascending order
    """

    # Calculate ECOD scores
    clf = ECOD()
    clf.fit(PX)
    y_train_scores = clf.decision_scores_

    n=PX.shape[0]

    # Combine ECOD scores with indices
    xaxis=[i for i in range(n)]
    ECOD_order=np.zeros((n,2))
    ECOD_order[:,0]=y_train_scores
    ECOD_order[:,1]=xaxis 

    # Sort by ECOD score (reverse order, higher score is more likely to be outlier)
    ECOD_order=sorted(ECOD_order, key=lambda x: x[0],reverse=True)
    ECOD_order=np.array(ECOD_order)

    return ECOD_order

def KNN_dist(PX,kchoice):

    n=PX.shape[0]
    index = pynndescent.NNDescent(PX)
    index.prepare()
    kchoice1=kchoice+1
    neighbors = index.query(PX,k=kchoice1)
    indices = neighbors[0]
    knn_list=indices[:,1:]
    knn_list=np.array(knn_list)

    knn_order=np.zeros((n,2))

    for i in range(n):
        knn_order[i,0]=min([np.linalg.norm(PX[i,:]-PX[knn_list[i,j],:]) for j in range(kchoice)])
        knn_order[i,1]=i


    knn_order=sorted(knn_order, key=operator.itemgetter(0),reverse=True)
    knn_order=np.array(knn_order)

    return knn_order

def generate_points(x, d, n, k, noise_lower, noise_upper, p=0.3, method='bernoulli'): 
    """
    
    Generate a (n, d) array of points with k clusters.
    :param x: (k, d) array of cluster centers
    :param d: number of dimensions
    :param n: number of points
    :param k: number of clusters
    :param noise_lower: lower bound of noise (per point)
    :param noise_upper: upper bound of noise (per point). Variance of Gaussian noise. 
    :param p: probability of Bernoulli noise
    :param method: method of noise generation. Options are 'bernoulli', 'uniform', and 'gaussian'
    :return: (n, d) array of points

    """

    Y = np.zeros((0, d))
    for i in range(k): 
        for j in range(n[i]):    
            noise = None 
            if method == 'bernoulli':
                m = np.random.uniform(noise_lower, noise_upper)
                noise = np.random.choice([m, 0, -m], size=(1, d), p=[(1-p)/2., p, (1-p)/2.])
            elif method == 'uniform':
                noise = np.random.uniform(-noise_upper, noise_upper, size=(1, d))
            elif method == 'gaussian': 
                noise = np.random.normal(0, np.square(noise_upper), size=(1, d)) 
            else:
                raise ValueError('Invalid noise generation method')

            point = x[i] + noise
            # point = np.abs(point)
            Y = np.vstack((Y, point))
    
    return Y

def generate_doublet_outliers(o, cluster_sizes, Y, lower_weight=0.5, upper_weight=0.7): 
    """
    
    Generate a (o, d) array of outlier points and append them to Y. Outliers are generated by taking a weighted average of two points in different clusters to create doublets.
    :param o: number of outliers
    :param cluster_sizes: array of cluster sizes, including outlier "cluster"
    :param Y: (n, d) array of points
    :param lower_weight: lower bound of weight
    :param upper_weight: upper bound of weight
    :return: (n + o, d) array of points
    
    """

    for i in range(o): 
        clusters = np.random.choice(range(len(cluster_sizes) - 1), size=2, replace=False)
        weight = np.random.uniform(lower_weight, upper_weight)
        sample = [
            Y[np.random.randint(sum(cluster_sizes[:clusters[0]]), sum(cluster_sizes[:clusters[0]+1]))],
            Y[np.random.randint(sum(cluster_sizes[:clusters[1]]), sum(cluster_sizes[:clusters[1]+1]))]
        ]
        point = (sample[0] * weight) + (sample[1] * (1 - weight))
        Y = np.vstack((Y, point))
    
    return Y 

def generate_doublet_outliers_with_labels(o, cluster_sizes, Y, labels, lower_weight=0.5, upper_weight=0.7):

    """
    
    Generate a (o, d) array of outlier points and append them to Y. Outliers are generated by taking a weighted average of two points in different clusters to create doublets.
    :param o: number of outliers
    :param cluster_sizes: array of cluster sizes, including outlier "cluster"
    :param Y: (n, d) array of points
    :param labels: (n,) array of cluster labels
    :param lower_weight: lower bound of weight
    :param upper_weight: upper bound of weight
    :return: (n + o, d) array of points
    
    """

    for i in range(o): 
        clusters = np.random.choice(range(len(cluster_sizes) - 1), size=2, replace=False)
        weight = np.random.uniform(lower_weight, upper_weight)
        sample = [
            Y[np.random.randint(sum(cluster_sizes[:clusters[0]]), sum(cluster_sizes[:clusters[0]+1]))],
            Y[np.random.randint(sum(cluster_sizes[:clusters[1]]), sum(cluster_sizes[:clusters[1]+1]))]
        ]
        point = (sample[0] * weight) + (sample[1] * (1 - weight))
        Y = np.vstack((Y, point))
        if weight <= 0.5: 
            labels = np.append(labels, clusters[0])
        else: 
            labels = np.append(labels, clusters[1])
    
    return Y, labels

def generate_doublet_outliers_centers(o, x, Y, noise=1.5, lower_weight=0.5, upper_weight=0.7): 
    """
    
    Generate a (o, d) array of outlier points and append them to Y. Outliers are generated by taking a weighted average of two points in different clusters to create doublets.
    :param o: number of outliers
    :param cluster_sizes: array of cluster sizes
    :param x: (k, d) array of cluster centers 
    :param lower_weight: lower bound of weight
    :param upper_weight: upper bound of weight
    :return: (n + o, d) array of points
    
    """ 

    for i in range(o): 
        clusters = np.random.choice(range(len(x) - 1), size=2, replace=False)
        weight = np.random.uniform(lower_weight, upper_weight)
        sample = [
            x[clusters[0]],
            x[clusters[1]]
        ]
        point = (sample[0] * weight) + (sample[1] * (1 - weight))
        noise = np.random.uniform(-noise, noise, size=(1, len(x[0])))
        Y = np.vstack((Y, point))
    
    return Y 

def generate_outliers(o, x, sigma, c, Y, method='uniform'): 
    """
    
    Generate a (o, d) array of outlier points and append them to Y. Outliers are generated by adding Gaussian noise to the points in Y.
    :param o: number of outliers
    :param x: (k, d) array of cluster centers
    :param sigma: standard deviation of Gaussian noise
    :param c: scaling factor of noise 
    :param Y: (n, d) array of points
    :param method: method of noise generation. Options are 'gaussian' and 'uniform'
    :return: (n + o, d) array of points
    
    """

    for i in range(o): 
        center = np.random.randint(0, len(x))

        noise = None
        if method == 'gaussian':
            noise = np.random.normal(0, np.square(sigma) * c, size=(1, len(x[0])))
            # noise = np.random.standard_cauchy(size=(1, len(x[0]))) * np.square(sigma) * c
            # noise = np.random.laplace(0, np.square(sigma) * c, size=(1, len(x[0])))
        elif method == 'uniform':
            noise = np.random.uniform(-sigma * np.sqrt(c), sigma * np.sqrt(c), size=(1, len(x[0])))
        else:
            raise ValueError('Invalid noise generation method')

        point = x[center] + noise
        # point = np.abs(point)
        Y = np.vstack((Y, point))
    
    return Y