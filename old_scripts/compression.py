import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

# Number of clusters
k = 4
# Points per cluster
n = 5
# Number of dimensions
d = 100

# Create empty numpy array
X = np.empty((0, d))

# Create cluster centers
for i in range(k):
    # Random center
    center = np.random.normal(i, 1, d)
    # Append points to X
    X = np.vstack((X, center))

# Compute Euclidean distance matrix between each cluster center
D = np.zeros((k, k))
for i in range(k):
    for j in range(k):
        D[i, j] = np.linalg.norm(X[i] - X[j])

print("Center distances are",D)

sigma=2
Y = np.empty((0, d))
for i in range(k): 
    for j in range(n): 
        # Random point
        point = X[i] + np.random.normal(0, sigma, d)
        # Append point to Y
        Y = np.vstack((Y, point))

print("Average noise perturbation is",sigma*np.sqrt(d))


# print(Y)

# Compute Euclidean distance matrix between each point
D = np.zeros((n * k, n * k))
for i in tqdm(range(n * k)):
    for j in range(n * k):
        D[i, j] = np.linalg.norm(Y[i] - Y[j])

print("Pre PCA distances")
#print(D)

pca = PCA(n_components=(k - 1))
pca.fit(Y)
Y_pca = pca.transform(Y)

D_pca = np.zeros((n * k, n * k))
for i in tqdm(range(n * k)):
    for j in range(n * k):
        D_pca[i, j] = np.linalg.norm(Y_pca[i] - Y_pca[j])

# Compute compression matrix
C = D / D_pca
#print(C)

avg_intracluster_compression = np.zeros(k)
avg_intercluster_compression = np.zeros(k)
for i in range(k):
    for j in range(i * n, (n * (i + 1)) - 1): 
        for l in range(i * n, (n * (i + 1)) - 1): 
            if j >= l: 
                continue
            avg_intracluster_compression[i] += C[j, l]

        t1=range(i * n, (n * (i + 1)) - 1)
        t2=[m for m in range(k * n)]
        t3=set(t2).difference(set(t1))
        
    
        for l in list(t3):
            avg_intercluster_compression[i] += C[j, l]


# n choose 2
avg_intracluster_compression /= (n * (n - 1)) / 2
avg_intercluster_compression /= (n * ((k - 1) * n))
print(avg_intracluster_compression)
print(avg_intercluster_compression)

