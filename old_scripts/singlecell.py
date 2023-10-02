import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

# Read data
cluster_identities = np.load('data/iden-tasic.npy')
ss_data = np.load('data/npdata-tasic.npy')

# Transpose ss_data
ss_data = np.transpose(ss_data)

ss_data=np.log2(1+ss_data)

pca = PCA(n_components=int(max(cluster_identities)))
pca.fit(ss_data)

ss_data_pca = pca.transform(ss_data)

d = ss_data_pca.shape[1]
k = int(max(cluster_identities)) + 1

# d = 24057
# Total data points = 1809

# Create empty numpy array
cluster_groupings = [[] for i in range(int(max(cluster_identities)) + 1)]

# Populate cluster_groupings
for i, v in enumerate(cluster_identities):
    cluster_groupings[int(v)].append(ss_data[i])

for i in range(len(cluster_groupings)):
    cluster_groupings[i] = np.array(cluster_groupings[i])
    print(cluster_groupings[i].shape)

# Compute average intracluster euclidean distance
avg_intracluster_euclidean_distance = np.zeros(len(cluster_groupings))
print(len(cluster_groupings[0][0]))
for i in tqdm(range(len(cluster_groupings))):
    for j in range(len(cluster_groupings[i])):
        for l in range(j + 1, len(cluster_groupings[i])):
            avg_intracluster_euclidean_distance[i] += np.linalg.norm(cluster_groupings[i][j] - cluster_groupings[i][l])
    
    avg_intracluster_euclidean_distance[i] /= ((len(cluster_groupings[i]) * (len(cluster_groupings[i]) - 1)) / 2)

print(avg_intracluster_euclidean_distance)

# Compute euclidean distances for ss_data matrix
D = np.zeros((ss_data.shape[0], ss_data.shape[0]))
for i in tqdm(range(ss_data.shape[0])):
    for j in range(ss_data.shape[0]):
        D[i, j] = np.linalg.norm(ss_data[i] - ss_data[j])

# Compute euclidean distances for ss_data_pca matrix
D_pca = np.zeros((ss_data_pca.shape[0], ss_data_pca.shape[0]))
for i in tqdm(range(ss_data_pca.shape[0])):
    for j in range(ss_data_pca.shape[0]):
        D_pca[i, j] = np.linalg.norm(ss_data_pca[i] - ss_data_pca[j])

# Compute compression matrix
C = D / D_pca

avg_intracluster_compression = np.zeros(k)
avg_intercluster_compression = np.zeros(k)
for i in range(k):
    for j in range(i * len(cluster_groupings[i]), (len(cluster_groupings[i]) * (i + 1)) - 1): 
        for l in range(i * len(cluster_groupings[i]), (len(cluster_groupings[i]) * (i + 1)) - 1): 
            avg_intracluster_compression[i] += C[j, l]
    
    avg_intracluster_compression[i] /= ((len(cluster_groupings[i]) * (len(cluster_groupings[i]) - 1)) / 2)

    for j in range(i * len(cluster_groupings[i]), (len(cluster_groupings[i]) * (i + 1)) - 1): 
        for l in range((i + 1) * len(cluster_groupings[i]), len(cluster_groupings[i]) * (i + 2)): 
            avg_intercluster_compression[i] += C[j, l]
    
    avg_intercluster_compression[i] /= (len(cluster_groupings[i]) * len(cluster_groupings[i + 1]))

print(avg_intracluster_compression)
print(avg_intercluster_compression)