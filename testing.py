from plot_functions import *

# Generate some random data for demonstration
latent_vectors = np.random.randn(10000, 10)

mydir ='/scratch/echickle/foo/'
plot_cluster(latent_vectors, mydir, cluster_method='kmeans')

