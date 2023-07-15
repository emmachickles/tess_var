from plot_functions import *
import numpy as np

mydir = '/Users/emma/Desktop/230708/'
# mydir ='/scratch/echickle/foo/'

# Generate some random data for demonstration
n_lc = 1000
latent_vectors = np.random.randn(n_lc, 10)
light_curves = np.random.randn(n_lc, 300)
errors = np.ones((n_lc,300))*0.1
cluster_labels = np.random.choice(np.arange(35), n_lc)
latent_tsne_2d = np.random.randn(n_lc,2)
latent_tsne_3d = np.random.randn(n_lc,3)

# Plot
# plot_cluster(latent_vectors, latent_tsne_2d, cluster_labels, mydir)
# movie_cluster( latent_vectors, latent_tsne_3d, cluster_labels, mydir)
# movie_light_curves(light_curves, errors, mydir)
movie_cluster_insets(light_curves, errors, latent_vectors, latent_tsne_3d, cluster_labels, mydir)

