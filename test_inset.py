import numpy as np
from plot_functions import plot_tsne_inset

# Generate fake latent vectors (you can replace this with your actual data)
np.random.seed(42)
num_data_points = 100
latent_vectors = np.random.rand(num_data_points, 3)

# Generate fake cluster labels (you can replace this with your actual data)
num_clusters = 5

# Generate fake light curve data
num_light_curves = 100
lc_data = []
lc_cluster_labels = []
lc_time = np.linspace(0, 10, num_data_points)
for i in range(num_light_curves):
    # Create a random light curve with some noise
    lc_flux = np.sin(lc_time) + np.random.normal(0, 0.2, size=num_data_points)
    lc_data.append((lc_time, lc_flux))
    
    # Assign a random cluster label to the light curve
    lc_cluster_labels.append(np.random.randint(0, num_clusters))

# Convert the list of light curve data and cluster labels to numpy arrays
lc_data = np.array(lc_data)
lc_cluster_labels = np.array(lc_cluster_labels)

# Call the modified movie_cluster function
mydir = '/scratch/echickle/'
# movie_cluster_inset(latent_vectors, lc_cluster_labels, lc_data, mydir)
plot_tsne_inset(latent_vectors, lc_data, lc_cluster_labels, mydir)
