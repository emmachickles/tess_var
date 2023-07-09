import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.keras.models import Model
import tensorflow_probability as tfp
import tensorflow as tf

def plot_loss(history, mydir):
    # Extract loss history
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    # Plot loss vs epoch
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(mydir + 'loss.png')

def plot_tsne(latent_vectors, mydir):
    """Create a scatter plot of the t-SNE latent space"""

    # Perform t-SNE dimensionality reduction on the latent vectors
    tsne = TSNE(n_components=2, random_state=42)
    latent_tsne = tsne.fit_transform(latent_vectors)

    # Extract the t-SNE coordinates
    tsne_x = latent_tsne[:, 0]
    tsne_y = latent_tsne[:, 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_x, tsne_y, c='b', alpha=0.5)
    plt.title('t-SNE Latent Space Visualization')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(mydir+'tsne.png')

def plot_latent_images(decoder, n, latent_dim, latent_vectors, mydir):
    """Plots n x n decoded images sampled from the latent space and saves the figure to mydir."""
    import tensorflow_probability as tfp

    norm = tfp.distributions.Normal(0, 1)
    grid_points = norm.quantile(np.linspace(0.05, 0.95, n))
    grid_points = tf.reshape(grid_points, (n, 1))
    z = tf.repeat(grid_points, repeats=latent_dim, axis=1)  # Repeat grid_points for each latent dimension
    x_decoded = decoder.predict(z, steps=1)  # Generate decoded images

    nrow = int(np.sqrt(n))
    ncol = int(np.sqrt(n))

    fig, ax = plt.subplots(nrow, ncol, figsize=(15,8)) # Plot light curves
    for i in range(nrow):
        for j in range(ncol):
            ax[i, j].plot(x_decoded[nrow*i + j])
            ax[i, j].set_xticks([])
            # ax[i, j].set_yticks([])

    fig.savefig(os.path.join(mydir, 'latent_images.png'))
    plt.close()

    # Perform t-SNE dimensionality reduction on the latent vectors and grid_points
    tsne = TSNE(n_components=2, random_state=42)
    all_vectors = np.append(latent_vectors, z.eval(session=tf.compat.v1.Session()), 0)
    latent_tsne = tsne.fit_transform(all_vectors)

    # Extract the t-SNE coordinates
    tsne_x = latent_tsne[:, 0]
    tsne_y = latent_tsne[:, 1]

    fig1, ax1 = plt.subplots(figsize=(8,6)) # Plot tsne
    ax1.set_title('t-SNE Latent Space Visualization')
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')
    ax1.plot(tsne_x[:-z.shape[0]], tsne_y[:-z.shape[0]], 'k.', ms=1, alpha=0.5)
    for i in range(z.shape[0]):
        x, y = tsne_x[z.shape[0]+i], tsne_y[z.shape[0]+i]
        with plt.rc_context({'lines.markerfacecolor': 'b'}):
            ax1.plot(x, y, 'o', ms=10)
        row = i // nrow
        col = i % ncol
        ax1.annotate('{},{}'.format(row,col), (x, y+1.),
                     ha='center', c='k')
    fig1.savefig(os.path.join(mydir, 'latent_tsne.png'))
    plt.close()

def plot_original_vs_reconstructed(original_data, reconstructed_data, num_examples, mydir):
    plt.figure(figsize=(20, 8))

    reconstruction_loss = np.mean(np.square(original_data - reconstructed_data), axis=1).reshape(-1)

    for i in range(num_examples):
        plt.subplot(2, num_examples, i + 1)
        plt.plot(original_data[i])
        plt.title('Original')
        plt.xticks([])
        # plt.yticks([])

        plt.subplot(2, num_examples, num_examples + i + 1)
        plt.plot(reconstructed_data[i])
        plt.title('Reconstructed\nLoss: '+str(np.round(reconstruction_loss[i], 5)))
        plt.xticks([])
        # plt.yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(mydir, 'original_vs_reconstructed.png'))
    plt.close()

def plot_best_worst_reconstructed(original_data, reconstructed_data, num_examples, mydir):
    reconstruction_loss = np.mean(np.square(original_data - reconstructed_data), axis=1).reshape(-1)
    worst_indices = np.argsort(reconstruction_loss)[-num_examples:][::-1]
    worst_loss = np.sort(reconstruction_loss)[-num_examples:][::-1]

    plt.figure(figsize=(20, 8))
    for i, idx in enumerate(worst_indices):
        plt.subplot(2, num_examples, i + 1)
        plt.plot(original_data[idx])
        plt.title('Original')
        plt.xticks([])
        # plt.yticks([])

        plt.subplot(2, num_examples, num_examples + i + 1)
        plt.plot(reconstructed_data[idx])
        plt.title('Reconstructed\nLoss: '+str(np.round(worst_loss[i], 5)))
        plt.xticks([])
        # plt.yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(mydir, 'worst_reconstructed.png'))
    plt.close()

    best_indices = np.argsort(reconstruction_loss)[:num_examples]
    best_loss = np.sort(reconstruction_loss)[:num_examples]

    plt.figure(figsize=(20, 8))
    for i, idx in enumerate(best_indices):
        plt.subplot(2, num_examples, i + 1)
        plt.plot(original_data[idx])
        plt.title('Original')
        plt.xticks([])
        # plt.yticks([])

        plt.subplot(2, num_examples, num_examples + i + 1)
        plt.plot(reconstructed_data[idx])
        plt.title('Reconstructed\nLoss: '+str(np.round(best_loss[i], 5)))
        plt.xticks([])
        # plt.yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(mydir, 'best_reconstructed.png'))
    plt.close()

def plot_reconstruction_distribution(original_data, reconstructed_data, mydir):
    reconstruction_loss = np.mean(np.square(original_data - reconstructed_data), axis=1).reshape(-1)
    plt.figure()
    _ = plt.hist(reconstruction_loss, bins=100)
    plt.xlabel('Reconstruction Loss')
    plt.ylabel('Training Instances')
    plt.tight_layout()
    plt.savefig(os.path.join(mydir, 'loss_dist.png'))

def visualize_layer_outputs(model, data, num_examples, mydir):
    layer_names = [layer.name for layer in model.layers[1:]]  # Get the names of all layers in the model
    layer_outputs = [layer.output for layer in model.layers[1:]]  # Get the output tensors of all layers

    # Create a new model that maps the input data to the output tensors of each layer
    intermediate_model = Model(inputs=model.input, outputs=layer_outputs)

    # Get the intermediate layer outputs for the given data
    intermediate_outputs = intermediate_model.predict(data)

    # Plot and save the output from each layer
    for num, (layer_name, intermediate_output) in enumerate(zip(layer_names, intermediate_outputs)):
        if len(intermediate_output.shape) == 3:  # Handle 1D convolutional layer outputs
            num_filters = intermediate_output.shape[-1]
            fig, axes = plt.subplots(num_filters, num_examples, figsize=(16, 16))
            fig.suptitle(layer_name)

            for i in range(num_filters):
                for j in range(num_examples):
                    axes[i,j].plot(intermediate_output[j, :, i])
                    axes[i,j].set_xticks([])
                    # axes[i].set_yticks([])

            plt.savefig(os.path.join(mydir, f'{num}_{layer_name}.png'))
            plt.close()

        elif len(intermediate_output.shape) == 2:  # Handle dense layer outputs
            fig, ax = plt.subplots(ncols=num_examples, figsize=(16, 5))
            fig.suptitle(layer_name)

            for i in range(num_examples):
                ax[i].plot(intermediate_output[i])
                ax[i].set_xticks([])
                # ax.set_yticks([])

            plt.savefig(os.path.join(mydir, f'{num}_{layer_name}.png'))
            plt.close()

def plot_cluster(latent_vectors, mydir, cluster_method='dbscan', n_clusters=35):
    import matplotlib.cm as cm
    import matplotlib as mpl

    # Set the text color to white
    mpl.rcParams['text.color'] = 'white'

    if cluster_method == 'kmeans':
        from sklearn.cluster import KMeans
        # Instantiate the k-means clurtering algorithm
        kmeans = KMeans(n_clusters=n_clusters)

        # Fit the k-means algorithm on the latent vectors
        kmeans.fit(latent_vectors)

        # Obtain the cluster labels for each sample
        cluster_labels = kmeans.labels_
    elif cluster_method == 'dbscan':
        from sklearn.cluster import DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = dbscan.fit_predict(latent_vectors)
        n_clusters = np.max(cluster_labels) # Lables go as 0, 1, 2, ... and -1 for unclustered

    tsne = TSNE(n_components=2, random_state=42)
    latent_tsne = tsne.fit_transform(latent_vectors)

    # Create a scatter plot
    fig, ax = plt.subplots(facecolor='black')

    # Define a color map for the clusters
    cmap = cm.get_cmap('tab20', n_clusters)

    # Scatter plot with points colored by cluster membership
    scatter = ax.scatter(latent_tsne[:, 0], latent_tsne[:, 1],
                         c=cluster_labels,
                         cmap=cmap, s=10, alpha=0.5)

    # Create a color bar legend
    cbar = fig.colorbar(scatter)
    ax.set_axis_off()

    plt.savefig(mydir + 'tsne_cluster.png')
    

def plot_anomaly(latent_vectors, mydir, n_neighbors=20):

    from sklearn.neighbors import LocalOutlierFactor

    # Instantiate the LOF algorithm
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)

    # Fit the LOF algorithm on the latent vectors
    lof.fit(latent_vectors)

    # Obtain the anomaly scores for each sample (negative scores indicate anomalies)
    anomaly_scores = -1*lof.negative_outlier_factor_

    # Define a threshold for classifying samples as anomalies
    threshold = -2.5

    # Create an array of anomaly labels based on the threshold
    anomaly_labels = [1 if score < threshold else 0 for score in anomaly_scores]

    # Count the number of anomalies
    num_anomalies = sum(anomaly_labels)

    # Print the number of anomalies
    print("Number of anomalies detected:", num_anomalies)


