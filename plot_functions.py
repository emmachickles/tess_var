import os
import numpy as np
import matplotlib.pyplot as plt

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

def tsne(latent_vectors, n_components=2):
    from sklearn.manifold import TSNE
    
    # Perform t-SNE dimensionality reduction on the latent vectors
    tsne = TSNE(n_components=n_components, random_state=42)
    latent_tsne = tsne.fit_transform(latent_vectors)
    
    return latent_tsne
    

def plot_tsne(latent_vectors, mydir):
    """Create a scatter plot of the t-SNE latent space"""
    from sklearn.manifold import TSNE
    
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


def plot_tsne_cmap(latent_vectors, label_values, label, mydir, latent_tsne=None, cluster_labels=None):
    import matplotlib.cm as cm
    import matplotlib as mpl
    from sklearn.manifold import TSNE

    # Set the text color to white
    mpl.rcParams['text.color'] = 'white'

    if latent_tsne is None:
        tsne = TSNE(n_components=2, random_state=42)
        latent_tsne = tsne.fit_transform(latent_vectors)

    # Create a scatter plot
    fig, ax = plt.subplots(facecolor='black')
    ax.set_axis_off()

    # Define a color map for the clusters
    n_clusters = len(np.unique(label_values))
    if n_clusters < 100:
        cmap = cm.get_cmap('tab20', n_clusters)
        # Scatter plot with points colored by cluster membership
        scatter = ax.scatter(latent_tsne[:, 0], latent_tsne[:, 1],
                             c=label_values,
                             cmap=cmap, s=2, alpha=0.8)
    else:
        cmap = plt.get_cmap('viridis')  # Choose any colormap you prefer
        scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=label_values, cmap=cmap)

    # Create a color bar legend
    cbar = fig.colorbar(scatter)
    if cluster_labels is not None:
        cbar.ax.set_yticklabels(cluster_labels)
    cbar.ax.yaxis.set_tick_params(color='white')  # Set color bar tick labels to white
    # cbar.ax.yaxis.label.set_color('white')
    cbar.ax.yaxis.set_ticklabels(cbar.ax.get_yticklabels(), color='white')

    # Set color bar edge color to white
    cbar.outline.set_edgecolor('white')

    plt.savefig(mydir + 'tsne_'+label+'.png', dpi=100)
    print('Saved '+mydir + 'tsne_'+label+'.png')
    

def plot_latent_images(decoder, n, latent_dim, latent_vectors, mydir):
    """Plots n x n decoded images sampled from the latent space and saves the figure to mydir."""
    import tensorflow_probability as tfp
    import tensorflow as tf
    from sklearn.manifold import TSNE

    norm = tfp.distributions.Normal(0, 1)
    grid_points = norm.quantile(np.linspace(0.05, 0.95, n))
    grid_points = np.reshape(grid_points, (n, 1))
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

def plot_original_vs_reconstructed(original_data, reconstructed_data, num_examples, mydir, cycles=3):
    plt.figure(figsize=(20, 8))

    reconstruction_loss = np.mean(np.square(original_data - reconstructed_data), axis=1).reshape(-1)

    for i in range(num_examples):
        plt.subplot(2, num_examples, i + 1)
        n_pts = len(original_data[i])
        for j in range(cycles):
            plt.plot(np.arange(j*n_pts, (j+1)*n_pts),
                     original_data[i])
        plt.title('Original')
        plt.xticks([])
        # plt.yticks([])

        plt.subplot(2, num_examples, num_examples + i + 1)
        n_pts = len(reconstructed_data[i])
        for j in range(cycles):
            plt.plot(np.arange(j*n_pts, (j+1)*n_pts),
                     reconstructed_data[i])
        

        plt.title('Reconstructed\nLoss: '+str(np.round(reconstruction_loss[i], 5)))
        plt.xticks([])
        # plt.yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(mydir, 'original_vs_reconstructed.png'))
    plt.close()

def plot_best_worst_reconstructed(original_data, reconstructed_data, num_examples, mydir, cycles=3):
    reconstruction_loss = np.mean(np.square(original_data - reconstructed_data), axis=1).reshape(-1)
    worst_indices = np.argsort(reconstruction_loss)[-num_examples:][::-1]
    worst_loss = np.sort(reconstruction_loss)[-num_examples:][::-1]

    plt.figure(figsize=(20, 8))
    for i, idx in enumerate(worst_indices):
        plt.subplot(2, num_examples, i + 1)
        n_pts = len(original_data[idx])
        for j in range(cycles):
            plt.plot(np.arange(j*n_pts, (j+1)*n_pts),
                     original_data[idx])
        plt.title('Original')
        plt.xticks([])
        # plt.yticks([])


        plt.subplot(2, num_examples, num_examples + i + 1)
        n_pts = len(reconstructed_data[idx])
        for j in range(cycles):
            plt.plot(np.arange(j*n_pts, (j+1)*n_pts),
                     reconstructed_data[idx])
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
        n_pts = len(original_data[idx])
        for j in range(cycles):
            plt.plot(np.arange(j*n_pts, (j+1)*n_pts),
                     original_data[idx])
        plt.title('Original')
        plt.xticks([])
        # plt.yticks([])

        plt.subplot(2, num_examples, num_examples + i + 1)
        n_pts = len(reconstructed_data[idx])
        for j in range(cycles):
            plt.plot(np.arange(j*n_pts, (j+1)*n_pts),
                     reconstructed_data[idx])
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
    from tensorflow.keras.moddels import Model
    
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

def cluster(latent_vectors, cluster_method='dbscan', n_clusters=35):
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
    return cluster_labels

def plot_cluster(latent_vectors, cluster_labels, mydir, latent_tsne=None):
    import matplotlib.cm as cm
    import matplotlib as mpl
    from sklearn.manifold import TSNE

    # Set the text color to white
    mpl.rcParams['text.color'] = 'white'

    if latent_tsne is None:
        tsne = TSNE(n_components=2, random_state=42)
        latent_tsne = tsne.fit_transform(latent_vectors)

    # Create a scatter plot
    fig, ax = plt.subplots(facecolor='black')
    ax.set_axis_off()

    # Define a color map for the clusters
    n_clusters = len(np.unique(cluster_labels))
    cmap = cm.get_cmap('tab20', n_clusters)

    # Scatter plot with points colored by cluster membership
    scatter = ax.scatter(latent_tsne[:, 0], latent_tsne[:, 1],
                         c=cluster_labels,
                         cmap=cmap, s=2, alpha=0.8)

    # Create a color bar legend
    cbar = fig.colorbar(scatter)
    cbar.ax.yaxis.set_tick_params(color='white')  # Set color bar tick labels to white
    # cbar.ax.yaxis.label.set_color('white')
    cbar.ax.yaxis.set_ticklabels(cbar.ax.get_yticklabels(), color='white')

    # Set color bar edge color to white
    cbar.outline.set_edgecolor('white')

    plt.savefig(mydir + 'tsne_cluster.png', dpi=100)
    print('Saved '+mydir + 'tsne_cluster.png')
    
def movie_cluster(latent_vectors, cluster_labels, mydir, latent_tsne=None):
    import matplotlib.cm as cm
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    from sklearn.manifold import TSNE
    
    # Set the text color to white
    mpl.rcParams['text.color'] = 'white'    
    
    # Apply t-SNE to reduce dimensionality to 3D
    if latent_tsne is None:
        tsne = TSNE(n_components=3)
        latent_tsne = tsne.fit_transform(latent_vectors)
    
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()    
    
    # Set black background
    ax.set_facecolor('black')

    # Define a color map for the clusters
    n_clusters = len(np.unique(cluster_labels))
    cmap = cm.get_cmap('tab20', n_clusters)
    
    # Scatter plot with points colored by cluster membership
    colors = np.random.rand(50, 3)
    scatter = ax.scatter(latent_tsne[:, 0], latent_tsne[:, 1],
                         latent_tsne[:, 2], c=cluster_labels,
                         cmap=cmap, s=2, alpha=0.8)
    
    # Remove white border
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Remove tick marks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Remove grid lines
    ax.grid(False)    
    
    # Animation parameters
    frames = 180  # Number of frames for the animation
    elev_start = 30
    elev_end = 60
    azim_start = 0
    azim_end = 100# 360
    zoom_start = 0.8
    zoom_end = 0.4
    
    # Update function for the animation
    def update(frame):
        elev = elev_start + (elev_end - elev_start) * frame / frames
        azim = azim_start + (azim_end - azim_start) * frame / frames
        zoom = zoom_start + (zoom_end - zoom_start) * frame / frames
    
        ax.view_init(elev, azim)
        ax.dist *= zoom 
    
    # Create the animation
    animation = FuncAnimation(fig, update, frames=frames)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    
    # Set up saving options
    save_as_mp4 = True  # Set to False if you want to save as a GIF instead
    if save_as_mp4:
        writer = FFMpegWriter(fps=30, metadata=dict(artist='Me')) 
        filename = '3d_tsne_rotation.mp4'
    else:
        filename = '3d_tsne_rotation.gif'
    
    # Save the animation
    animation.save(mydir+filename, writer=writer if save_as_mp4 else 'pillow')
    print('Saved '+mydir+filename)
    

def plot_anomaly(latent_vectors, mydir, n_neighbors=20, latent_tsne=None):
    import matplotlib as mpl
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.manifold import TSNE

    # Instantiate the LOF algorithm
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)

    # Fit the LOF algorithm on the latent vectors
    lof.fit(latent_vectors)

    # Obtain the anomaly scores for each sample (negative scores indicate anomalies)
    anomaly_scores = -1*lof.negative_outlier_factor_
    anomaly_scores = np.array(anomaly_scores)

    # Define a threshold for classifying samples as anomalies
    threshold = np.quantile(anomaly_scores, 0.9)

    # Create an array of anomaly labels based on the threshold
    anomaly_labels = np.nonzero(anomaly_scores > threshold)

    # Set the text color to white
    mpl.rcParams['text.color'] = 'white'

    if latent_tsne is None:
        tsne = TSNE(n_components=2, random_state=42)
        latent_tsne = tsne.fit_transform(latent_vectors)

    # Define color map and normalize anomaly scores
    cmap = plt.get_cmap('viridis')  # Choose any colormap you prefer
    normalized_scores = (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores))

    # Create a scatter plot
    fig, ax = plt.subplots(facecolor='black')
    ax.set_axis_off()

    # Scatter plot with points colored by cluster membership
    scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=normalized_scores, s=100 * normalized_scores, cmap=cmap)

    # Set colorbar
    cbar = fig.colorbar(scatter)
    cbar.set_label('Anomaly Score', rotation=270, c='white')
    cbar.ax.yaxis.set_tick_params(color='white')  # Set color bar tick labels to white
    # cbar.ax.yaxis.label.set_color('white')
    cbar.ax.yaxis.set_ticklabels(cbar.ax.get_yticklabels(), color='white')

    # Set color bar edge color to white
    cbar.outline.set_edgecolor('white')

    plt.tight_layout()
    plt.savefig(mydir + 'tsne_anomaly.png', dpi=100)
    print('Saved '+mydir + 'tsne_anomaly.png')


def movie_light_curves(light_curves, mydir, total_frames=100, errors=None, ticid=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    
    # Assuming you have your light curves and errors stored in variables called 'light_curves' and 'errors'
    # light_curves and errors should be 2D numpy arrays with the same shape
    
    plt.rcParams['lines.color'] = 'black'    
    
    # Define the number of rows and columns to display
    num_rows = 20
    num_cols = 10
     
    # Calculate the total number of frames and the maximum number of rows
    # total_frames = light_curves.shape[0] - num_rows + 1
    total_frames = total_frames
    max_rows = min(num_rows, light_curves.shape[0])
    
    # Create a figure and axes for the animation
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(12, 8), tight_layout=True)
    

    # Update function for the animation
    def update(frame):
        # Calculate the starting and ending indices for the rows to display in this frame
        start_row = frame
        end_row = start_row + max_rows
    
        # Update the data for each subplot
        for i in range(max_rows):
            for j in range(num_cols):
                idx = start_row + i
                if idx < light_curves.shape[0]:
                    x = np.arange(light_curves.shape[1])
                    y = light_curves[idx]
                    if errors is not None:
                        y_err = errors[idx]
    
                    # Clear the current plot
                    ax[i, j].cla()
    
                    # Calculate the alpha value for the current row
                    alpha = (i + 1) / max_rows
    
                    # Plot the light curve with error bars and set the alpha value
                    if ticid is not None:
                        ax[i,j].text(0, 0, 'TIC '+str(np.int64(ticid[idx])), ha='left', va='center', transform=ax[i,j].transAxes)
                    if errors is None:
                        ax[i,j].plot(x,y, '-', c='k', alpha=alpha)
                    else:
                        ax[i, j].errorbar(x, y, yerr=y_err, fmt='-', color='black', alpha=alpha)
    
                    ax[i, j].set_xlim(0, light_curves.shape[1])
                    ax[i, j].set_ylim(np.min(light_curves - errors), np.max(light_curves + errors))
                    ax[i, j].set_axis_off()

    # Create the animation
    animation = FuncAnimation(fig, update, frames=total_frames, interval=300)

    # Set up saving options
    save_as_mp4 = True  # Set to False if you want to save as a GIF instead
    if save_as_mp4:
        filename = 'scrolling_light_curves.mp4'
        writer = FFMpegWriter(fps=30, metadata=dict(artist='Me')) 
    else:
        filename = 'scrolling_light_curves.gif'
    
    # Save the animation as an MP4 file
    animation.save(mydir+filename, writer='ffmpeg')#, dpi=200)
    print('Saved '+mydir+filename)

def movie_cluster_insets(light_curves, errors, latent_vectors, latent_tsne, cluster_labels, mydir):
        
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import matplotlib.cm as cm
    
    # Assuming you have your 3D t-SNE data stored in a variable called 'tsne_data'
    # tsne_data should be a 2D numpy array of shape (n_samples, 3)
    
    # Assuming you have your light curves stored in a variable called 'light_curves'
    # light_curves should be a 2D numpy array of shape (n_samples, n_time_points)
    
    # Assuming you have your cluster labels stored in a variable called 'cluster_labels'
    # cluster_labels should be a 1D numpy array of shape (n_samples,)
    
    # Define a color map for the clusters
    n_clusters = len(np.unique(cluster_labels))
    cmap = cm.get_cmap('tab20', n_clusters)    
    
    # Create a figure and axes for the animation
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize the scatter plot
    scatter = ax.scatter(latent_tsne[:, 0], latent_tsne[:, 1], latent_tsne[:, 2], c=cluster_labels, cmap=cmap)
    
    # Animation parameters
    frames = 10  # Number of frames for the animation
    elev_start = 30
    elev_end = 60
    azim_start = 0
    azim_end = 100# 360
    zoom_start = 0.8
    zoom_end = 0.4
    
    # Update function for the animation
    def update(frame):
        elev = elev_start + (elev_end - elev_start) * frame / frames
        azim = azim_start + (azim_end - azim_start) * frame / frames
        zoom = zoom_start + (zoom_end - zoom_start) * frame / frames
    
        ax.view_init(elev, azim)
        ax.dist *= zoom 
        
    # Create the animation
    animation = FuncAnimation(fig, update, frames=np.arange(0, frames, 2), interval=100)
    
    # Define the inset parameters
    inset_size = 0.2  # Size of the inset as a fraction of the main plot size
    inset_pad = 0.1  # Padding between the inset and the main plot

    
    # Function to create the inset axes for each point
    def create_inset_axes(ax, light_curve, cluster_label):
        inset_ax = inset_axes(ax, width=inset_size, height=inset_size, loc='upper left', borderpad=inset_pad)
        inset_ax.plot(light_curve)
    
        # Set the border color based on the cluster label
        border_color = cmap(cluster_label)
        for spine in inset_ax.spines.values():
            spine.set_edgecolor(border_color)
    
        inset_ax.set_axis_off()
    
    # Function to update the insets in each frame
    def update_insets(frame):
        for point, light_curve, cluster_label in zip(scatter.get_offsets(), light_curves, cluster_labels):
            create_inset_axes(ax, light_curve, cluster_label)
    
    # Call the update_insets function once to initialize the insets
    update_insets(0)
    
    # Create the animation
    animation = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=100)
    
    # Set up saving options
    filename = '3d_tsne_with_light_curves.mp4'
    
    # Save the animation as an MP4 file
    animation.save(mydir+filename, writer='ffmpeg')
    print('Saved '+mydir+filename)
        
