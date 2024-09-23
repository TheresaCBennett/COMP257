# Load the MNIST dataset using OpenML
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA, IncrementalPCA
import matplotlib.pyplot as plt

#Question1: 

#Retrieve and load mnist_784 dataset
mnist_data = fetch_openml('mnist_784', version=1, as_frame=False)
features = mnist_data.data
print(f"Shape of MNIST data: {mnist_data.data.shape}")  

#Display digits
fig, axes = plt.subplots(10, 10, figsize=(8, 8))
for idx, ax in enumerate(axes.flat):
    ax.imshow(features[idx].reshape(28, 28), cmap='gray')
    ax.axis('off')


pca_model = PCA(n_components=2)
pca_transformed = pca_model.fit_transform(features)

# Display the variance ratio for the principal components
variance_ratio = pca_model.explained_variance_ratio_
print("Variance Explained by Each Principal Component:", variance_ratio)

#plot the PCA results
plt.figure(figsize=(8, 6))
plt.scatter(pca_transformed[:, 0], pca_transformed[:, 1], c=mnist_data.target.astype(int), cmap='jet', s=10)
plt.colorbar(label='Digit Labels')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Visualization of MNIST')
plt.show()

# Apply Incremental PCA for compression using 154 components
ipca_model = IncrementalPCA(n_components=154)
compressed_data = ipca_model.fit_transform(features)

# Displayoriginal digit and compressed version
plt.figure(figsize=(8, 4))

# Original digit plot
plt.subplot(1, 2, 1)
plt.imshow(features[0].reshape(28, 28), cmap='gray')
plt.title('Original Digit')
plt.axis('off')

# Compressed digit plot
plt.subplot(1, 2, 2)
plt.imshow(ipca_model.inverse_transform(compressed_data[0]).reshape(28, 28), cmap='gray')
plt.title('Reconstructed Digit from 154 PCs')
plt.axis('off')

plt.show()

#Question2: 


from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np

# Generate a Swiss Roll dataset
swiss_roll_data, color_map = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

#Plot dataset
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(swiss_roll_data[:, 0], swiss_roll_data[:, 1], swiss_roll_data[:, 2], c=color_map, cmap='viridis', s=20)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('Visualization of Swiss Roll Dataset')
plt.show()

# Use Kernel PCA with lineral kernel. RBF and sigmoid kernel
kernel_types = ['linear', 'rbf', 'sigmoid']
kpca_models = {
    'Linear': KernelPCA(kernel='linear'),
    'RBF': KernelPCA(kernel='rbf', gamma=0.1),
    'Sigmoid': KernelPCA(kernel='sigmoid', gamma=0.01)
}

# Transform the dataset
transformed_data = {kernel: model.fit_transform(swiss_roll_data) for kernel, model in kpca_models.items()}

# Plot the results
plt.figure(figsize=(12, 4))

for idx, (kernel_name, data) in enumerate(transformed_data.items()):
    plt.subplot(1, 3, idx + 1)
    plt.scatter(data[:, 0], data[:, 1], c=color_map, cmap='viridis', s=20)
    plt.title(f'Kernel PCA with {kernel_name} Kernel')

plt.show()

# Set up a pipeline for Kernel 
pipeline = Pipeline([
    ('kpca', KernelPCA()),
    ('logistic', LogisticRegression(max_iter=1000))
])

# Define grid parameters 
param_grid = {
    'kpca__kernel': ['linear', 'rbf', 'sigmoid'],
    'kpca__gamma': [0.001, 0.01, 0.1, 1],
    'logistic__C': [0.001, 0.01, 0.1, 1]
}

# Execute grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=3)
grid_search.fit(swiss_roll_data, color_map)

# Output the best parameters 
best_parameters = grid_search.best_params_
print("Optimal Parameters Found:", best_parameters)

# Visualize grid search results
results = grid_search.cv_results_

plt.figure(figsize=(12, 6))
plt.subplots_adjust(wspace=0.4)

for idx, kernel in enumerate(['linear', 'rbf', 'sigmoid']):
    plt.subplot(1, 3, idx + 1)
    scores_matrix = results['mean_test_score'][idx::3].reshape(len(param_grid['kpca__gamma']), -1)
    plt.imshow(scores_matrix, cmap='viridis', interpolation='nearest')
    plt.xlabel('Regularization C')
    plt.ylabel('Gamma')
    plt.colorbar()
    plt.title(f'{kernel.capitalize()} Kernel Performance')

plt.show()


