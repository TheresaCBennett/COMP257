from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt



# Load the Olivetti faces dataset
faces = fetch_olivetti_faces()
X = faces.data  
y = faces.target  


# Split the data 70% train, 30% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Split the temp data into validation and test sets 15% for each
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test.shape}")

# Fit PCA on the training data
pca = PCA(0.99)  # Retain 99% of variance
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

print(f"Reduced training set shape: {X_train_pca.shape}")

# Test different covariance types
covariance_types = ['full', 'tied', 'diag', 'spherical']
best_aic = np.inf
best_gmm = None

for cov_type in covariance_types:
    gmm = GaussianMixture(n_components=10, covariance_type=cov_type)
    gmm.fit(X_train_pca)
    aic = gmm.aic(X_train_pca)
    if aic < best_aic:
        best_aic = aic
        best_gmm = gmm
    print(f"Covariance Type: {cov_type}, AIC: {aic}")

print(f"Best covariance type: {best_gmm.covariance_type}")


#Determine nunber of clusters 
n_components_range = range(1, 21)
aic_values = []
bic_values = []

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.fit(X_train_pca)
    aic_values.append(gmm.aic(X_train_pca))
    bic_values.append(gmm.bic(X_train_pca))


#plot 

plt.figure(figsize=(12, 6))
plt.plot(n_components_range, aic_values, label='AIC', marker='o')
plt.plot(n_components_range, bic_values, label='BIC', marker='o')
plt.xlabel('Number of Components')
plt.ylabel('AIC/BIC')
plt.title('AIC/BIC vs Number of Components')
plt.legend()
plt.show()