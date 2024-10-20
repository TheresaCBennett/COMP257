from sklearn.datasets import fetch_olivetti_faces
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras import layers, regularizers # type: ignore
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

# Load the Olivetti faces dataset
data = fetch_olivetti_faces()
X = data.images  
y = data.target  

# Flatten the images 
X_flat = X.reshape(X.shape[0], -1)

# Split the dataset
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Use preserving 99% of the variance to reduce the dataset’s dimensionality
pca = PCA(0.99)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

def build_face_autoencoder(input_size, layer_sizes, reg_strength=0.01):

# Input layer 
    input_image = tf.keras.Input(shape=(input_size,)) 
    
# Encoding 
    encoded = layers.Dense(layer_sizes[0], activation='relu', 
                           kernel_regularizer=regularizers.l2(reg_strength))(input_image)
    
# Second layer of encoding
    compressed = layers.Dense(layer_sizes[1], 
                              kernel_regularizer=regularizers.l2(reg_strength))(encoded)  
    
# Decoding 
    decoded = layers.Dense(layer_sizes[0], activation='relu')(compressed)  
    
# Output 
    reconstructed_image = layers.Dense(input_size)(decoded)  
    
# Build the autoencoder model
    autoencoder = tf.keras.Model(inputs=input_image, outputs=reconstructed_image)
    
    return autoencoder


# Set up 5-fold cross-validation
kf = KFold(n_splits=5)

# Define hyperparameter search space
learning_rates = [0.001, 0.01, 0.1]  
reg_strengths = [0.001, 0.01, 0.1]  

# Initialize variables
best_val_loss = float('inf')  
best_params = None  

# Outer loop over learning rates
for lr in learning_rates:
# Loop over regularization strengths
    for reg in reg_strengths:
        fold_val_losses = []  


# Perform k-fold cross-validation
        for fold_idx, (train_index, val_index) in enumerate(kf.split(X_train_pca)):
            print(f"Training fold {fold_idx + 1} with lr={lr}, reg={reg}")  

# Split the data into training and validation for this fold
            X_fold_train, X_fold_val = X_train_pca[train_index], X_train_pca[val_index]
            y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

 # Build the autoencoder with current regularization 
            autoencoder = build_face_autoencoder(input_size=X_train_pca.shape[1], 
                                                 layer_sizes=[128, 64], 
                                                 reg_strength=reg)

# Compile the model with current learning rate
            autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
            
# Train the model on this fold's training data
            history = autoencoder.fit(X_fold_train, X_fold_train, 
                                      epochs=10, 
                                      batch_size=32, 
                                      validation_data=(X_fold_val, X_fold_val), 
                                      verbose=0)
            
        
# Calculate average validation loss across all folds
        avg_val_loss = np.mean(fold_val_losses)
        print(f"Avg validation loss with lr={lr}, reg={reg}: {avg_val_loss:.4f}")
        

# Run the best model on the test set 
autoencoder = build_face_autoencoder(input_size=X_train_pca.shape[1], layer_sizes=[128, 64], reg_strength=best_params[1])
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_params[0]), loss='mse')

# Train the autoencoder 
autoencoder.fit(X_train_pca, X_train_pca, epochs=50, batch_size=32, validation_data=(X_val_pca, X_val_pca))

# Get the reconstructed images 
reconstructed_images = autoencoder.predict(X_test_pca)

# Plot and compare the original images with their reconstructions
n = 10  
plt.figure(figsize=(20, 4))

for i in range(n):
    # Display original image
    ax = plt.subplot(2, n, i + 1)
    
    plt.imshow(X_test[i].reshape(64, 64)) 
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Display the reconstructed image
    ax = plt.subplot(2, n, i + 1 + n)

    plt.imshow(reconstructed_images[i].reshape(64, 64))  
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
