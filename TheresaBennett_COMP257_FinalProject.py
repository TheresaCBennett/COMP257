import scipy.io as io
import os
import numpy as np
import cv2  
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  # type: ignore
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Get the absolute path of script
current_file_location = os.path.abspath(__file__)

# Define the path 
mat_file_loc = os.path.join(os.path.dirname(current_file_location), 'umist_cropped.mat')

# Load the data 
mat_data = io.loadmat(mat_file_loc)

# Extract 'facedat' 
facedat_data = mat_data.get('facedat')  

# Print data type and inspect a sample 
print(f"Type of 'facedat': {type(facedat_data)}")
print(facedat_data)  


# Flattening and creating the dataset array 
dataset = []
labels = [] 

for i in range(facedat_data[0].size()):  
    for j in range(facedat_data[0][i].size):  
        image = facedat_data[0][i][j]
        dataset.append(image.flatten())  
        labels.append(i)  

# Convert to numpy arrays 
dataset = np.array(dataset)
labels = np.array(labels)

# Inspect the dataset
print(f"Shape of dataset: {dataset.shape}") 
print(f"Labels shape: {labels.shape}")


# Normalizing dataset 
dataset = dataset / 255.0 

# Applying PCA 
scaler = StandardScaler()
dataset_scaled = scaler.fit_transform(dataset)

# Trying PCA to reduce features 
pca = PCA(n_components=100)  
dataset_pca = pca.fit_transform(dataset_scaled)

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(dataset_pca, labels, test_size=0.3, stratify=labels)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)

# print statements to check if working
print(f"Train set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")


# Applying K-means 
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X_train)

# 4. Building a CNN for image classification
model = Sequential([ # type: ignore
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),  
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(set(labels)), activation='softmax')  
])

# Compiling model 
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy']) # type: ignore

# Training model 
X_train_resized = np.array([cv2.resize(image.reshape(64, 64), (64, 64)) for image in X_train])
X_val_resized = np.array([cv2.resize(image.reshape(64, 64), (64, 64)) for image in X_val])
X_test_resized = np.array([cv2.resize(image.reshape(64, 64), (64, 64)) for image in X_test])

X_train_resized = X_train_resized.reshape(-1, 64, 64, 3)  
X_val_resized = X_val_resized.reshape(-1, 64, 64, 3)
X_test_resized = X_test_resized.reshape(-1, 64, 64, 3)

# fit model
model.fit(X_train_resized, y_train, validation_data=(X_val_resized, y_val), epochs=10)

# Evaluate on test set 
test_loss, test_accuracy = model.evaluate(X_test_resized, y_test)
print(f"Test accuracy: {test_accuracy}")

# 7. Results Evaluation 
predictions = model.predict(X_test_resized)
predicted_labels = np.argmax(predictions, axis=1)

# Plotting matrix
cm = confusion_matrix(y_test, predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(labels), yticklabels=np.unique(labels))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


