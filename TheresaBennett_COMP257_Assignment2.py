from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.svm import SVC

 
#Load Olivetti Faces dataset 
data = fetch_olivetti_faces()
X = data.images    
y = data.target  

#Split training set, validation set and test set - using stratified sampling
# 20% of the dataset to the test set, and the remaining 80% is used for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 80% of data allocated for training in the first step and split it further into a training and validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=42)

# Using k-fold cross validation, train a classifier to predict which person is represented in each picture
kfold = StratifiedKFold(n_splits=5, random_state=42)
clf = LogisticRegression()
for train_index, val_index in kfold.split(X_train, y_train):
    X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
    y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]
    clf.fit(X_train_cv, y_train_cv)
    y_pred = clf.predict(X_val_cv)
    print(f'Accuracy: {accuracy_score(y_val_cv, y_pred)}')


#Use K-Means to reduce the dimensionality of the set
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X_train)
reduced_X_train = kmeans.transform(X_train)

# train a classifier 
svm_clf = SVC(kernel='linear', random_state=42)
svm_clf.fit(X_train, y_train)
val_score = svm_clf.score(X_val, y_val)
print(f"Validation accuracy: {val_score}")




#References
#2.3. clustering. scikit. (n.d.-a). https://scikit-learn.org/stable/modules/clustering.html 
#Sklearn.datasets.fetch_olivetti_faces¶. scikit. (n.d.-b). https://scikit-learn.org/0.19/modules/generated/sklearn.datasets.fetch_olivetti_faces.html#sklearn.datasets.fetch_olivetti_faces 