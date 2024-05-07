import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster

# Suppress joblib warning
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

print("Initializing...")

# Function to detect face in an image
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]

# Function to extract HOG features from an image (with error handling)
def extract_features(img):
    try:
        face, _ = detect_face(img)
        if face is not None:
            if face.shape[0] > 0 and face.shape[1] > 0:
                face_resized = cv2.resize(face, (50, 50))  # Resize to 50x50
                win_size = (50, 50)
                block_size = (10, 10)
                block_stride = (5, 5)
                cell_size = (5, 5)
                nbins = 9
                hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
                hog_feature = hog.compute(face_resized)
                return hog_feature.flatten()
        return None
    except cv2.error as e:
        print(f"OpenCV Error: {e}")
        return None  # Return None in case of errors

print("Loading and preprocessing the dataset...")

# Function to load and preprocess the dataset
def load_dataset(dataset_path):
    X, y = [], []
    label = 0
    for dir_name in os.listdir(dataset_path):
        dir_path = os.path.join(dataset_path, dir_name)
        for img_file in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                face, _ = detect_face(img)
                if face is not None:
                    features = extract_features(img)
                    if features is not None:
                        X.append(features)
                        y.append(label)
        label += 1
    return np.array(X), np.array(y)

# Load the pre-trained Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load and preprocess the dataset
dataset_path = r"lfw_funneled"  # Use raw string for paths with backslashes
X, y = load_dataset(dataset_path)

print("Performing Dimensionality Reduction using KMeans...")

# Dimensionality Reduction using KMeans
kmeans = KMeans(n_clusters=50)
clusters = kmeans.fit_predict(X)

print("Performing Hierarchical Clustering...")

# Hierarchical Clustering
Z = linkage(X, method='ward', metric='euclidean')
hov_labels = fcluster(Z, t=1, criterion='maxclust')

print("Splitting the dataset into train and test sets...")

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Training SVM Classifier...")

# SVM Classification
clf = SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Calculating accuracy...")

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Plotting the accuracy graph
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(y_test, y_pred, 'o', alpha=0.5)
plt.xlabel('Trained')
plt.ylabel('Test')
plt.title('Trained vs Test')
plt.show()
