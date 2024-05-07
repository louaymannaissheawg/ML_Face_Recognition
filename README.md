### Face Recognition using SVM

This project implements a face recognition system using Support Vector Machine (SVM) classifier. It detects faces in images, extracts Histogram of Oriented Gradients (HOG) features, performs dimensionality reduction using KMeans clustering, and applies hierarchical clustering. Finally, it trains an SVM classifier for face recognition.

#### Technologies Used

- **OpenCV (cv2)**: For image processing and face detection.
- **NumPy**: For numerical computing and array manipulation.
- **Matplotlib**: For data visualization, particularly plotting accuracy graphs.
- **scikit-learn (sklearn)**: For machine learning tasks such as SVM classification, train-test split, and accuracy calculation.
- **SciPy**: For hierarchical clustering.
- **KMeans**: For dimensionality reduction.
- **SVM (Support Vector Machine)**: For training the face recognition model.

#### Usage Instructions

1. Ensure you have the required Python libraries installed: OpenCV, NumPy, Matplotlib, scikit-learn, and SciPy.
   
2. Run the provided Python script to initialize the face recognition system.
   
3. The script will load and preprocess the dataset, perform dimensionality reduction using KMeans, apply hierarchical clustering, split the dataset into train and test sets, train the SVM classifier, and calculate accuracy.

4. Explore the accuracy graph to evaluate the performance of the trained model.

Feel free to customize and extend the project for your specific use case or dataset! Let's explore the fascinating world of face recognition with SVM. 🧑‍💻📸🤖
