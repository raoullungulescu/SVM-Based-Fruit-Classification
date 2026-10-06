# Binary Fruit Classifier: SVM and Dimensionality Reduction with PCA

This repository contains the implementation of a Machine Learning model capable of distinguishing between two red apple varieties based on image features. 
The project utilizes the Support Vector Machine (SVM) algorithm for binary decision classification and explores the impact of Principal Component Analysis (PCA) on feature space reduction and overall accuracy.

## 📊 Dataset and Preprocessing

The data is split into a 75% training set and a 25% testing set.
* **Training Set:** 984 images (492 images per variety).
* **Testing Set:** 328 images (164 images per variety).

<img width="1193" height="375" alt="Screenshot 2026-10-06 at 22 26 05" src="https://github.com/user-attachments/assets/9240d506-ecc4-402d-ad82-36d663cd9d67" />

## 📈 Results and Evaluation

The application evaluates performance across two classification scenarios to demonstrate the impact of feature compression:

* **Model 1 (Standard SVM, without PCA):** Achieved an accuracy of **98.78%**.
* **Model 2 (SVM with PCA):** Retaining only the first 2 principal components for effective 2D linear decision visualization, the model achieved an accuracy of **85.67%**.

<img width="1077" height="487" alt="Screenshot 2026-10-06 at 22 26 26" src="https://github.com/user-attachments/assets/6eed6bae-d049-41e7-a879-fd087335b4dd" />
