from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


knn_model = KNeighborsClassifier(n_neighbors=5)

# Train the KNN model
knn_model.fit(X_train, y_train.values.ravel())

# Make predictions on both the training and test sets
y_train_pred = knn_model.predict(X_train)
y_test_pred = knn_model.predict(X_test)

# performance metrics for the training set
model_train_accuracy = accuracy_score(y_train, y_train_pred)
model_train_f1 = f1_score(y_train, y_train_pred, average='weighted')
model_train_precision = precision_score(y_train, y_train_pred, average='weighted')
model_train_recall = recall_score(y_train, y_train_pred, average='weighted')

# performance metrics for the test set
model_test_accuracy = accuracy_score(y_test, y_test_pred)
model_test_f1 = f1_score(y_test, y_test_pred, average='weighted')
model_test_precision = precision_score(y_test, y_test_pred, average='weighted')
model_test_recall = recall_score(y_test, y_test_pred, average='weighted')

# performance metrics for the KNN model
print('Model performance for Training set (K-Nearest Neighbors)')
print("- Accuracy: {:.4f}".format(model_train_accuracy))
print('- F1 score: {:.4f}'.format(model_train_f1))
print('- Precision: {:.4f}'.format(model_train_precision))
print('- Recall: {:.4f}'.format(model_train_recall))
print('----------------------------------')
print('Model performance for Test set (K-Nearest Neighbors)')
print('- Accuracy: {:.4f}'.format(model_test_accuracy))
print('- F1 score: {:.4f}'.format(model_test_f1))
print('- Precision: {:.4f}'.format(model_test_precision))
print('- Recall: {:.4f}'.format(model_test_recall))
