import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def digit_to_binary_matrix(digit):
    binary_representation = bin(digit)[2:].zfill(15)
    return [[int(binary_representation[i * 3 + j]) for j in range(3)] for i in range(5)]

digits = {num: digit_to_binary_matrix(num) for num in [0, 1, 2, 39]}

X = np.array([np.array(digits[key]).flatten() for key in digits.keys()])
y = np.array([0, 1, 2, 3])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', solver='adam', max_iter=1000, random_state=42)

model.fit(X_train, y_train)

def predict_number(digit):
    matrix = np.array(digit_to_binary_matrix(digit)).flatten().reshape(1, -1)
    prediction = model.predict(matrix)
    label_map = {0: 0, 1: 1, 2: 2, 3: 39}
    return label_map[prediction[0]]

test_digits = [2, 39]

for i, digit in enumerate(test_digits):
    print(f"Test {i+1} Prediction: {predict_number(digit)}")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
