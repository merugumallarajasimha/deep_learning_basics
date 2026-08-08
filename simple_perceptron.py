import numpy as np


def step_function(z):
    return np.where(z>=0,1,0)

class Perceptron:

    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        # Initialize weights with small random values and bias with 0
        self.weights = np.zeros(input_size)
        self.bias = 0.0
        self.lr = learning_rate
        self.epochs = epochs

    def predict(self, X):
        # Calculate net input: z = (X . W) + b
        linear_output = np.dot(X, self.weights) + self.bias
        # Apply step activation function
        return step_function(linear_output)

    def fit(self, X, y):
        # Training loop over specified epochs
        for epoch in range(self.epochs):
            total_errors = 0
            for xi, target in zip(X, y):
                # 1. Compute predicted output
                prediction = self.predict(xi)

                # 2. Compute error: (target - prediction)
                error = target - prediction

                # 3. Update weights and bias if there's an error
                update = self.lr * error
                self.weights += update * xi
                self.bias += update

                total_errors += int(error != 0)

            # Early stopping if model converged
            if total_errors == 0:
                print(f"Converged at epoch {epoch + 1}!")
                break
    # Inputs: 4 pairs of 0s and 1s
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Target outputs for AND logic
y = np.array([0, 0, 0, 1])
# Initialize Perceptron with 2 input features
perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=10)

# Train the model
perceptron.fit(X, y)
print("\n--- Learned Parameters ---")
print(f"Weights: {perceptron.weights}")
print(f"Bias:    {perceptron.bias:.2f}")

print("\n--- Predictions ---")
for xi, target in zip(X, y):
    pred = perceptron.predict(xi)
    print(f"Input: {xi} | Target: {target} | Predicted: {pred}")          
