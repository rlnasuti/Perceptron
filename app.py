import perceptron
import numpy as np

X = np.array([[0, 0],  # Convert X to a 2D NumPy array
              [0, 1],
              [1, 0],
              [1, 1]])

d = np.array([0, 0, 0, 1])

p = perceptron.Perceptron(input_size = 2)
p.fit(X, d)

# Make predictions for each input
for x in X:
    y = p.predict(x)
    print(f"Input: {x} Prediction: {y}")

# Save the weights to a text file
np.savetxt('weights.txt', p.W)

