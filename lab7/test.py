import numpy as np

# Create an empty array
array = np.zeros((5, 0))
print(array)
# Perform iterations
for i in range(1, 6):
    # Generate a random column
    column = np.random.randint(10, size=(5, 1))
    print(column.shape)
    # Add the column to the array
    array = np.concatenate((array, column), axis=1)

    # Display the updated array
    print(f"Iteration {i}:\n{array}\n")