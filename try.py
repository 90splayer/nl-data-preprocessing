import numpy as np

# Given array
arr = np.array([[9, 9, 9],
                [1, 0, 5],
                [2, 0, 1]])

# Get the indices that would sort the array by the first column
sorted_indices = np.argsort(arr[:, 0])

# Sort the array by the first column
sorted_arr = arr[sorted_indices]

print(sorted_arr)