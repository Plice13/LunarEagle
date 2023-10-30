import numpy as np

# Coordinates of the points
provided_points = np.array([[1549, 519], [1577, 581], [1588, 501], [1616, 563]])

# Calculate pairwise distances between all provided points
distances = np.sqrt(np.sum((provided_points[:, None] - provided_points) ** 2, axis=-1))

# Print all distances between the provided points
print("Distances between all pairs of provided points:")
for i in range(len(provided_points)):
    for j in range(i + 1, len(provided_points)):
        distance = distances[i, j]
        if distance < boundry:
            return True
return False
