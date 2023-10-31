import numpy as np

def is_right_angle(approx):
    # Ensure there are exactly 4 points
    if len(approx) != 4:
        return False

    # Define vectors for each side of the shape (assuming the points are ordered)
    vectors = [np.array(approx[i]) - np.array(approx[(i + 1) % 4]) for i in range(4)]
    print(f'Vectors are:\n{vectors}\n')
    # Calculate the dot product of adjacent sides
    dot_products = [np.dot(vectors[i], vectors[(i + 1) % 4]) for i in range(4)]
    print(f'Dot products are:\n{dot_products}\n')

    # Calculate the cosine of the angles between sides using dot product properties
    cosines = [dot_products[i] / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[(i + 1) % 4])) for i in range(4)]
    print(f'Cosines are:\n{cosines}\n')

    # Check if any angle is close to 90 degrees
    angle_threshold = 0.173  # Cosine of approximately 10 degrees
    print(f'Konec je: {any(abs(cosine) < angle_threshold for cosine in cosines)}')
    return any(abs(cosine) < angle_threshold for cosine in cosines)

# Example usage:
approx = [[0, 0], [0, 1], [1,0], [1,1]]
result = is_right_angle(approx)
print(f"Does the shape form a right angle?: {result}")