import math

# Center point
x1, y1 = 1000, 875

# Input for the second point
x2 = float(input("Enter the x-coordinate of the second point: "))
y2 = float(input("Enter the y-coordinate of the second point: "))

# Calculate the angle
angle_radians = math.atan2(x2 - x1, y2 - y1)
angle_degrees = math.degrees(angle_radians)

# Display the result
print(f"The angle formed between the y-axis and the line connecting ({x1},{y1}) and ({x2},{y2}) is {angle_degrees} degrees.")
