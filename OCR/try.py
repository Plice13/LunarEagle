import math

def calculate_y_in_x_middle_from_polar(r, theta, x_value):
    # Convert polar coordinates to Cartesian coordinates
    x = r * math.cos(math.radians(theta))
    y = r * math.sin(math.radians(theta))
    
    # Calculate the closest y-intercept
    a = -x / y  # Calculate the slope using the provided point
    b = y - a * x
    
    # Calculate the y-value when x = x_value
    y_value = a * x_value + b
    return y_value

# Example usage:
r = float(input("Enter the radius (r) in polar coordinates: "))
theta = float(input("Enter the angle (θ) in polar coordinates (in degrees): "))
x_value = 1000

result = calculate_y_from_polar(r, theta, x_value)
print(f"When x = {x_value}, y = {result}")
