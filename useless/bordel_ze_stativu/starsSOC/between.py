import numpy as np

def cartesian_to_equatorial(x, y, RA_c, DEC_c, x_c, y_c, pixel_scale):
    RA = (x - x_c) * np.cos(np.radians(DEC_c)) * pixel_scale + RA_c
    DEC = (y - y_c) * pixel_scale + DEC_c
    return RA, DEC

def equatorial_to_cartesian(RA, DEC, RA_c, DEC_c, x_c, y_c, pixel_scale):
    x = (RA - RA_c) / (np.cos(np.radians(DEC_c)) * pixel_scale) + x_c
    y = (DEC - DEC_c) / pixel_scale + y_c
    return x, y

# Define the center of the celestial sphere in Equatorial coordinates (RA, DEC)
RA_c, DEC_c = 114, 32

# Define the center of the celestial sphere in Cartesian coordinates (x, y)
x_c, y_c = 3900, 1500

# Define the pixel scale (arcseconds per pixel)
pixel_scale = 0.5  # Replace with the actual pixel scale of your Cartesian system

# First point in Cartesian coordinates
x1, y1 = 3900, 1500
RA1, DEC1 = 114, 32

# Second point in Cartesian coordinates
x2, y2 = 4600, 2800
RA2, DEC2 = 108, 30

# Convert Cartesian to Equatorial (RA, DEC)
RA1_comp, DEC1_comp = cartesian_to_equatorial(x1, y1, RA_c, DEC_c, x_c, y_c, pixel_scale)
RA2_comp, DEC2_comp = cartesian_to_equatorial(x2, y2, RA_c, DEC_c, x_c, y_c, pixel_scale)

print(f"Point 1: (RA={RA1_comp:.2f} deg, DEC={DEC1_comp:.2f} deg) - Expected (RA={RA1:.2f} deg, DEC={DEC1:.2f} deg)")
print(f"Point 2: (RA={RA2_comp:.2f} deg, DEC={DEC2_comp:.2f} deg) - Expected (RA={RA2:.2f} deg, DEC={DEC2:.2f} deg)")

# Convert Equatorial (RA, DEC) to Cartesian
x1_conv, y1_conv = equatorial_to_cartesian(RA1, DEC1, RA_c, DEC_c, x_c, y_c, pixel_scale)
x2_conv, y2_conv = equatorial_to_cartesian(RA2, DEC2, RA_c, DEC_c, x_c, y_c, pixel_scale)

print(f"Converted back to Cartesian: (x={x1_conv:.2f}, y={y1_conv:.2f}) - Expected (x={x1:.2f}, y={y1:.2f})")
print(f"Converted back to Cartesian: (x={x2_conv:.2f}, y={y2_conv:.2f}) - Expected (x={x2:.2f}, y={y2:.2f})")
