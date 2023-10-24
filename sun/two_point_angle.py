import numpy as np

reference_point = (10, 10)  
point2 = (8, 12)

# Spočtěte úhel vůči svislé ose
angle = np.degrees(np.arctan2(point2[0] - reference_point[0], -(point2[1] - reference_point[1])))

if angle < 0:
    angle = angle+360
elif angle >360:
    angle = angle-360
else:
    angle=angle

print(f'Úhel vůči svislé ose: {angle} stupňů')
