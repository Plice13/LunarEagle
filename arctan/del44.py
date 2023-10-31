import math

def calculate_angle(p1, p2, p3, p4):
    angle1 = math.atan2(p1['y'] - p2['y'], p1['x'] - p2['x']) - math.atan2(p3['y'] - p4['y'], p3['x'] - p4['x'])
    angle2 = math.atan2(p1['y'] - p3['y'], p1['x'] - p3['x']) - math.atan2(p2['y'] - p4['y'], p2['x'] - p4['x'])
    return angle1, angle2

# Define the points
P1 = {'x': 0, 'y': 0}
P2 = {'x': 1, 'y': 0}
P3 = {'x': 0, 'y': 1}
P4 = {'x': 2, 'y': 2}

angle1, angle2 = calculate_angle(P1, P2, P3, P4)


print("Angle between lines formed by P1-P2 and P3-P4:", math.degrees(angle1))
print("Angle between lines formed by P1-P3 and P2-P4:", math.degrees(angle2))
