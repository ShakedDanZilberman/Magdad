
def coordinate_to_angle(x, y):
    """
    Convert coordinates to angle relative to the constant coordinates GUN in degrees.
    """
    from math import atan2, degrees

    # Calculate the angle in radians
    angle_rad = atan2(y - GUN[1], x - GUN[0])

    # Convert to degrees
    angle_deg = degrees(angle_rad)

    # Normalize the angle to be between 0 and 360 degrees
    if angle_deg < 0:
        angle_deg += 360

    return angle_deg