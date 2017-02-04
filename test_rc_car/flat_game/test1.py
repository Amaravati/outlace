def make_sonar_arm(x, y):
        spread = 10  # Default spread.
        distance = 20  # Gap before first sensor.
        arm_points = []
        # Make an arm. We build it flat because we'll rotate it about the
        # center later.
        for i in range(1, 10):
            arm_points.append((distance + x + (spread * i), y))

        return arm_points
 
print(make_sonar_arm(5,5))

