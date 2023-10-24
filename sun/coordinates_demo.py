import sunpy.coordinates

B0_str = str(sunpy.coordinates.sun.B0(time='20230926084800'))

parts = B0_str.split('d')
degrees = float(parts[0])

minutes_str, seconds_str = parts[1].split('m')
minutes = float(minutes_str)

seconds_str = seconds_str.rstrip('s')
seconds = float(seconds_str)

# Převeďte na základní desetinný tvar ve stupních
decimal_degrees = degrees + minutes / 60 + seconds / 3600

print(decimal_degrees)