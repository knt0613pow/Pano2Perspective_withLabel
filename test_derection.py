import math

def calcBearing (lat1, long1, lat2, long2):
    dLon = (long2 - long1)
    x = math.cos(math.radians(lat2)) * math.sin(math.radians(dLon))
    y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(dLon))
    bearing = math.atan2(x,y)   # use atan2 to determine the quadrant
    bearing = math.degrees(bearing)

    return bearing %360

def calcNSEW(lat1, long1, lat2, long2):
    # points = ["north", "north east", "east", "south east", "south", "south west", "west", "north west"]
    bearing = calcBearing(lat1, long1, lat2, long2)
    # bearing += 22.5
    bearing = bearing % 360
    # bearing = int(bearing / 45) # values 0 to 7
    # NSEW = points [bearing]

    return  bearing

# White house 38.8977° N, 77.0365° W
lat1 = 38.8976763
long1 = -77.0365298
# Lincoln memorial 38.8893° N, 77.0506° W
lat2 = 38.9076763
long2 = -77.0365298

points = calcNSEW(lat1, long1, lat2, long2)
# print ("The Lincoln memorial is " + points + " of the White House")
print(points)
print ("Actually bearing of 231.88 degrees")

print ("Output says: The Lincoln memorial is south west of the White House ")