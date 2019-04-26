import math
import geopy.distance
from geopy import Point
import numpy as np
import pandas as pd

#Functions to Determine locations of wellbore trajectories and project new points

def get_bearing(lat1,lon1,lat2,lon2):
    dLon = lon2 - lon1;
    y = math.sin(dLon) * math.cos(lat2);
    x = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dLon);
    brng = np.rad2deg(math.atan2(y, x));
    if brng < 0: brng+= 360
    return brng

def get_heel(row):
    '''Calculate the coordinates of the heel of the well using surface and bottomhole locations'''
    surf_lat = row[1]['SURFACE_LATITUDE']
    surf_lon = row[1]['SURFACE_LONGITUDE']
    bot_lat = row[1]['BOTTOM_HOLE_LATITUDE']
    bot_lon = row[1]['BOTTOM_HOLE_LONGITUDE']
    lateral_length = row[1]['TREATABLE_LENGTH']
    
    #If bottom hole is nan assign tophole coordinates
    if math.isnan(bot_lat) | math.isnan(bot_lon):
        bot_lat = surf_lat
        bot_lon = surf_lon
    
    #If lateral length doesn't exist, make it 1 foot long
    if math.isnan(lateral_length):
        lateral_length = 1
    
    distance = geopy.distance.distance((surf_lat, surf_lon), (bot_lat, bot_lon)).ft
    
    if lateral_length**2 > distance**2:
        distance = lateral_length
    
    c = math.sqrt(distance**2 - lateral_length**2)
    
    if math.isnan(c):
        c=1

    angle_a = math.degrees(math.acos(c / distance))
    angle_b = math.degrees(math.acos( lateral_length / distance))

    bearing = angle_a + get_bearing(surf_lat, surf_lon, bot_lat, bot_lon)
    
    if math.isnan(bearing):
        bearing = 0

    if not ((math.isnan(surf_lat)) & (math.isnan(surf_lon))):
        heel_lat, heel_lon, _ = geopy.distance.distance(feet=c).destination(Point(surf_lat, surf_lon), bearing)
    else:
        heel_lat, heel_lon = bot_lat, bot_lon
    
    return (heel_lat, heel_lon)

def get_midpoint(row):
    bot_lat = row[1]['BOTTOM_HOLE_LATITUDE']
    bot_lon = row[1]['BOTTOM_HOLE_LONGITUDE']
    heel_lat = row[1]['heel_lat']
    heel_lon = row[1]['heel_lon']
    
    if math.isnan(bot_lat) | math.isnan(bot_lon):
        bot_lat = row[1]['SURFACE_LATITUDE']
        bot_lon = row[1]['SURFACE_LONGITUDE']
    
    midpoint_lat = (heel_lat + bot_lat)/2
    midpoint_lon = (heel_lon + bot_lon)/2
    return (midpoint_lat, midpoint_lon)

def DistanceToOthers(df, currentWellMidLat, currentWellMidLon):
    """Take a dataframe and a row number: currentwell.  Return the list of distances to all other wells"""
    distanceList = []
    
    #currentWellMidLat = df['mid_lat'][currentwell]
    #currentWellMidLon = df['mid_lon'][currentwell]
    
    for i in range(df.shape[0]):
        distance = geopy.distance.distance((currentWellMidLat, currentWellMidLon),
                                           (df['mid_lat'][i], df['mid_lon'][i])).ft
        distanceList.append(distance)

    return distanceList

def IsMissing(currentValue):
    #Check if the current value to be filled in must be NaT, NaN, or None
    if (np.isnan(currentValue)) | ((currentValue is None)) | (pd.isnull(currentValue)):
        return True
    else:
        return False

def FillValues(df, varName):
    """Take a dataframe and a particular variable name.  Return a list of interpolated values based off of distance."""
    valueList = []
    for row in range(df.shape[0]):
        currentValue = df[varName][row]
        
        if IsMissing(currentValue):
            #Calculate the distances to all the other properties
            #But we only want to pass in non-null values
            currentWellMidLat = df['mid_lat'][row]
            currentWellMidLon = df['mid_lon'][row]
            tempDF = df[~df[varName].apply(IsMissing)].reset_index()
            
            distanceList = DistanceToOthers(tempDF, currentWellMidLat, currentWellMidLon)
            tempDF['DistanceToOthers'] = distanceList
            
            #Look only at not missing values then sort by distance and take the first value
            #currentValue = df[~df[varName].apply(IsMissing)].sort_values(by=['DistanceToOthers']).reset_index()[varName][0]
            currentValue = tempDF.sort_values(by=['DistanceToOthers']).reset_index()[varName][0]
            
        valueList.append(currentValue)
    
    return valueList