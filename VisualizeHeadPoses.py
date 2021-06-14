#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import csv
from scipy.spatial.transform import Rotation as R
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math

#file = "video_log_labels/participant_video_csvs/9-23-2020-Furhat__5455.csv"
#fileTimings = "video_log_labels/5455_gazetargets.txt"

file = "video_log_labels/participant_video_csvs/9-18-2020-Furhat__4041.csv"
fileTimings = "video_log_labels/4041_gazetargets.txt"

anglesLeft = []
anglesRight = []

with open(file, newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
     for row in spamreader:
         anglesLeft.append([row[i] for i in [1,2,3]])
         anglesRight.append([row[i] for i in [1,4,5]])

timingsToTargetLeft = []
timingsToTargetRight = []
with open(fileTimings, newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
     for row in spamreader:
        print(row)
        if row[0].find("Left") != -1:
            timingsToTargetLeft.append([row[i] for i in [3,6,11]])
        else:
            timingsToTargetRight.append([row[i] for i in [3,6,11]])


# In[2]:


def angleToPointOnCircle(angle, radius=1, origin=[0,0]):
    #r = R.from_euler('zyx', [
    #[0, 0, angle[0]],
    #[0, angle[1], 0],
    #[0, 0, 0]], degrees=True)
    yaw = float(angle[1])*0.0174533
    roll = 0
    pitch = float(angle[2])*0.0174533
    yawMatrix = np.matrix([
    [math.cos(yaw), -math.sin(yaw), 0],
    [math.sin(yaw), math.cos(yaw), 0],
    [0, 0, 1]
    ])

    pitchMatrix = np.matrix([
    [math.cos(pitch), 0, math.sin(pitch)],
    [0, 1, 0],
    [-math.sin(pitch), 0, math.cos(pitch)]
    ])

    rollMatrix = np.matrix([
    [1, 0, 0],
    [0, math.cos(roll), -math.sin(roll)],
    [0, math.sin(roll), math.cos(roll)]
    ])

    R = yawMatrix * pitchMatrix * rollMatrix
    vector = np.array([[radius], [0], [0]])
    return R*vector

def convertListAngleToListPoint(angles):
    angleToPoint = []
    for angle in angles:
        angleResult = angleToPointOnCircle(angle)
        angleToPoint.append(angleResult)
    return np.array(angleToPoint)

def findPointsToTarget(angles, listOfTargets, target):
    selectedTargets = [elem for elem in timingsToTargetLeft if elem[2]==target]
    listOfTargetAngles = []
    for elem in selectedTargets:
        for angle in angles:
            if float(angle[0]) >= float(elem[0]) and             float(angle[0]) <= float(elem[1]):
               listOfTargetAngles.append(angle)
    return convertListAngleToListPoint(listOfTargetAngles), listOfTargetAngles

left = convertListAngleToListPoint(anglesLeft[1:])
#otherLeft = np.array([left[27,:], left[709,:], left[718,:]])#convertListAngleToListPoint(OtherR)
otherLeft, targetAng = findPointsToTarget(anglesLeft[1:], timingsToTargetLeft, "Other")
#RobotR = [[-56,-53]]
#robotLeft = np.array([left[25,:], left[40,:]])#convertListAngleToListPoint(RobotR)
robotLeft, targetAng = findPointsToTarget(anglesLeft[1:], timingsToTargetLeft, "Robot")

#TabletR =[[-55,-51],[-54,-55], [-52,-54]]
#tabletLeft = np.array([left[159,:], left[715,:]])#convertListAngleToListPoint(TabletR)
tabletLeft, targetAng = findPointsToTarget(anglesLeft[1:], timingsToTargetLeft, "Tablet")

right = convertListAngleToListPoint(anglesRight[1:])

#OtherR=[[-57,-51], [-46,-48], [-53,-41]]
#otherRight = np.array([right[26,:], right[337,:], right[393,:], right[734,:]])#convertListAngleToListPoint(OtherR)
otherRight, tar = findPointsToTarget(anglesRight[1:], timingsToTargetRight, "Other")
print(otherRight)
#RobotR = [[-56,-53]]
#robotRight = np.array([right[25,:], right[40,:]])#convertListAngleToListPoint(RobotR)
robotRight, tar = findPointsToTarget(anglesRight[1:], timingsToTargetRight, "Robot")

#TabletR =[[-55,-51],[-54,-55], [-52,-54]]
#tabletRight = np.array([right[159,:], right[393,:], right[715,:]])#convertListAngleToListPoint(TabletR)
tabletRight, tar = findPointsToTarget(anglesRight[1:], timingsToTargetRight, "Tablet")


# In[5]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(otherLeft[:,0], otherLeft[:,1], otherLeft[:,2], marker='o')
ax.scatter(tabletLeft[:,0], tabletLeft[:,1], tabletLeft[:,2], marker='x')
ax.scatter(robotLeft[:,0], robotLeft[:,1], robotLeft[:,2], marker='s')
ax.scatter(left[:,0], left[:,1], left[:,2], color='gray',marker='^', alpha=0.005)


# In[6]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(otherRight[:,0], otherRight[:,1], otherRight[:,2], marker='o')
ax.scatter(tabletRight[:,0], tabletRight[:,1], tabletRight[:,2], color='red',marker='x')
ax.scatter(robotRight[:,0], robotRight[:,1], robotRight[:,2], marker='s')
ax.scatter(right[:,0], right[:,1], right[:,2], color='gray',marker='^', alpha=0.01)


# In[ ]:




