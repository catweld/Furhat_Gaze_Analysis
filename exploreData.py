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

file = "video_log_labels/participant_video_csvs/9-23-2020-Furhat__5455.csv"
fileTimings = "video_log_labels/5455_gazetargets.txt"

#file = "video_log_labels/participant_video_csvs/9-18-2020-Furhat__4041.csv"
#fileTimings = "video_log_labels/4041_gazetargets.txt"

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
    yaw = float(angle[1])*math.pi/180
    roll = 0
    pitch = float(angle[2])*math.pi/180
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
        if angleResult[2]<0:
            print(angle)
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
otherLeft, tarOtherLeft = findPointsToTarget(anglesLeft[1:], timingsToTargetLeft, "Other")
#RobotR = [[-56,-53]]
#robotLeft = np.array([left[25,:], left[40,:]])#convertListAngleToListPoint(RobotR)
robotLeft, tarRobotLeft = findPointsToTarget(anglesLeft[1:], timingsToTargetLeft, "Robot")

#TabletR =[[-55,-51],[-54,-55], [-52,-54]]
#tabletLeft = np.array([left[159,:], left[715,:]])#convertListAngleToListPoint(TabletR)
tabletLeft, tarTabLeft = findPointsToTarget(anglesLeft[1:], timingsToTargetLeft, "Tablet")

right = convertListAngleToListPoint(anglesRight[1:])

#OtherR=[[-57,-51], [-46,-48], [-53,-41]]
#otherRight = np.array([right[26,:], right[337,:], right[393,:], right[734,:]])#convertListAngleToListPoint(OtherR)
otherRight, tarOtherRight = findPointsToTarget(anglesRight[1:], timingsToTargetRight, "Other")
#RobotR = [[-56,-53]]
#robotRight = np.array([right[25,:], right[40,:]])#convertListAngleToListPoint(RobotR)
robotRight, tarRobotRight = findPointsToTarget(anglesRight[1:], timingsToTargetRight, "Robot")

#TabletR =[[-55,-51],[-54,-55], [-52,-54]]
#tabletRight = np.array([right[159,:], right[393,:], right[715,:]])#convertListAngleToListPoint(TabletR)
tabletRight, tarTabRight = findPointsToTarget(anglesRight[1:], timingsToTargetRight, "Tablet")


# In[3]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(otherLeft[:,0], otherLeft[:,1], otherLeft[:,2], marker='o')
ax.scatter(tabletLeft[:,0], tabletLeft[:,1], tabletLeft[:,2], marker='x')
ax.scatter(robotLeft[:,0], robotLeft[:,1], robotLeft[:,2], marker='s')
time = np.array(anglesLeft[1:], float)[:,0]*(1/np.array(anglesLeft[1:], float)[np.array(anglesLeft[1:]).shape[0]-1,0])
ax.scatter(left[:,0], left[:,1], left[:,2], c=time, cmap=plt.cm.nipy_spectral, marker='^', alpha=0.005)
fig = plt.figure()
# plotting of labeled angles to get an overview on the distribution
plt.title("Left user labeled distribution of angle pairs")
plt.scatter(np.array(tarOtherLeft)[:,1], np.array(tarOtherLeft)[:,2], marker='o', label="Other")
plt.scatter(np.array(tarRobotLeft)[:,1], np.array(tarRobotLeft)[:,2], marker='s', label="Robot")
plt.scatter(np.array(tarTabLeft)[:,1], np.array(tarTabLeft)[:,2], marker='x', label="Tablet")
plt.legend()
fig = plt.figure()
# simple cutting of z axis to plot/project to 2D
plt.title("Left user labeled distribution of x- and y-values of rotated unit vectors")
plt.scatter(otherLeft[:,0], otherLeft[:,1], marker='o', label="Other")
plt.scatter(tabletLeft[:,0], tabletLeft[:,1], marker='s', label="Robot")
plt.scatter(robotLeft[:,0], robotLeft[:,1], marker='x', label="Tablet")
plt.legend()
#print(tarOtherLeft)


# In[4]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(otherRight[:,0], otherRight[:,1], otherRight[:,2], marker='o')
ax.scatter(tabletRight[:,0], tabletRight[:,1], tabletRight[:,2], color='red',marker='x')
ax.scatter(robotRight[:,0], robotRight[:,1], robotRight[:,2], marker='s')
time = np.array(anglesRight[1:], float)[:,0]*(1/np.array(anglesRight[1:], float)[np.array(anglesRight[1:]).shape[0]-1,0])
ax.scatter(right[:,0], right[:,1], right[:,2], c=time, cmap=plt.cm.nipy_spectral,marker='^', alpha=0.001)
fig = plt.figure()
# plotting of labeled angles to get an overview on the distribution
plt.title("Right user labeled distribution of angle pairs")
plt.scatter(np.array(tarOtherRight)[:,1], np.array(tarOtherRight)[:,2], marker='o',label="Other")
plt.scatter(np.array(tarRobotRight)[:,1], np.array(tarRobotRight)[:,2], marker='s', label="Robot")
plt.scatter(np.array(tarTabRight)[:,1], np.array(tarTabRight)[:,2], marker='x', label="Tablet")
plt.legend()
fig = plt.figure()
# simple cutting of z axis to plot/project to 2D
plt.title("Right user labeled distribution of x- and y-values of rotated unit vectors")
plt.scatter(otherRight[:,0], otherRight[:,1],marker='o', label="Other")
plt.scatter(tabletRight[:,0], tabletRight[:,1],marker='s', label="Robot")
plt.scatter(robotRight[:,0], robotRight[:,1],marker='x',label="Tablet")
plt.legend()


# In[5]:


### SVM training below, first data preparation


# different cells allow for trying different inputs to the SVMS, execute only one for testing purposes
########## 3D version that does not work :D ####################
X = np.append(otherLeft, robotLeft,0)
X = np.append(X, tabletLeft,0)
otherClass = np.zeros(otherLeft.shape[0])
y = np.append(otherClass, np.ones(robotLeft.shape[0]),0)
y = np.append(y, np.ones(tabletLeft.shape[0])*2, 0)
print(X.shape, otherLeft.shape, robotLeft.shape, tabletLeft.shape)
print(y.shape)
########## End ---- 3D version that does not work :D ####################


# In[6]:


########## Pure angles in degrees as extracted from head pose analysis system ####################
X = np.append(np.array(tarOtherLeft, float)[:,1:], np.array(tarRobotLeft, float)[:,1:],0)
X = np.append(X, np.array(tarTabLeft, float)[:,1:],0)
X = X.reshape(X.shape[0:2])
otherClass = np.zeros(otherLeft.shape[0])
y = np.append(otherClass, np.ones(robotLeft.shape[0]),0)
y = np.append(y, np.ones(tabletLeft.shape[0])*2, 0)
print(X.shape, otherLeft.shape, robotLeft.shape, tabletLeft.shape)
print(y.shape)
########## End ---- Pure angles in degrees as extracted from head pose analysis system ####################


# In[7]:


########## Cutting z - 2D projection for left user ####################
X = np.append(otherLeft[:,0:2], robotLeft[:,0:2],0)
X = np.append(X, tabletLeft[:,0:2],0)
X = X.reshape(X.shape[0:2])
otherClass = np.zeros(otherLeft.shape[0])
y = np.append(otherClass, np.ones(robotLeft.shape[0]),0)
y = np.append(y, np.ones(tabletLeft.shape[0])*2, 0)
print(X.shape, otherLeft.shape, robotLeft.shape, tabletLeft.shape)
print(y.shape)
########## End ---- Cutting z - 2D projection for left user ####################


# In[8]:


########## Cutting z - 2D projection for right user ####################
X = np.append(otherRight[:,0:2], robotRight[:,0:2],0)
X = np.append(X, tabletRight[:,0:2],0)
X = X.reshape(X.shape[0:2])
otherClass = np.zeros(otherRight.shape[0])
y = np.append(otherClass, np.ones(robotRight.shape[0]),0)
y = np.append(y, np.ones(tabletRight.shape[0])*2, 0)
print(X.shape, otherRight.shape, robotRight.shape, tabletRight.shape)
print(y.shape)
########## End ---- Cutting z - 2D projection for right user ####################


# In[9]:


# split train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


# In[10]:


# code in the next 2 cells from SVM python tutorial on the iris dataset, modfications for models

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import classification_report, confusion_matrix

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 0.1, x.max() + 0.1
    y_min, y_max = y.min() - 0.1, y.max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# In[11]:


# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.5  # SVM regularization parameter, Sarah: decided to try different ones in the different models
models = (svm.SVC(kernel='poly', degree=3, gamma=10, C=1.5),#,
          svm.SVC(kernel='poly', degree=5, gamma=10, C=2),
          svm.SVC(kernel='rbf', gamma=10, C=1.5),
          svm.SVC(kernel='rbf', gamma=2, C=1.5))
         #svm.SVC(kernel='poly', degree=10, gamma='auto', C=C))
models = (clf.fit(X_train, y_train) for clf in models)

# title for the plots
titles = ('SVC with polynomial (degree 3) kernel, \n gamma=10, C=1.5', #'LinearSVC (linear kernel) no C')#,
          'SVC with polynomial (degree 5) kernel, \n gamma=10, C=2',
          'SVC with RBF kernel \n gamma=10, C=1.5',
          'SVC with RBF kernel,\n gamma=2, C=1.5')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X_train[:, 0], X_train[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    y_pred = clf.predict(X_test)
    print(title)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("Score on test dataset: ", clf.score(X_test, y_test), "\n")
    ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    #ax.set_xlim(xx.min(), xx.max())
    #ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()
plt.savefig("svm_right_5455_2.png",dpi=150)


# In[108]:


# unused so far
#f = open("leftAnnotationsSVM5455.txt", "w")

#for angles in anglesLeft:
#    tier = "left"
#    clf = modesl[2]
#    classPr = clf.predict([np.array(angles[1:])])
#    f.write()
#f.close


# In[12]:


# to get the overview on the right user, exchange anglesLeft to anglesRight
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

#leftDiff = np.array([0 if np.linalg.norm(a-b)>0.2 else np.linalg.norm(a-b) for a,b in zip(right[:-1,:], right[1:,:])], 'float')
#leftDiff = np.array([np.linalg.norm(a-b) for a,b in zip(left[:-1,:], left[1:,:])], 'float')
leftAngleDiff = np.array([np.linalg.norm(a-b) for a,b in zip(np.array(anglesLeft[1:], 'float')[:-1,1:], np.array(anglesLeft[1:], 'float')[1:,1:])], 'float')

#mov_average = moving_average(leftDiff, 10)
#plt.figure()
#plt.plot(np.arange(mov_average.shape[0]), mov_average)
#plt.ylim(top=0.3)
plt.figure()
plt.plot(np.arange(leftAngleDiff.shape[0]), leftAngleDiff)
plt.title("5455 left Euclidean Angle Diff")
plt.savefig("5455_left_angle_diff.png", dpi=150)


# In[164]:


# some tries to correct for noise but doesn't work
leftAngleDiff = np.array([np.linalg.norm(a-b) for a,b in zip(np.array(anglesLeft[1:], 'float')[:-1,1:], np.array(anglesLeft[1:], 'float')[1:,1:])], 'float')
anglesToChange = np.array(anglesLeft[1:], 'float')[:, 1:]
for i in range(leftAngleDiff.shape[0]):
    if leftAngleDiff[i] > 30:
        anglesToChange[i+1] = anglesToChange[i+1]*-1.0
        if anglesToChange.shape[0] > i+3:
            leftAngleDiff[i+1] = np.linalg.norm(anglesToChange[i+1]-anglesToChange[i+2])
for i in range(leftAngleDiff.shape[0]):
    if leftAngleDiff[i] > 30:
        anglesToChange[i+1][0] = anglesToChange[i+1][0]*-1.0
        if anglesToChange.shape[0] > i+3:
            leftAngleDiff[i+1] = np.linalg.norm(anglesToChange[i+1]-anglesToChange[i+2])
for i in range(leftAngleDiff.shape[0]):
    if leftAngleDiff[i] > 30:
        anglesToChange[i+1][1] = anglesToChange[i+1][1]*-1.0
        if anglesToChange.shape[0] > i+3:
            leftAngleDiff[i+1] = np.linalg.norm(anglesToChange[i+1]-anglesToChange[i+2])
leftAngleDiff = np.array([np.linalg.norm(a-b) for a,b in zip(anglesToChange[:-1,:], anglesToChange[1:,:])])
plt.plot(np.arange(leftAngleDiff.shape[0]), leftAngleDiff)
plt.title("5455 left Euclidean Angle Diff")
plt.savefig("5455_left_angle_diff_corrected.png", dpi=150)


# In[149]:


print(leftAngleDiff.shape)
print(anglesToChange.shape)


# In[161]:


plt.figure()
plt.plot(np.arange(leftAngleDiff.shape[0]), leftAngleDiff)
plt.title("5455 left Euclidean Angle Diff")


# In[ ]:




