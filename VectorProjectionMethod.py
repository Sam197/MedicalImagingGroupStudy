'''
Final code made for the Vector Projection Method of image recon. This is not an full script from start to end of image recon, but
represents the final step of taken processed data, projecting the data as cones, and finding the intersection of them. This was also not
the only method, with one of my lab partners creating the meshgrid method. This is also not the only method prototyped, developed and tested
on expirmental and monte carlo data by me, but it is the most advanced and complete, and the only one of my methods expolored in the final
report. I also aplogise for some code gore you may encounter when flicking through (Namely the many, many, commented out lines of code). While
I have no amazing excuse for this, my excuse will be the time crunch we were under to get some code that worked and spent little time on any form
of front end and keeping our code clean. With that aside,many thanks go the the image recontruction team, consisting of Heather and James who 
worked on energy calibration, conicidence finding and general data cleaning, and Sinéad for her work on the meshgrid method.
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from enum import Enum
from tqdm import tqdm
from scipy.interpolate import interp1d
import pandas as pd
from time import time
import pyvista as pv
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog

class Axes(Enum):
    x = 0
    y = 1
    z = 2

class Cone:
    '''
    The class that defines each cone and provides intersections logic and helper functions such as normalise and draw
    '''

    NUM_OF_ANGLES = 100
    THRESH = 1000
    CLOSE_CUTTOFF = 5

    @staticmethod
    def normalise(vector):
        return vector / np.linalg.norm(vector)
    
    def makeVectors(self):
        '''
        Creates the unit vectors that define each cone. Each vector makes an angle equal to the opening angle rotated around 2pi
        '''
        if self.direc[0] != 1:
            vec = [1, 0, 0]
        else:
            vec = [0, 0, 1]
        
        perp_u = Cone.normalise(np.cross(self.direc, vec))
        perp_v = Cone.normalise(np.cross(self.direc, perp_u))

        phis = np.linspace(0, 2*np.pi, self.NUM_OF_ANGLES)
        vecs = [np.cos(self.openingAngle)*self.direc + np.sin(self.openingAngle)*(np.cos(phi)*perp_u + np.sin(phi)*perp_v) for phi in phis]
        return np.array(vecs)

    def __init__(self, vertex, direc, angle):
        self.vertex = vertex
        self.direc = Cone.normalise(direc)
        self.openingAngle = angle
        self.vectors = self.makeVectors()
        
    def projectToPlane(self, axis, plane):
        '''
        Projects each vector to a given plane using simple vector logic
        '''   
        if axis == Axes.x:
            samplePlane = 0
        elif axis == Axes.y:
            samplePlane = 1
        elif axis == Axes.z:
            samplePlane = 2
        coords = []
        for vec in self.vectors:
            if vec[samplePlane] == 0:
                continue
            llambda = (plane - self.vertex[samplePlane])/vec[samplePlane]
            if llambda < 0: #Means the intersection point is behind the cone - not physical
                continue
            x = self.vertex[0] + llambda*vec[0]
            y = self.vertex[1] + llambda*vec[1]
            z = self.vertex[2] + llambda*vec[2]
            coord = np.array([x,y,z])
            if np.max(coord) < self.THRESH: #If the intersection is very far off, it's not useful
                coords.append(np.array([x,y,z]))
        
        return np.array(coords)

    def draw(self, ax, lengthy = 1, lineDensity = 1):
        '''
        Draws a representation of each cone to a 3D plot. Does not draw all vectors as to make the plot more readable
        '''
        ax.quiver(*self.vertex, self.direc[0], self.direc[1], self.direc[2], color='r', linewidth=2, label="Centerline", length = lengthy)

        # Plot cone vectors
        numOfAngles = int(self.NUM_OF_ANGLES*lineDensity)
        skip = int(1/lineDensity)

        ax.quiver(np.full(numOfAngles, self.vertex[0]), np.full(numOfAngles, self.vertex[1]), np.full(numOfAngles, self.vertex[2]), 
                self.vectors[:,0][::skip], self.vectors[:,1][::skip], self.vectors[:,2][::skip], color='b', alpha=0.5, length = lengthy)
    
def makeData():
    '''
    Makes dummy data, was useful for prototyping and bug checking, but was not used in later stages of development when
    there was ample expirmental/monte carlo data to be used
    '''
    SOURCE_POS = np.array([[0,0,5], [0,0,5]])
    SCATTERERS = np.array([[-1,-1,0], [-1,1,0], [1,-1,0], [1,1,0]])
    ABSORBERS = np.array([[0,0,-5], [0,0,-5]])

    NUMBER_OF_CONES = 100

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    so = np.array(SOURCE_POS)
    sc = np.array(SCATTERERS)
    ab = np.array(ABSORBERS)
    ax.scatter(sc[:,0], sc[:,1], sc[:,2], color = 'red', label = 'Scatterers')
    ax.scatter(ab[:,0], ab[:,1], ab[:,2], color = 'black', label = 'Absorbers')
    ax.scatter(SOURCE_POS[:,0], SOURCE_POS[:,1], SOURCE_POS[:,2], color = 'yellow', label = 'Source')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    ax.legend()

    plt.show()

    data = []

    for i in range(NUMBER_OF_CONES):
        so = SOURCE_POS[np.random.randint(len(SOURCE_POS))]
        sc = SCATTERERS[np.random.randint(len(SCATTERERS))]
        ab = ABSORBERS[np.random.randint(len(ABSORBERS))]
        contructLine = sc-ab
        n = np.dot(contructLine, so-sc)
        d = np.linalg.norm(contructLine)*np.linalg.norm(so-sc)
        openAngle = np.arccos(n/d)
        openAngle += (np.random.rand())/1
        data.append(Cone(sc, contructLine, openAngle))

    return data

def findAngle(scatterer, absorber):
    '''
    Finds the opening angle of a cone using the energies deposited in the scatterer and absorber and the compton formula
    Input energies must be in keV - Geant4 is MeV natively
    '''
    meC = 511
    numer = meC*scatterer
    denom = absorber*(scatterer+absorber)
    theta = 1 - (numer/denom)
    return np.arccos(theta)

def dectorPos(detector):
    '''
    Collection of detecotr positions from different datasets (units are cm)
    '''
    #Some Actual Data, from the 11th Feb
    #SOURCE: [0,0,0]
    # detecPos = {1: [15,7,7], 2: [15,7,-7], 3: [15,-7,-7], 4: [15,-7,7],
    #             5: [40, 0, 5.75], 6: [40,5.75,0], 7: [40,0,-5.75], 8: [40,-5.75,0]}

    #Monte 20/2/25
    #SOURCE: [0,0,40]
    # detecPos = {1: [7,7,25], 2: [7,-7,25], 3: [-7,7,25], 4: [-7,-7,25],
    #             5: [0,-5.75,-40], 6: [0, 5.75,-40], 7: [5.75,0,40], 8: [-5.75,0,-40]}

    #Monte MoveSource
    #SOURCE: [36,0,0]
    # detecPos = {1:[25,7,7], 2:[25,7,-7], 3:[25,-7,-7], 4:[25,-7,7],
    #             5:[-40,0,5.75], 6:[-40,5.75,0], 7:[-40,0,5.75], 8:[-40,-5.75,0]}

    #Expirmental 20.02 15cm SOURCE: (0,0,0)
    # detecPos = {'1A':[15,26.5,26.5], '1B':[15,26.5,-26.5], '1E':[15,-26.5,-26.5], '1D':[15,-26.5,26.5],
    #             '3A':[80,0,5.75], '3B':[80,5.75,0], '3E':[80,0,-5.75], '3D':[80,-5.75,0]}

    #Expirmental 18/02 13cm SOURCE: (2,0,0)
    # detecPos = {'1A':[15,7,7], '1B':[15,7,-7], '1E':[15,-7,-7], '1D':[15,-7,7],
    #             '3A':[80,0,5.75], '3B':[80,5.75,0], '3E':[80,0,-5.75], '3D':[80,-5.75,0]}

    #Monte Carlo Source Offsets SOURCE: (-10,4,8)
    # detecPos = {1:[15,26.5,26.5], 2:[15,26.5,-26.5], 3:[15,-26.5,-26.5], 4:[15,-26.5,26.5],
    #             5:[80,0,5.75], 6:[80,5.75,0], 7:[80,0,5.75], 8:[80,-5.75,0]}

    #Monte Carlo MoveSource SOURCE: (6,0,0)
    # detecPos = {1:[15,7,7], 2:[15,7,-7], 3:[15,-7,-7], 4:[15,-7,7],
    #             5:[80,0,5.75], 6:[80,5.75,0], 7:[80,0,-5.75], 8:[80,-5.75,0]}

    #Monte Carlo Scat Out SOURCE: (-20,0,0)/(-10,0,0)
    # detecPos = {1:[15,26.5,26.5], 2:[15,26.5,-26.5], 3:[15,-26.5,-26.5], 4:[15,-26.5,26.5],
    #             5:[80,0,5.75], 6:[80,5.75,0], 7:[80,0,-5.75], 8:[80,-5.75,0]}

    #Expirmental 20.02 Long Run 65cm SOURCE: (-50,0,0)
    # detecPos = {'1A':[15,26.5,26.5], '1B':[15,26.5,-26.5], '1E':[15,-26.5,-26.5], '1D':[15,-26.5,26.5],
    #             '3A':[80,0,5.75], '3B':[80,5.75,0], '3E':[80,0,-5.75], '3D':[80,-5.75,0]}

    #Monte ScatOut AbsorbOut 20cm SOURCE: (-10,0,0)
    # detecPos = {1:[15,26.5,26.5], 2:[15,26.5,-26.5], 3:[15,-26.5,-26.5], 4:[15,-26.5,26.5],
    #             5:[80,0,10], 6:[80,10,0], 7:[80,0,-10], 8:[80,-10,0]}

    #Expirmental 25.02 30cm SOURCE: (-15,0,0)
    # detecPos = {'1A':[15,15,15], '1B':[15,15,-15], '1E':[15,-15,-15], '1D':[15,-15,15],
    #             '3A':[80,0,5.75], '3B':[80,5.75,0], '3E':[80,0,-5.75], '3D':[80,-5.75,0]}

    #Monte MoveScatters SOURCE: (-10,0,0)
    detecPos = {1:[15,15,15], 2:[15,15,-15], 3:[15,-15,-15], 4:[15,-15,15],
                5:[80,0,5.75], 6:[80,5.75,0], 7:[80,0,-5.75], 8:[80,-5.75,0]}

    #Monte Source Offset: SOURCE: (6,2,8)
    # detecPos = {1:[15,7,7], 2:[15,7,-7], 3:[15,-7,-7], 4:[15,-7,7],
    #             5:[80,0,5.75], 6:[80,5.75,0], 7:[80,0,-5.75], 8:[80,-5.75,0]}

    #Monte Sphere SOURCE: (0,0,0)
    # detecPos = {1:[5,0,5], 2:[-5,0,5], 3:[-5,0,-5], 4:[5,0,-5],
    #             5:[-14,0,0], 6:[0,0,-14], 7:[0,0,14], 8:[14,0,0]}

    #Expirmental run 8 30cm SOURCE: (-15,0,0)
    # detecPos = {1:[15,15,15], 2:[15,15,-15], 3:[15,-15,-15], 4:[15,-15,15],
    #             5:[80,0,5.75], 6:[80,5.75,0], 7:[80,0,-5.75], 8:[80,-5.75,0]}

    #Monte Spherical Further out
    # detecPos = {1:[15,0,15], 2:[-15,0,15], 3:[-15,0,-15], 4:[15,0,-15],
    #             5:[40,0,0], 6:[0,0,-40], 7:[0,0,40], 8:[40,0,0]}

    #Expirmental Run 9 SOURCE (-15,-15,10)
    # detecPos = {1:[15,15,15], 2:[15,15,-15], 3:[15,-15,-15], 4:[15,-15,15],
    #             5:[80,0,5.75], 6:[80,5.75,0], 7:[80,0,-5.75], 8:[80,-5.75,0]}

    return detecPos[detector]

def findContructVec(detecPair):
    '''
    Finds the contruction vector, or centre line vector, of a cone based on the scatter and absober positons
    '''
    d1, d2 = detecPair
    d1 = np.array(dectorPos(d1))
    d2 = np.array(dectorPos(d2))
    cv = d1-d2
    return cv/np.linalg.norm(cv)

def loadBGO():
    '''
    Dedicated code to load data taken from the pixalised BGO data, as the data was saved in a different format, and the
    positons were not as static as the standard NaI setup - due to the pixalised nature of the detector. The function loads
    the data and outputs the data as list of cones.
    '''
    detecs = {1:[15,15,15], 2:[15,15,-15], 3:[15,-15,-15], 4:[15,-15,15]}
    absorb = np.array([81.5,0,0])

    da = pd.read_csv('5Coincident_Events_CHx_vs_3X.csv')
    e1 = da['Energy_A'].to_list()
    e2 = da['Energy_B'].to_list()
    sacatter = da['Detector_A'].to_list()
    yoffset = da['y'].to_list()
    zoffset = da['z'].to_list()


    absorbs = list(zip(yoffset, zoffset))
    contructVecs = []
    for i, sc in enumerate(sacatter):
        contructVecs.append(np.array(detecs[sc])-np.array([81.5, absorbs[i][0], absorbs[i][1]]))

    angles = []
    for e1, e2 in zip(e1, e2):
        #e1, e2 = e1*1000, e2*1000 #Needed is energy is given in MeV not keV
        angle = findAngle(min(e1,e2), max(e1,e2))
        angles.append(angle)

    data = []
    print("Coneing", end='\r')
    for i, sc in enumerate(sacatter):
        data.append(Cone(np.array(detecs[sc]), contructVecs[i], angles[i]))
    
    return data        

def loadSourceCentre():
    '''
    A monte carlo simulation was run to calcuate the effective centres of each detector when paired with another detector. Below are those effective
    centres, with each scatterer 1-4 paired with each absorber 5-8. Func loads normal expirmental data, but uses the effective centres rather than
    the dector centres as seen before. Outputs data as a list of cones
    '''

    dpairs = {15: [(15.02078728255528, 14.888215370188371, 14.920448493857496), (78.749785428115, 0.2610921821086262, 5.804166754792332)],
            16: [(14.997602826688363, 14.87177091131001, 14.873438143205858), (78.81880090119522, 5.884045456573705, 0.23876771075697206),],
            17: [(15.022814326810177, 14.89972267808219, 14.891391675146773), (78.49749727289895, 0.1039076610009443, -5.515752634560906)],
            18: [(15.02847520085929, 14.897508148227711, 14.868800116004298), (78.60954419371728, -5.375209683769634, 0.08577522198952879)],
            25: [(15.023217889230768, 14.816911595897437, -14.866851736410258), (78.6204693221344, 0.20911284881422926, 5.405660449604743)],
            26: [(15.051148741092636, 14.889364760095013, -14.905874392715756), (78.76239321089494, 5.8078520054474705, -0.3391411571984436)],
            27: [(15.007173139041635, 14.869062009426552, -14.905848614296936), (78.78306433359013, 0.22592650693374422, -5.835947453775039)],
            28: [(15.01833565677547, 14.878998762611277, -14.877246658753709), (78.56955806340058, -5.526360715658021, -0.09117555907780978)],
            35: [(14.99061751863354, -14.860803308488611, -14.930386267080744), (78.53159666767371, -0.1819805679758308, 5.380175705941592)],
            36: [(14.992751461306533, -14.876488209045228, -14.860346532663316), (78.53448414636543, 5.400013761296661, 0.04926254715127702)],
            37: [(15.036816693813627, -14.91796180579483, -14.905449506656227), (78.79244221084797, -0.20917002368220017, -5.817573659281894)],
            38: [(15.000891015313936, -14.906218580398164, -14.895289996937214), (78.70110349214659, -5.86294405684368, -0.2539456050860135)],
            45: [(14.996704993311038, -14.901256904682272, 14.888051574414717), (78.63209813327883, -0.3354810547833197, 5.884731287816844)],
            46: [(14.976903138624339, -14.88027662222222, 14.871304935449734), (78.69873386749482, 5.433350482401656, 0.05521261076604555)],
            47: [(14.990245393034824, -14.850434044776119, 14.870748968159203), (78.54459935185186, -0.11496367251461986, -5.437725030214425)],
            48: [(14.960316428794991, -14.920448585289517, 14.894217949139279), (78.70493625019304, -5.876838787644788, 0.16156916061776064)]}

    da = pd.read_csv('Energy Coincidence 30cm 137Cs (25.02) 2.csv')
    e1 = da['Energy_A'].to_list()
    e2 = da['Energy_B'].to_list()
    d1 = da['Detector_A'].to_list()
    d2 = da['Detector_B'].to_list()
    detecPair = list(zip(d1, d2))
    detecPair = [int(str(p[0])+str(p[1])) for p in detecPair]
    contrucVecs = np.array([np.array(dpairs[pair][0])-np.array(dpairs[pair][1]) for pair in detecPair])
    contrucVecs /= np.linalg.norm(contrucVecs)
    scatter = [np.array(dpairs[pair][0]) for pair in detecPair]

    angles = []
    for e1, e2 in zip(e1, e2):
        #e1, e2 = e1*1000, e2*1000
        angle = findAngle(min(e1,e2), max(e1,e2))
        angles.append(angle)
    
    data = []
    print("Coneing", end='\r')
    for i, detectors in enumerate(detecPair):
        #data.append(Cone(dectorPos(int(detectors[9])), contrucVecs[i], angles[i]))
        data.append(Cone(scatter[i], contrucVecs[i], angles[i]))
    
    return data

def loadExact():
    '''
    Loads monte carlo data that gives the exact postions of photon interactions in the detectors.
    '''
    da = pd.read_csv('(-10,0,0) RUN 8 good event positions.csv')
    e1 = da['Energy 1'].to_list()
    e2 = da['Energy 2'].to_list()
    x1, y1, z1 = da['x1'].to_list(), da['y1'].to_list(), da['z1'].to_list()
    x2, y2, z2 = da['x2'].to_list(), da['y2'].to_list(), da['z2'].to_list()

    # da = pd.read_csv('MC _3_18_16_1.csv')  #Needed to load legacy data - due to the slight differences in column names
    # e1 = da['E_A'].to_list()
    # e2 = da['E_B'].to_list()
    # x1, y1, z1 = da['x_A'].to_list(), da['y_A'].to_list(), da['z_A'].to_list()
    # x2, y2, z2 = da['x_B'].to_list(), da['y_B'].to_list(), da['z_B'].to_list()

    scatterer = np.column_stack((x1, y1, z1))
    aborber = np.column_stack((x2, y2, z2))
    detecPair = list(zip(scatterer, aborber))

    angles = []
    for e1, e2 in zip(e1, e2):
        #e1, e2 = e1*1000, e2*1000
        angle = findAngle(min(e1,e2), max(e1,e2))
        angles.append(angle)

    contrucVecs = [(pair[0]-pair[1])/np.linalg.norm(pair[0]-pair[1]) for pair in detecPair]
    data = []
    print("Coneing", end='\r')
    for i, detectors in enumerate(detecPair):
        #data.append(Cone(dectorPos(int(detectors[9])), contrucVecs[i], angles[i]))
        data.append(Cone(detectors[0], contrucVecs[i], angles[i]))
    
    return data

def loadData():
    '''
    The most commonly used function to load data, using the detector centres, not effective or exact. Contains
    support for three different dataframe formats used in the project. Loads data, then returns a list of cones 
    '''

    print("Reading", end='\r')

    #Format for Monte Carlo Data
    da = pd.read_csv('good_events_list_3_13_14_54.csv')
    e1 = da['Energy 1'].to_list()
    e2 = da['Energy 2'].to_list()
    d1 = da['Scatterer'].to_list()
    d2 = da['Absorber'].to_list()
    detecPair = list(zip(d1, d2))

    #Format for Legacy Expirmental
    # da = pd.read_csv('Time coincidence CC4 long run 30cm Cs137 (25.02).csv')
    # e1 = da['Energy_A'].to_list()
    # e2 = da['Energy_B'].to_list()
    # d1 = da['Scatterer'].to_list()
    # d2 = da['Absorber'].to_list()
    # detecPair = list(zip(d1, d2))

    #Format for current Expirmental
    # da = pd.read_csv('Energy Coincidence 30cm 137Cs (25.02) 2.csv')
    # e1 = da['Energy_A'].to_list()
    # e2 = da['Energy_B'].to_list()
    # d1 = da['Detector_A'].to_list()
    # d2 = da['Detector_B'].to_list()
    # detecPair = list(zip(d1, d2))

    print("Angleing", end='\r')
    angles = []
    for e1, e2 in zip(e1, e2):
        #e1, e2 = e1*1000, e2*1000 #Are you in MeV?
        angle = findAngle(min(e1,e2), max(e1,e2))
        angles.append(angle)
    
    print("Vectoring", end='\r')
    contrucVecs = [findContructVec(pair) for pair in detecPair]
    data = []
    print("Coneing", end='\r')
    for i, detectors in enumerate(detecPair):
        data.append(Cone(dectorPos(detectors[0]), contrucVecs[i], angles[i]))
    
    return data

def evenSpace(orgiCoords, numOfPoints, axes):
    '''
    Takes an ellipse of points after the cone has been intersected with a plane. Then evenly spaces out the points so there
    is no bias of points at any part of the ellipse
    '''

    if axes == Axes.x:
        x, y = orgiCoords[:,1], orgiCoords[:,2]
    elif axes == Axes.y:
        x, y = orgiCoords[:,1], orgiCoords[:,2]
    elif axes == Axes.z:    
        x, y = orgiCoords[:,0], orgiCoords[:,1]

    distances = np.sqrt(np.diff(x)**2+np.diff(y)**2)
    arclen = np.concatenate(([0], np.cumsum(distances)))

    arclen /= arclen[-1]

    interpx = interp1d(arclen, x, kind='cubic')
    interpy = interp1d(arclen, y, kind='cubic')

    points = np.linspace(0,1,numOfPoints)

    newX = interpx(points)
    newY = interpy(points)

    return newX, newY

def makeHeatMap(cones, axes):
    '''
    Creates heat maps for the specified planes to show where the interection point(s) of the cones are. Each cone can only add to any one
    histogram once, this removes bias towards the vertex of the cones and the most populated bin cannot have a value greater than the number
    of cones used - again used to remove bias and increase image resolution and accuracy. 
    '''

    bins = 200
    rnge = ([-100,100],[-100,100])  #How big are the heatmaps
    histos = []
    startPlane = -9
    endPlane = -11
    step = -1

    s = time()
    for plane in np.arange(startPlane, endPlane, step):
        planeHist = np.zeros((bins,bins)) #Start with empty heatmap
        for cone in cones:
            planeCoords = cone.projectToPlane(axes, plane)
            if len(planeCoords) == 0:
                continue
            x, y = evenSpace(planeCoords, 1000, axes)
            his, x, y = np.histogram2d(x, y, bins=bins, range=rnge)
            his = np.where(his > 0, 1, 0) #Truncate any bin that has more than one cone in it to 1
            planeHist += his

        histos.append(planeHist)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.pcolor(x, y, planeHist.T, cmap='plasma') #pcolor renders the heatmap in a way that is transpose to how the data is formatted

        maxx = np.argmax(planeHist)
        print(f"Plane: {plane} Num in MaxBin {np.max(planeHist)}, At y {x[maxx%bins]}, z {y[maxx//bins]}")

        ax.set_aspect('equal')
        ax.set_title(f"{axes} plane {plane}")
        ax.set_xlabel("Z")
        ax.set_ylabel("Y")
        plt.savefig(f'Plane{plane}.png', dpi=600)
        #plt.show()
    print(time()-s)
    return np.array(histos)

def main():

    data = loadData()
    #data = np.random.choice(data, 100)
    #data = loadExact()
    #data = loadSourceCentre()
    #data = loadBGO()
    print(len(data))

    histos = makeHeatMap(data, Axes.x)
    plt.show()

    #Create 3D render of the reconstructed image - this was not shown in the final report due to page limitations and
    #its lack of use to draw conclutions from
    grid = pv.ImageData()
    voxelsArr = np.transpose(histos, (0,1,2))
    np.save(f'SafteySave.npy', voxelsArr)

    try:
        root = tk.Tk()
        root.withdraw()
        root.filename = filedialog.asksaveasfile(mode = "w")
        np.save(root.filename.name+'.npy', voxelsArr)
        messagebox.showinfo('Save', 'Saved Sucessfully')
        root.destroy()
    except:
        pass

    grid.dimensions = np.array(voxelsArr.shape) + 1

    grid.origin = (-50,-50,26)
    grid.spacing = (1,1,1)

    grid.cell_data["values"] = voxelsArr.flatten(order="F") 

    thresh = grid.threshold(0.5)

    plotter = pv.Plotter()

    plotter.add_mesh_threshold(grid)
    #plotter.add_mesh(grid, show_edges=True, opacity=1)
    plotter.add_axes(interactive=True)
    plotter.show_axes()
    plotter.show_bounds()
    plotter.show()

if __name__ == '__main__':
    main()
