import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from enum import Enum

class Axes(Enum):
    x = 0
    y = 1
    z = 2

class Cone:

    NUM_OF_ANGLES = 1000
    NUM_OF_Z = 50

    def __init__(self, angle = np.pi/4, **kwargs):
        '''
        Creates a cone with a given angle. Key words: height - the height of the cone
        offsets - where the vertex of the cone will be
        rot - rotatation of the cone, given as a tuple (axis, angle (in RADIANS))
        '''

        height = kwargs.get('height', 2)
        offsets = kwargs.get('offsets', (0,0,0))
        rot = kwargs.get('rot', None)

        self.theta = np.linspace(0, 2 * np.pi, self.NUM_OF_ANGLES)
        self.z = np.linspace(0, height, self.NUM_OF_Z)
        self.offsets = offsets
        self.height = height

        #Creates a matrix of the different z values and the angles at each z
        self.Z, self.T = np.meshgrid(self.z, self.theta)

        self.R = self.Z*np.tan(angle)

        self.X = self.R * np.cos(self.T)
        self.Y = self.R * np.sin(self.T)

        if rot:
            self.rotate(*rot)
        
        self.offset((*self.offsets, 0))

    def makeRotMat(self, axis, angle):
        
        if axis == Axes.x:
            rotMat = [[1, 0, 0],
                      [0, np.cos(angle), -np.sin(angle)],
                      [0, np.sin(angle), np.cos(angle)]]
        elif axis == Axes.y:
            rotMat = [[np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)]]
        elif axis == Axes.z:
            rotMat = [[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]]
        return rotMat

    def rotate(self, axis, angle):
        '''
        Rotate the cone by angle in RADIANS
        '''
        rotMat = self.makeRotMat(axis, angle)

        #Turn the 3 2D arrays into 3XN matrix
        allPoints = np.vstack((self.X.ravel(), self.Y.ravel(), self.Z.ravel()))

        #Rotate the matrix
        rotated_points = rotMat @ allPoints

        # Reshape the rotated array back into the original shapes for the cone
        self.X = rotated_points[0, :].reshape(self.X.shape)
        self.Y = rotated_points[1, :].reshape(self.Y.shape)
        self.Z = rotated_points[2, :].reshape(self.Z.shape)

    def offset(self, offsets):
        '''
        Apply offsets to the X, Y and Z axis
        '''
        self.X += offsets[0]
        self.Y += offsets[1]
        self.Z += offsets[2]

    @property
    def xyz(self):
        return self.X, self.Y, self.Z
    
    def getCircle(self, index) -> np.ndarray:
        '''
        Returns the points on the circle of input index of the cone
        '''
        return np.column_stack((self.X[:,index], self.Y[:,index], self.Z[:, index]))
    
    def recentreVertex(self):
        '''
        Places the vertex of the cone back on the offset point. Useful if cone has been rotated and the vertex is no longer where it should be
        '''
        displacements = (self.offsets[0]-self.X[0,0], self.offsets[1]-self.Y[0,0], 0-self.Z[0,0])
        self.offset(displacements)

    def projectToZPlane(self, zPlane, maxSize = 100):
        pass

    def projectAsEllipse(self, plane, planeHeight, **kwargs) -> np.ndarray:
        '''
        Returns an ellipse where the cone intersects the given plane
        zErr is the z value allowed either side of the planeHeight
        flattenZ sets all returned Z values to the planeheight
        '''
        zErr = kwargs.get('zErr', (self.height/self.NUM_OF_Z)/2)
        flattenZ = kwargs.get('flattenZ', False)

        indices = np.argwhere(np.isclose(self.Z, planeHeight, atol=zErr))

        x = self.X[indices[:, 0], indices[:, 1]]
        y = self.Y[indices[:, 0], indices[:, 1]]
        if not flattenZ:
            z = self.Z[indices[:, 0], indices[:, 1]]
        else:
            z = np.zeros(x.shape) + planeHeight

        return np.column_stack((x, y, z))

    @staticmethod
    def findClose(circ1, circ2, threshold = 0.01) -> list:
        #Ngl this line is from chatGPT
        distances = np.linalg.norm(circ1[:, None, :] - circ2[None, :, :], axis=2)

        # Check if any distance is below the threshold
        threshold = 0.01
        close_pairs = np.argwhere(distances < threshold)
        closeCoords = []
        for idx1, idx2 in close_pairs:
            closeCoords.append((circ1[idx1], circ2[idx2]))
        
        return closeCoords

    @staticmethod
    def findAverage(closeCoords) -> float:
        '''
        Finds the average position of a set of input coordinates
        '''
        if not closeCoords:
            return 0, 0
        avX, avY = 0, 0
        for coords in closeCoords:
            avX += coords[0][0] + coords[1][0]
            avY += coords[0][1] + coords[1][1]

        avX /= len(closeCoords)*2
        avY /= len(closeCoords)*2

        return avX, avY
    
    @staticmethod
    def findZplane(cone1, cone2, height = 2, returnAll = False) -> tuple:

        zToSearch = np.linspace(0, height, Cone.NUM_OF_Z)
        zOfInterest = {}
        for z in zToSearch:
            ellip1 = cone1.projectAsEllipse(z, flattenZ = True)
            ellip2 = cone2.projectAsEllipse(z, flattenZ = True)
            close = Cone.findClose(ellip1, ellip2)
            if len(close) == 0:
                continue
            zOfInterest[z] = close

        if returnAll:
            return zOfInterest
        most = max(zOfInterest, key=lambda k: len(zOfInterest[k]))
        return (most, zOfInterest[most])

def main():
    
    height = 2

    cone1 = Cone(np.pi/4)
    cone1.projectAsEllipse(Axes.z, 2)

    cone2 = Cone(np.pi/8, height=3, offsets = (1,0,0), rot = (Axes.y, np.pi/8))    
    cone2.projectAsEllipse(Axes.z, 1)

    #Testing to see if rotating the cone one way then back returns it to orginal position
    cone3 = Cone(np.pi/8, height = height, offsets = (1,1,0), rot = (Axes.y, -np.pi/4))
    cone3.rotate(Axes.y, np.pi/4)
    cone3.recentreVertex()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ellip1 = cone1.projectAsEllipse(Axes.z, 2, flattenZ = True)
    ellip2 = cone2.projectAsEllipse(Axes.z, 2, flattenZ = True)

    closeCoords = Cone.findClose(ellip1, ellip2)
    avX, avY = Cone.findAverage(closeCoords)
    x, y = [], []
    for coords in closeCoords:
        i, j = coords
        x.append(i[0])
        x.append(j[0])
        y.append(i[1])
        y.append(j[1])
    
    ax.scatter(x, y, height, label = 'Close Points', color='green')
    print(f"Average of close points: ({avX}, {avY})")

    ax.scatter(avX, avY, color = 'grey', label='Close Point Average')
    ax.scatter(ellip2[:,0], ellip2[:,1], ellip2[:,2], color = 'red') 

    ax.plot_surface(*cone1.xyz, color = 'blue', alpha=0.5)
    ax.plot_surface(*cone2.xyz, color = 'orange', alpha=0.5)
    ax.plot_surface(*cone3.xyz, color = 'pink', alpha = 0.5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    ax.legend()

    plt.show()

if __name__ == '__main__':
    main()
