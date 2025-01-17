import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Cone:

    NUM_OF_ANGLES = 1000
    NUM_OF_Z = 50

    def __init__(self, height = 2, angle = np.pi/4, offsets = (0,0), rot = 0):
        self.theta = np.linspace(0, 2 * np.pi, self.NUM_OF_ANGLES)
        self.z = np.linspace(0, height, self.NUM_OF_Z)
        self.offsets = offsets
        self.height = height

        #Creates a matrix of the different z values and the angles at each z
        self.Z, self.T = np.meshgrid(self.z, self.theta)

        #TODO this line of code needs to be changed, atm this is assuming a 45 degree angle. I will make it so it is more generic
        #self.R = self.Z
        self.R = self.Z*np.tan(angle)

        #Ok so roating first then offseting is better imo as it means the cone behaves as expected
        #self.X = self.R * np.cos(self.T) + self.offsets[0]
        #self.Y = self.R * np.sin(self.T) + self.offsets[1]
        
        self.X = self.R * np.cos(self.T)
        self.Y = self.R * np.sin(self.T)

        if rot:
            self.roate(rot)
        
        self.offset((*self.offsets, 0))

    def makeRotMat(self, angle):
        yRotMat = [[np.cos(angle), 0, np.sin(angle)],
                   [0, 1, 0],
                   [-np.sin(angle), 0, np.cos(angle)]]
        return yRotMat

    def roate(self, angle):
        '''
        Rotate the cone by angle in RADIANS
        '''
        yRotMat = self.makeRotMat(angle)

        #Turn the 3 2D arrays into 3XN matrix
        allPoints = np.vstack((self.X.ravel(), self.Y.ravel(), self.Z.ravel()))

        #Rotate the matrix
        rotated_points = yRotMat @ allPoints

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
    
    def getCircle(self, index):
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

    def projectAsElipise(self, planeHeight, **kwargs):
        '''
        Returns an ellipse where the cone intersects the given plane
        zErr is the z value allowed either side of the planeHeight
        flattenZ sets all returned Z values to the planeheight
        '''
        zErr = kwargs.get('zErr')
        flattenZ = kwargs.get('flattenZ')
        if zErr is None:
            zErr = (self.height/self.NUM_OF_Z)/2

        indices = np.argwhere(np.isclose(self.Z, planeHeight, atol=zErr))

        x = self.X[indices[:, 0], indices[:, 1]]
        y = self.Y[indices[:, 0], indices[:, 1]]
        if not flattenZ:
            z = self.Z[indices[:, 0], indices[:, 1]]
        else:
            z = np.zeros(x.shape) + planeHeight

        return np.column_stack((x, y, z))

def findClose(circ1, circ2, threshold = 0.01):
    '''
    This code doesn't currently work, but was used to find were two circles meet
    '''

    #Ngl this line is from chatGPT
    distances = np.linalg.norm(circ1[:, None, :] - circ2[None, :, :], axis=2)

    # Check if any distance is below the threshold
    threshold = 0.01
    close_pairs = np.argwhere(distances < threshold)
    closeCoords = []
    for idx1, idx2 in close_pairs:
        closeCoords.append((circ1[idx1], circ2[idx2]))
    
    return closeCoords

def findAverage(closeCoords):
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

def main():
    
    height = 2

    cone1 = Cone(height, np.pi/4)

    cone1.projectAsElipise(2)

    cone2 = Cone(3, np.pi/8, (1,0), np.pi/8)
    cone2.projectAsElipise(1)

    #Testing to see if rotating the cone one way then back returns it to orginal position
    cone3 = Cone(height, np.pi/8, (1,1), -np.pi/4)
    cone3.roate(np.pi/4)
    cone3.recentreVertex()

    circ1 = cone1.getCircle(-1)
    circ2 = cone2.getCircle(-1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #Since I broke my intersection code, this will be commented out..... for now
    #closeCoords = findClose(circ1, circ2)
    #avX, avY = findAverage(closeCoords)

    ellip1 = cone1.getCircle(-1)
    ellip2 = cone2.projectAsElipise(2, flattenZ = True)

    closeCoords = findClose(ellip1, ellip2)
    avX, avY = findAverage(closeCoords)
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
   
    #ax.scatter(circ1[:,0], circ1[:,1], circ1[:,2], color = 'blue')
    ax.scatter(circ2[:,0], circ2[:,1], circ2[:,2], color = 'orange')


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
