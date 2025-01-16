import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Cone:

    def __init__(self, height = 2, offsets = (0,0), rot = 0):
        self.theta = np.linspace(0, 2 * np.pi, 1000)
        self.z = np.linspace(0, height, 50)
        self.offsets = offsets

        self.Z, self.T = np.meshgrid(self.z, self.theta)
        self.R = self.Z
        self.X = self.R * np.cos(self.T) + self.offsets[0]
        self.Y = self.R * np.sin(self.T) + self.offsets[1]

    @property
    def xyz(self):
        return self.X, self.Y, self.Z
    
    def getCircle(self, index):
        return np.column_stack((self.X[:,index], self.Y[:,index]))

def findClose(circ1, circ2, threshold = 0.01):

    distances = np.linalg.norm(circ1[:, None, :] - circ2[None, :, :], axis=2)

    threshold = 0.01
    # Check if any distance is below the threshold
    close_pairs = np.argwhere(distances < threshold)
    closeCoords = []
    for idx1, idx2 in close_pairs:
        closeCoords.append((circ1[idx1], circ2[idx2]))
    
    return closeCoords

def findAverage(closeCoords):

    if not closeCoords:
        raise Exception("No coordinates close", "Hello")
    avX, avY = 0, 0
    for coords in closeCoords:
        avX += coords[0][0] + coords[1][0]
        avY += coords[0][1] + coords[1][1]

    avX /= len(closeCoords)*2
    avY /= len(closeCoords)*2

    return avX, avY

def main():
    
    height = 2

    cone1 = Cone(height)
    cone2 = Cone(height, (2,0))

    circ1 = cone1.getCircle(-1)
    circ2 = cone2.getCircle(-1)

    closeCoords = findClose(circ1, circ2)
    avX, avY = findAverage(closeCoords)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x, y = [], []
    for coords in closeCoords:
        i, j = coords
        x.append(i[0])
        x.append(j[0])
        y.append(i[1])
        y.append(j[1])
    
    ax.scatter(x, y, height, label = 'Close Points')
    
    ax.scatter(circ1[:,0], circ1[:,1], height, color = 'blue')
    ax.scatter(circ2[:,0], circ2[:,1], height, color = 'orange')


    ax.plot_surface(*cone1.xyz, color = 'blue', alpha=0.5)
    ax.plot_surface(*cone2.xyz, color = 'orange', alpha=0.5)

    print(f"Average of close points: ({avX}, {avY})")
    ax.scatter(avX, avY, color = 'grey', label='Close Point Average')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plt.show()

if __name__ == '__main__':
    main()
