import numpy as np


width=int(input())
height=width

matrix = np.random.rand(height, width)
f = open("matrix_" + str(height)  + "_" + str(width) + ".dat", "w")
f.write(str(height)  + " " + str(width) + "\n")
for i in range(height):
    for j in range(width):
        f.write(str(matrix[i, j] * 100) + "\n")
f.close()

