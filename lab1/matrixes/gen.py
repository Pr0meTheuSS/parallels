import numpy as np

width=128
height=128

matrix = np.random.rand(height, width)
f = open("matrix_" + str(height)  + "_" + str(width) + ".dat", "w")
f.write(str(height)  + " " + str(width) + "\n")
for i in range(height):
    for j in range(width):
        f.write(str(matrix[i, j] * 100) + "\n")
f.close()

