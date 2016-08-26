size = [2, 2, 2]
[a, b, c] = [5.038, 5.038, 13.772]
for xSize in range(-size[0]+1, size[0]):
    for ySize in range(-size[1]+1, size[1]):
        for zSize in range(-size[2]+1, size[2]):
            print [xSize, ySize, zSize]
