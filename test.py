from PIL import Image
import numpy as np

image = Image.open('orig.jpg')
im2arr = np.array(image)
# print(im2arr)
# arr2im = Image.fromarray(im2arr)
# image.show()
for i in im2arr:
    for j in i:
        for k in j:
            print(type(k))
            im2arr[i][j][k] = int(bin(k)[2:])
            print(k)
print(im2arr)