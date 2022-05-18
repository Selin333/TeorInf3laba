import numpy as np
from PIL import Image # pip install Pillow

# img = np.asarray(Image.open('orig.jpg').convert('RGB'), dtype=uint16)

image = Image.open('orig.jpg')
# img = np.array(image, dtype=uint8)
img = np.asarray(image, dtype='uint64')


for i in range(len(img)):
    for j in range(len(img[i])):
        for k in range(len(img[i][j])):
            img[i][j][k] = bin(img[i][j][k])[2:]
            print(img[i][j][k])

