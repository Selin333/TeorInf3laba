from PIL import Image
import numpy as np

image = Image.open('orig.jpg')
im2arr = np.array(image)
print(im2arr)
# arr2im = Image.fromarray(im2arr)
# image.show()
