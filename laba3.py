from PIL import Image, ImageDraw
import numpy as np

image = Image.open('orig.jpg')
im2arr = np.array(image)
perem = image.load()
draw =ImageDraw.Draw(image)
draw.point((1,1),(255,255,255))
# image.show()
rgb = list(perem[1,1])
print(type(rgb[0]))
#print(im2arr)
# arr2im = Image.fromarray(im2arr)
# image.show()
# for i in im2arr:
#     for j in i:
#         for k in j:
#
#             arrara[i][j][k].append(bin(k))
#             print(k)
