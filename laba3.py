from tkinter import filedialog as fd
from tkinter import *
import os
import matplotlib.image as img
import matplotlib.pyplot as plt
import random
import numpy as np
from sympy import *
import math

name = ""
def callback():
    ftypes = [("Картинки",".jpg")]
    name = fd.askopenfilename(filetypes = ftypes)
    return name

def call():
    global name
    name = callback()


def oshibaisya2(opa1):

    rng=list(range(0,19))
    rng1=random.sample(rng,2)
    for i in range(len(rng1)):
        if (opa1[rng1[i]])=="1":
            opa1 = opa1[:rng1[i]] + "0" + opa1[rng1[i] + 1:]
        else:
            opa1 = opa1[:rng1[i]] + "1" + opa1[rng1[i] + 1:]

    return (opa1)

def oshibaisya(opa):
    opa = "{0:08b}".format(opa)
    print("Без ошибки\n",opa)
    # ошибка рандомом
    rng = list(range(0, 8))
    rng1 = random.sample(rng, 2)
    for i in range(len(rng1)):
        if (opa[rng1[i]]) == "1":
            opa = opa[:rng1[i]] + "0" + opa[rng1[i] + 1:]
        else:
            opa = opa[:rng1[i]] + "1" + opa[rng1[i] + 1:]
        print("С ошибкой\n",opa)

    return ( int(opa[:],2))



# Первая часть
def pervoe():
    Way=name
    os.system(Way)

# #Вторая часть
def vtoroe():
    Way=name
    photo=img.imread(Way)

    razmer=photo.shape
#размерность
    y=razmer[0]
    x=razmer[1]
#
    for i in range(y):
        for j in range(x):
            photo[i][j][0] = int(oshibaisya(photo[i][j][0]))
            photo[i][j][1] = int(oshibaisya(photo[i][j][1]))
            photo[i][j][2] = int(oshibaisya(photo[i][j][2]))
    print("Фото в матрице\n",photo)
    plt.imshow(photo)
    plt.axis("off")
    plt.show()

# # Третья часть
