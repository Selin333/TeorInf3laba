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
def tri():
    # G sys
    matrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
     [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1],
     [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0]])


    # УДАЛЕНИЕ ИЗ МАТРИЦЫ СТОЛБЦОВ ДЛЯ ЕДИНИЧНОЙ это из старой лабы(из gsys опять gsys как в прошол)
    tutu = []
    column = 0
    column1 = 0
    ed = np.eye(len(matrix))
    # замена в колонне столбцов
    for i in range(len(matrix)):
        column = ed[:,i]
        for j in range(len(matrix[0])):
            column1 = matrix[:,j]
            if (column == column1).all():
                tutu.append(j)
                break
    matrix = np.delete(matrix,np.s_[tutu],axis=1)
# та же
    matrix2 =np.array ([
     [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
     [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1],
     [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0]])
    # print("Gsys =\n", matrix2)
    matrix1 = np.transpose(matrix)
    matrix3 = np.c_[matrix1, np.eye(len(matrix1))]
    print("Hsys=\n", matrix3)
    k = len(matrix2)
    n = len(matrix2[0])
    print("k = ", k)
    print("n = ", n)
    hihi = []
    I = []
    for i in range(0, 2 ** k):
        hihi.append([int(i)])
    m = []
    for i in range(0, len(hihi)):
        haha = np.binary_repr(hihi[i][0], width=k)
        for j in range(0, len(haha)):
            m.append(int(haha[j]))
        I.append(m)
        m = []

    I = np.array(I)
    # print("")
    # print("i=\n", I)
    # print(len(I))
    matrix_c = np.dot(I, matrix2)
# матрица кодовыйх слов
    for i in range(len(matrix_c)):
        for j in range(len(matrix_c[i])):
            if matrix_c[i][j] % 2 == 0:
                matrix_c[i][j] = 0
            else:
                matrix_c[i][j] = 1
    print("")
    print("c=\n", matrix_c)
    print("")
    d = np.sum(matrix_c, axis=1)
    print("wth =\n", d)
    print("")
    d = int(min(d[1:]))
    print("d =", d)
# это мб можно убрать
    x = Symbol('x')
    t = solve(2 * x + 1 - d, x)
    t[0] = math.floor(t[0])
    # print("t =", t[0])
    # print("Ro = ", Ro[0])

    HsysT = np.transpose(matrix3)


    pum = []
    for i in range(2**(19)):
        s = bin(i)[2:]
        pum.append(s.zfill(19))
    pum2 = []

    for i in range(len(pum)):
        if pum[i].count("1") == 1:
            pum2.append(pum[i])
        if pum[i].count("1") == 2:
            pum2.append(pum[i])