import math

import numpy as np
import itertools
from PIL import Image, ImageDraw
from timeit import default_timer as timer
from numba import jit
import numba as nb
from numba import vectorize, cuda
import matplotlib.pyplot as plt
from multiprocessing import Pool


class Box:
    def __init__(self, dims=None):
        if dims == None:
            self.dim = []
        else:
            self.dim = dims[:]

    def addDim(self, x1, x2):
        self.dim.append((x1, x2))

    def diam(self):
        d = 0
        for x1, x2 in self.dim:
            d2 = math.fabs(x1 - x2)
            if d < d2:
                d = d2
        return d

    def split(self):
        d = 0
        n = 0
        i = 0
        for x1, x2 in self.dim:
            d2 = math.fabs(x1 - x2)
            if d < d2:
                n = i
                d = d2
            i += 1
        diml = []
        dimr = []
        x1, x2 = self.dim[n]
        mid = (x1 + x2) / 2
        i = 0
        for a in self.dim:
            if i != n:
                diml.append(tuple(a))
                dimr.append(tuple(a))
            else:
                diml.append((x1, mid))
                dimr.append((mid, x2))
            i += 1
        l = Box(diml)
        r = Box(dimr)
        return l, r

    def __str__(self):
        s = ""
        i = 0
        for x1, x2 in self.dim:
            s = s + "Axis " + str(i) + ": (" + str(x1) + "," + str(x2) + ")\n"
            i += 1
        return s


class Function:
    def __init__(self, func, *args):
        self.f = func
        self.argsNums = args


@cuda.jit
def cudaFunc(x1, x2, y1, y2, max1, max2, min1, min2, r, n=128):
    a = cuda.shared.array(128, nb.float32)
    b = cuda.shared.array(128, nb.float32)
    c = cuda.shared.array(128, nb.float32)
    d = cuda.shared.array(128, nb.float32)
    ret = cuda.shared.array(4, nb.float32)
    i = cuda.grid(1) % n
    j = cuda.grid(1) // n
    x = abs(x1 - x2) / n * i + x1
    y = abs(y1 - y2) / n * j + y1
    ind = cuda.threadIdx.x
    a[ind] = math.sin(x) + math.cos(y)
    b[ind] = math.sin(x) + math.cos(y)
    c[ind] = (x - 15) ** 2 + (y - 15) ** 2 - 100
    d[ind] = (x - 15) ** 2 + (y - 15) ** 2 - 100
    cuda.syncthreads()
    ind2 = cuda.blockIdx.x
    s = cuda.blockDim.x >> 1
    while s != 0:
        if ind < s:
            su = ind + s
            a[ind] = max(a[ind], a[su])
            b[ind] = min(b[ind], b[su])
            c[ind] = max(c[ind], c[su])
            d[ind] = min(d[ind], d[su])
        cuda.syncthreads()
        s >>= 1
    cuda.syncthreads()
    max1[ind2] = a[0]
    min1[ind2] = b[0]
    max2[ind2] = c[0]
    min2[ind2] = d[0]
    cuda.syncthreads()
    a[ind] = max1[ind]
    b[ind] = min1[ind]
    c[ind] = max2[ind]
    d[ind] = min2[ind]
    cuda.syncthreads()
    s = cuda.blockDim.x >> 1
    while s != 0:
        if ind < s:
            su = ind + s
            a[ind] = max(a[ind], a[su])
            b[ind] = min(b[ind], b[su])
            c[ind] = max(c[ind], c[su])
            d[ind] = min(d[ind], d[su])
        cuda.syncthreads()
        s >>= 1
    cuda.syncthreads()
    r[0] = a[0]
    r[1] = b[0]
    r[2] = c[0]
    r[3] = d[0]


@cuda.jit
def cudaRed(m, r, mode, n=100):
    b = cuda.shared.array(100, nb.float32)
    i = cuda.grid(1) % n
    j = cuda.grid(1) // n
    ind = i + j * cuda.blockDim.x
    b[ind] = m[ind]
    cuda.syncthreads()
    s = 100 >> 1
    while s != 0:
        if ind < s:
            su = ind + s
            if mode == 1:
                b[ind] = max(b[ind], b[su])
            else:
                b[ind] = min(b[ind], b[su])
        cuda.syncthreads()
        s >>= 1
    cuda.syncthreads()
    if ind == 0:
        r[0] = b[0]


@cuda.jit
def cudaFunc2(x1, x2, y1, y2, maxes, mode, n=100):
    a = cuda.shared.array(100, nb.float32)
    i = cuda.grid(1) % n
    j = cuda.grid(1) // n
    x = (abs(x1 - x2) / n) * i + x1
    y = (abs(y1 - y2) / n) * j + y1
    i, j = cuda.threadIdx.x, cuda.threadIdx.y
    ind = cuda.threadIdx.x
    a[cuda.threadIdx.x] = (x - 15) ** 2 + (y - 15) ** 2 - 100
    cuda.syncthreads()
    s = cuda.blockDim.x * cuda.blockDim.y >> 1
    while s != 0:
        if cuda.threadIdx.x < s:
            su = ind + s
            if mode == 1:
                a[ind] = max(a[ind], a[su])
            else:
                a[ind] = min(a[ind], a[su])
        cuda.syncthreads()
        s >>= 1
    cuda.syncthreads()
    ind2 = cuda.blockIdx.x
    if ind == 0:
        maxes[ind2] = a[0]
    cuda.syncthreads()


def f1(x, y):
    return math.sin(x) + math.cos(y)


def f2(x, y):
    return (x - 15) ** 2 + (y - 15) ** 2 - 100


def minMaxGrid(box, n=128):
    threads_per_block = 128
    blocks = 128
    x1, x2 = box.dim[0]
    y1, y2 = box.dim[1]
    maxes = cuda.device_array(128, dtype=float)
    mins = cuda.device_array(128, dtype=float)
    maxes2 = cuda.device_array(128, dtype=float)
    mins2 = cuda.device_array(128, dtype=float)
    """maxes = np.zeros(100, dtype=float)
    mins = np.zeros(100, dtype=float)
    maxes2 = np.zeros(100, dtype=float)
    mins2 = np.zeros(100, dtype=float)"""
    r = np.zeros(4, dtype=float)
    cudaFunc[blocks, threads_per_block](x1, x2, y1, y2, maxes, maxes2, mins, mins2, r, n)
    return r[0], r[1], r[2], r[3]


def minMaxFunc(func, p):
    max1, min1, max2, min2 = minMaxGrid(p)
    return max(min1, min2), max(max1, max2)


def algo(P, d, func):
    temp = []
    main = []
    I = []
    E = []
    main.append(P)
    print("Main size is " + str(len(main)) + " id:" + str(id))
    curD = P.diam()
    while curD > d and len(main) > 0:
        for p in main:
            min, max = minMaxFunc(func, p=p)
            # print(min,max)
            if min > 0:
                E.append(p)
                continue
            if max < 0:
                I.append(p)
                continue
            l, r = p.split()
            temp.append(l)
            temp.append(r)
            curD = l.diam()
        main = temp[:]
        temp = []
        print("Main size is " + str(len(main)) + "\nCurrent radius is " + str(curD) + " id:" + str(id))
    return I, E, main


P = Box()
P.addDim(0, 30)
P.addDim(0, 30)
functions = [Function(cudaFunc, 0, 1), Function(cudaFunc2, 0, 1)]
start = timer()
I, E, B = algo(P, 0.01, functions)
dt = timer() - start
print(str(dt))
im = Image.new('RGB', (3000, 3000), color='white')
draw = ImageDraw.Draw(im)
for p in E:
    x1, x2 = p.dim[0]
    y1, y2 = p.dim[1]
    draw.rectangle([x1 * 100, y1 * 100, x2 * 100, y2 * 100], outline='black', fill='white')
for p in I:
    x1, x2 = p.dim[0]
    y1, y2 = p.dim[1]
    draw.rectangle([x1 * 100, y1 * 100, x2 * 100, y2 * 100], outline='black', fill='red')
for p in B:
    x1, x2 = p.dim[0]
    y1, y2 = p.dim[1]
    draw.rectangle([x1 * 100, y1 * 100, x2 * 100, y2 * 100], outline='black', fill='green')

del draw
im.save('pilcuda.png')
