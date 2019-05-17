import math

import numpy as np
import itertools
from PIL import Image, ImageDraw
from timeit import default_timer as timer
from numba import jit
from numba import vectorize, cuda
import matplotlib.pyplot as plt
from multiprocessing import Pool
import requests

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
        self.f = vectorize("float64(float64,float64)", target="parallel")(func)
        self.argsNums = args

def f1(x, y):
    return math.sin(x) + math.cos(y)

def f2(x, y):
    return (x - 15) ** 2 + (y - 15) ** 2 - 100

@jit(parallel=True,nopython=True)
def parallelPart(z):
    return np.min(z), np.max(z)

def minMaxGrid(func, box, n=128):
    l = []
    for x1, x2 in box.dim:
        l.append(np.arange(x1, x2, np.abs(x1 - x2) / n))
    l = np.meshgrid(*l,sparse=True)
    z = func.f(*l)
    return parallelPart(z)


def minMaxFunc(func, p):
    mins = []
    maxs = []
    for f in func:
        dims = []
        for n in f.argsNums:
            dims.append(tuple(p.dim[n]))
        b = Box(dims)
        min, max = minMaxGrid(f, b)
        mins.append(min)
        maxs.append(max)
    return np.max(np.array(mins)), np.max(np.array(maxs))


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
functions = [Function(f1, 0, 1), Function(f2, 0, 1)]
start = timer()
I, E, B = algo(P,0.01,functions)
dt = timer() - start
print(str(dt))
start = timer()
I, E, B = algo(P,00.1,functions)
dt = timer() - start
print(str(dt))
im = Image.new('RGB', (3000, 3000), color='white')
draw = ImageDraw.Draw(im)
for p in E:
    x1, x2 = p.dim[0]
    y1, y2 = p.dim[1]
    draw.rectangle([x1 * 100, y1 * 100, x2 * 100, y2 * 100], outline='white', fill='white')
for p in I:
    x1, x2 = p.dim[0]
    y1, y2 = p.dim[1]
    draw.rectangle([x1 * 100, y1 * 100, x2 * 100, y2 * 100], outline='red', fill='red')
for p in B:
    x1, x2 = p.dim[0]
    y1, y2 = p.dim[1]
    draw.rectangle([x1 * 100, y1 * 100, x2 * 100, y2 * 100], outline='green', fill='green')

del draw
im.save('pil.png')