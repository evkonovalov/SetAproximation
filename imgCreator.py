from PIL import Image, ImageDraw
import math


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


I, E, B = [], [], []
f = open("out2d.txt", "r")
vec_size = int(f.readline())
mins = [0] * vec_size
maxs = [0] * vec_size
pictureSize = 0
for i in range(vec_size):
    mins[i], maxs[i] = map(float, f.readline().split())
    pictureSize = max(pictureSize,abs(mins[i]-maxs[i]))
main_size = int(f.readline())
for i in range(main_size):
    x1, x2, y1, y2 = map(float, f.readline().split())
    B.append(Box([[x1, x2], [y1, y2]]))
i_size = int(f.readline())
for i in range(i_size):
    x1, x2, y1, y2 = map(float, f.readline().split())
    I.append(Box([[x1, x2], [y1, y2]]))
e_size = int(f.readline())
for i in range(e_size):
    x1, x2, y1, y2 = map(float, f.readline().split())
    E.append(Box([[x1, x2], [y1, y2]]))
im = Image.new('RGB', (int(pictureSize*100), int(pictureSize*100)), color='white')
draw = ImageDraw.Draw(im)
for p in E:
    x1, x2 = p.dim[0]
    y1, y2 = p.dim[1]
    draw.rectangle([(x1 - mins[0]) * 100, (y1 - mins[1]) * 100, (x2 - mins[0]) * 100, (y2 - mins[1]) * 100],
                   outline='black', fill='white')
for p in B:
    x1, x2 = p.dim[0]
    y1, y2 = p.dim[1]
    draw.rectangle([(x1 - mins[0]) * 100, (y1 - mins[1]) * 100, (x2 - mins[0]) * 100, (y2 - mins[1]) * 100],
                   outline='black', fill='green')
for p in I:
    x1, x2 = p.dim[0]
    y1, y2 = p.dim[1]
    draw.rectangle([(x1 - mins[0]) * 100, (y1 - mins[1]) * 100, (x2 - mins[0]) * 100, (y2 - mins[1]) * 100],
                   outline='black', fill='red')

del draw
im.save('pilcudac++round2.png')
