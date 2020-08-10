#Cristopher jose Rodolfo Barrios Solis
#SR4

import struct
from obj import Obj
import random
from collections import namedtuple

def char(c):
		return struct.pack('=c', c.encode('ascii'))

def word(c):
	return struct.pack('=h', c)

def dword(c):
	return struct.pack('=l', c)

def normalizeColorArray(colors_array):
    return [round(i*255) for i in colors_array]

def color(r,g,b):
	return bytes([int(b * 255), int(g * 255), int(r*255)])

##Codigo de dado##
def sum(v0, v1):
    """
        Input: 2 size 3 vectors
        Output: Size 3 vector with the per element sum
    """
    return V3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z)

def sub(v0, v1):
    """
        Input: 2 size 3 vectors
        Output: Size 3 vector with the per element substraction
    """
    return V3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z)

def mul(v0, k):
    """
        Input: 2 size 3 vectors
        Output: Size 3 vector with the per element multiplication
    """
    return V3(v0.x * k, v0.y * k, v0.z *k)

def dot(v0, v1):
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z
	
def length(v0):
    """
        Input: 1 size 3 vector
        Output: Scalar with the length of the vector
    """  
    return (v0.x**2 + v0.y**2 + v0.z**2)**0.5


def norm(v0):
    """
        Input: 1 size 3 vector
        Output: Size 3 vector with the normal of the vector
    """  
    v0length = length(v0)

    if not v0length:
        return V3(0, 0, 0)

    return V3(v0.x/v0length, v0.y/v0length, v0.z/v0length)

def bbox(*vertices):
    xs = [ vertex.x for vertex in vertices ]
    ys = [ vertex.y for vertex in vertices ]

    xs.sort()
    ys.sort()

    xmin = xs[0]
    xmax = xs[-1]
    ymin = ys[0]
    ymax = ys[-1]

    return xmin, xmax, ymin, ymax

def cross(v1, v2):
    return V3(
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x,
    )

def baryCoords(A, B, C, P):
    try:
        u = ( ((B.y - C.y)*(P.x - C.x) + (C.x - B.x)*(P.y - C.y) ) /
              ((B.y - C.y)*(A.x - C.x) + (C.x - B.x)*(A.y - C.y)) )

        v = ( ((C.y - A.y)*(P.x - C.x) + (A.x - C.x)*(P.y - C.y) ) /
              ((B.y - C.y)*(A.x - C.x) + (C.x - B.x)*(A.y - C.y)) )

        w = 1 - u - v
    except:
        return -1, -1, -1

    return u, v, w
####

V2 = namedtuple('Point2', ['x', 'y'])
V3 = namedtuple('Point3', ['x', 'y', 'z'])
V4 = namedtuple('Point4', ['x', 'y', 'z','w'])

class Render(object):

	def __init__(self):
		self.framebuffer = []
		self.curr_color = color(1, 1, 1)

	def glInit(self):
		self.curr_color = color(0, 0, 0)

            
	def glClear(self):
		self.framebuffer = [[color(0, 0, 0) for x in range(
		    self.width)] for y in range(self.height)]
		self.zbuffer = [ [ -float('inf') for x in range(self.width)] for y in range(self.height) ]

	def glCreateWindow(self, width, height):
		self.width = width
		self.height = height

	def glClearColor(self, r, g, b):
		clearColor = color(
				round(r * 255),
				round(g * 255),
				round(b * 255)
			)

		self.framebuffer = [[clearColor for x in range(
		    self.width)] for y in range(self.height)]

	def glColor(self, r, g, b):
		self.curr_color = color(round(r * 255), round(g * 255), round(b * 255))

	def glViewport(self, x, y, width, height):
		self.viewportX = x
		self.viewportY = y
		self.viewportWidth = width
		self.viewportHeight = height

	def point(self, x, y):
		self.framebuffer[x][y] = self.curr_color

	def glVertex(self, x, y):
		X = round((x+1) * (self.viewportWidth/2) + self.viewportX)
		Y = round((y+1) * (self.viewportHeight/2) + self.viewportY)
		self.point(X, Y)

	def transform(self, vertex, translate=V3(0,0,0), scale=V3(1,1,1)):
		return V3(round(vertex[0] * scale.x + translate.x),round(vertex[1] * scale.y + translate.y),round(vertex[2] * scale.z + translate.z))
		

	def glVertCord(self, x, y, color= None):
		if x >= self.width or x < 0 or y >= self.height or y < 0:
			return
		try:
			self.framebuffer[y][x] = color or self.curr_color
		except:
			pass

	def glLine(self, x0, y0, x1, y1):
		x0 = round(( x0 + 1) * (self.viewportWidth  / 2 ) + self.viewportX)
		x1 = round(( x1 + 1) * (self.viewportWidth  / 2 ) + self.viewportX)
		y0 = round(( y0 + 1) * (self.viewportHeight / 2 ) + self.viewportY)
		y1 = round(( y1 + 1) * (self.viewportHeight / 2 ) + self.viewportY)

		dx = abs(x1 - x0)
		dy = abs(y1 - y0)

		steep = dy > dx

		if steep:
			x0, y0 = y0, x0
			x1, y1 = y1, x1

		if x0 > x1:
			x0, x1 = x1, x0
			y0, y1 = y1, y0

		dx = abs(x1 - x0)
		dy = abs(y1 - y0)

		offset = 0
		limit = 0.5

		m = dy/dx
		y = y0

		for x in range(x0, x1 + 1):
			if steep:
				self.glVertCord(y, x)
			else:
				self.glVertCord(x, y)

			offset += m
			if offset >= limit:
				y += 1 if y0 < y1 else -1
				limit += 1

	def glCord(self, x0, y0, x1, y1):

		dx = abs(x1 - x0)
		dy = abs(y1 - y0)

		steep = dy > dx

		if steep:
			x0, y0 = y0, x0
			x1, y1 = y1, x1

		if x0 > x1:
			x0, x1 = x1, x0
			y0, y1 = y1, y0

		dx = abs(x1 - x0)
		dy = abs(y1 - y0)

		offset = 0
		limit = 0.5
		
		try:
			m = dy/dx
		except ZeroDivisionError:
			pass
		else:
			y = y0

			for x in range(x0, x1 + 1):
				if steep:
					self.glVertCord(y, x)
				else:
					self.glVertCord(x, y)

				offset += m
				if offset >= limit:
					y += 1 if y0 < y1 else -1
					limit += 1



	def triang(self, A, B, C, _color = color(1, 1, 1), texture = None, texcoords = (), intensity = 1):

		minX = min(A.x, B.x, C.x)
		minY = min(A.y, B.y, C.y)
		maxX = max(A.x, B.x, C.x)
		maxY = max(A.y, B.y, C.y)

		for x in range(minX, maxX + 1):
			for y in range(minY, maxY + 1):
				if x >= self.width or x < 0 or y >= self.height or y < 0:
					continue

				u, v, w = baryCoords(A, B, C, V2(x, y))

				if u >= 0 and v >= 0 and w >= 0:

					z = A.z * u + B.z * v + C.z * w
					if z > self.zbuffer[y][x]:
						
						b, g , r = _color
						b /= 255
						g /= 255
						r /= 255

						b *= intensity
						g *= intensity
						r *= intensity

						if texture:
							ta, tb, tc = texcoords
							tx = ta.x * u + tb.x * v + tc.x * w
							ty = ta.y * u + tb.y * v + tc.y * w

							texColor = texture.getColor(tx, ty)
							b *= texColor[0] / 255
							g *= texColor[1] / 255
							r *= texColor[2] / 255

						self.glVertCord(x, y, color(r,g,b))
						self.zbuffer[y][x] = z

	def glLoad(self, filename, translate=V3(0,0,0), scale=V3(1,1,1), texture = None, isWireframe = False):
		model = Obj(filename)

		light = V3(0,0,1)

		for face in model.faces:

			vertCount = len(face)

			if isWireframe:
				for vert in range(vertCount):
					
					v0 = model.vertices[ face[vert][0] - 1 ]
					v1 = model.vertices[ face[(vert + 1) % vertCount][0] - 1]

					x0 = round(v0[0] * scale[0]  + translate[0])
					y0 = round(v0[1] * scale[1]  + translate[1])
					x1 = round(v1[0] * scale[0]  + translate[0])
					y1 = round(v1[1] * scale[1]  + translate[1])

					self.glCord(x0, y0, x1, y1)
			else:
				v0 = model.vertices[ face[0][0] - 1 ]
				v1 = model.vertices[ face[1][0] - 1 ]
				v2 = model.vertices[ face[2][0] - 1 ]
				if vertCount > 3:
					v3 = model.vertices[ face[3][0] - 1 ]

				v0 = self.transform(v0,translate, scale)
				v1 = self.transform(v1,translate, scale)
				v2 = self.transform(v2,translate, scale)
				if vertCount > 3:
					v3 = self.transform(v3,translate, scale)

				if texture:
					vt0 = model.texcoords[face[0][1] - 1]
					vt1 = model.texcoords[face[1][1] - 1]
					vt2 = model.texcoords[face[2][1] - 1]
					vt0 = V2(vt0[0], vt0[1])
					vt1 = V2(vt1[0], vt1[1])
					vt2 = V2(vt2[0], vt2[1])
					if vertCount > 3:
						vt3 = model.texcoords[face[3][1] - 1]
						vt3 = V2(vt3[0], vt3[1])
				else:
					vt0 = V2(0,0) 
					vt1 = V2(0,0) 
					vt2 = V2(0,0) 
					vt3 = V2(0,0) 

				normal = cross(sub(V3(v1.x,v1.y,v1.z),V3(v0.x,v0.y,v0.z)), sub(V3(v2.x,v2.y,v2.z),V3(v0.x,v0.y,v0.z)))
				normal = norm(normal)
				intensity = dot(normal, light)

				if intensity >=0:
					self.triang(v0,v1,v2, texture = texture, texcoords = (vt0,vt1,vt2), intensity = intensity )
					if vertCount > 3: 
						self.triang(v0,v2,v3, texture = texture, texcoords = (vt0,vt2,vt3), intensity = intensity)

	def glFinishBuffer(self, filename):
		f = open(filename, 'wb')

		f.write(bytes('B'.encode('ascii')))
		f.write(bytes('M'.encode('ascii')))
		f.write(dword(14 + 40 + self.width * self.height * 3))
		f.write(dword(0))
		f.write(dword(14 + 40))

		f.write(dword(40))
		f.write(dword(self.width))
		f.write(dword(self.height))
		f.write(word(1))
		f.write(word(24))
		f.write(dword(0))
		f.write(dword(self.width * self.height * 3))
		f.write(dword(0))
		f.write(dword(0))
		f.write(dword(0))
		f.write(dword(0))

		minZ = float('inf')
		maxZ = -float('inf')
		for x in range(self.height):
			for y in range(self.width):
				if self.zbuffer[x][y] != -float('inf'):
					if self.zbuffer[x][y] < minZ:
						minZ = self.zbuffer[x][y]

					if self.zbuffer[x][y] > maxZ:
						maxZ = self.zbuffer[x][y]

		for x in range(self.height):
			for y in range(self.width):
				depth = self.zbuffer[x][y]
				if depth == -float('inf'):
					depth = minZ
				depth = (depth - minZ) / (maxZ - minZ)
				f.write(color(depth,depth,depth))

		f.close()
        
	def glFinish(self, filename):
		f = open(filename, 'wb')

		f.write(char("B"))
		f.write(char("M"))
		f.write(dword(14+40+self.width*self.height))
		f.write(dword(0))
		f.write(dword(14+40))

		f.write(dword(40))
		f.write(dword(self.width))
		f.write(dword(self.height))
		f.write(word(1))
		f.write(word(24))
		f.write(dword(0))
		f.write(dword(self.width * self.height * 3))
		f.write(dword(0))
		f.write(dword(0))
		f.write(dword(0))
		f.write(dword(0))

		for x in range(self.height):
			for y in range(self.width):
				f.write(self.framebuffer[x][y])

		f.close()

bitmap = Render()
print("Generando...\n")
bitmap.glInit()
bitmap.glCreateWindow(600,600)
bitmap.glClear()
bitmap.glLoad('./pengin.obj', V3(325,300,0), V3(25,25,25))
bitmap.glFinish('finito.bmp')
bitmap.glFinishBuffer('buffer.bmp')