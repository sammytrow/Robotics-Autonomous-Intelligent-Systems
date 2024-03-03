#!/usr/bin/env python3

# Copyright (c) 2019 Matthias Rolf, Oxford Brookes University

'''

'''
import numpy as np
import math

from matplotlib import pyplot as plt
import matplotlib.patches as patches

import cozmo

from frame2d import Frame2D
from object2d import Object2D


class Coord2D:
    def __init__(self, xp: float, yp: float):
        self.x = xp
        self.y = yp
    def __str__(self):
        return "[x="+str(self.x)+",y="+str(self.y)+"]"

class Coord2DGrid:
    def __init__(self, xp: int, yp: int):
        self.x = xp
        self.y = yp
    def __str__(self):
        return "[index-x="+str(self.x)+",index-y="+str(self.y)+"]"

class OccupancyGrid:
    FREE = 0
    OCCUPIED = 1
    HOLE = -1
    def __init__(self, start: Coord2D, stepSize, sizeX, sizeY):
        self.gridStart = start
        self.gridStepSize = stepSize # step size is notionally in units of mm
        self.gridSizeX = sizeX # grid size is in units of stepSize
        self.gridSizeY = sizeY
        self.gridData = np.zeros((sizeX,sizeY),int)

    def validateIndex(self, c: Coord2DGrid):
        if c.x < 0 or self.gridSizeX <= c.x:
            raise Exception("OccupancyGrid coordinate ", str(c), " is out of bounds.")
        if c.y < 0 or self.gridSizeY <= c.y:
            raise Exception("OccupancyGrid coordinate ", str(c), " is out of bounds.")

    def validateIndexStop(self, c: Coord2DGrid):
        if c.x < -1 or self.gridSizeX < c.x:
            raise Exception("OccupancyGrid coordinate ", str(c), " is out of bounds for index stop.")
        if c.y < -1 or self.gridSizeY < c.y:
            raise Exception("OccupancyGrid coordinate ", str(c), " is out of bounds for index stop.")
    
    def float2grid(self, c: Coord2D):
        xIndex = round( (c.x - self.gridStart.x) / self.gridStepSize )
        yIndex = round( (c.y - self.gridStart.y) / self.gridStepSize )
        ci = Coord2DGrid(xIndex,yIndex)
        self.validateIndex(ci)
        return ci

    def grid2float(self, ci: Coord2DGrid):
        self.validateIndex(ci)
        x = self.gridStart.x + ci.x*self.gridStepSize
        y = self.gridStart.y + ci.y*self.gridStepSize
        return Coord2D(x,y)

    def frame2D2Coord2DGrid(self,frame: Frame2D):
        x = frame.x()
        y = frame.y()
        # transform the native x and y coordinates into grid points.
        x = round((x-self.gridStart.x)/self.gridStepSize)
        y = round((y-self.gridStart.y)/self.gridStepSize)
        xy = Coord2DGrid(x,y)
        self.validateIndex(xy)
        return xy
    
    def coord2D2GridFrame2D(self,coord: Coord2D,a):
        x = (coord.x*self.gridStepSize)+self.gridStart.x
        y = (coord.y*self.gridStepSize)+self.gridStart.y
        return Frame2D.fromXYA(x,y,a)

    def isFreeGrid(self, ci: Coord2DGrid):
        self.validateIndex(ci)
        return self.gridData[int(ci.x),int(ci.y)] == self.FREE

    def isFree(self, c: Coord2D):
        return self.isFreeGrid(self.float2grid(c))

    def isOccupiedGrid(self, ci: Coord2DGrid):
        self.validateIndex(ci)
        return self.gridData[int(ci.x),int(ci.y)] != self.FREE
        #return self.gridData[int(ci.x),int(ci.y)] == self.OCCUPIED

    def isOccupied(self, c: Coord2D):
        return self.isOccupiedGrid(self.float2grid(c))

    def isHoleGrid(self, ci: Coord2DGrid):
        self.validateIndex(ci)
        return self.gridData[int(ci.x),int(ci.y)] == self.HOLE

    def isHole(self, c: Coord2D):
        return self.isHoleGrid(self.float2grid(c))

    def setGridState(self, start: Coord2DGrid, end: Coord2DGrid, state=None):
        self.validateIndex(start)
        self.validateIndexStop(end)
        setState = state
        if setState is None:
           setState = self.FREE
        for x in range(start.x, end.x):
            for y in range(start.y, end.y):
                self.gridData[x,y] = setState

    def setFree(self, start: Coord2DGrid, end: Coord2DGrid):
        self.setGridState(start, end, self.FREE)

    def setOccupied(self, start: Coord2DGrid, end: Coord2DGrid):
        self.setGridState(start, end, self.OCCUPIED)

    def setHole(self, start: Coord2DGrid, end: Coord2DGrid):
        self.setGridState(start, end, self.HOLE)                    
        
    def minX(self):
        return self.gridStart.x - 0.5*self.gridStepSize

    def minY(self):
        return self.gridStart.y - 0.5*self.gridStepSize

    def maxX(self):
        return self.gridStart.x + (self.gridSizeX - 0.5)*self.gridStepSize

    def maxY(self):
        return self.gridStart.y + (self.gridSizeY - 0.5)*self.gridStepSize

    def __str__(self):
        g = ""
        for x in range(0, self.gridSizeX):
            line = ""
            for y in range(0, self.gridSizeY):
                if self.gridData[x,y] == self.FREE:
                    line = line+".. "
                elif self.gridData[x,y] == self.OCCUPIED:
                    line = line+"XX "
                elif self.gridData[x,y] == self.HOLE:
                    line = line+"OO "
            g = g+line+"\n"
        return g


class MapCustomObject:
        def __init__(self,object_type,x_size,y_size,z_size,marker_width,marker_height):
            self._objectType = object_type
            self.descriptive_name=None
            self.is_unique=False
            self.marker_height_mm=marker_height
            self.marker_width_mm=marker_width
            self.x_size_mm=x_size
            self.y_size_mm=y_size
            self.z_size_mm=z_size

class Wall:
        def __init__(self,wallID,typeID,xpos,ypos,horizontal):
            self.objectID = wallID
            self.objectType = typeID
            self.x = xpos
            self.y = ypos
            if horizontal:
               self.theta = 0
            else:
               self.theta = math.pi/2

class WallType:
        def __init__(self,typeID,width,height,marker):
            self._ID = typeID
            self.width = width
            self.height = height
            self.marker = marker
    
class TypeMap:
        def __init__(self):
               self.types = {}
               self.byType = {}
               self.byID = {}
               self.markers = {}

        def addType(self,typeID,marker,width,height,mWidth,mHeight):
            if typeID in self.types:
               raise TypeError("Duplicate types are not allowed")
            if marker in self.markers.values():
               raise ValueError("A different type already has marker type {0}".format(marker))
            self.types[typeID] = MapCustomObject(typeID,10,width,height,mWidth,mHeight)
            self.markers[typeID] = marker

        def addObject(self,objectID,typeID):
            if typeID not in self.types:
               raise TypeError("Undefined object type {0}".format(typeID))
            if typeID not in self.byType:
               self.byType[typeID] = [objectID]
            else:
                self.byType[typeID].append(objectID)
            self.byID[objectID] = typeID                        

# Map storing an occupancy grid and a set of landmarks
# landmarks are stored in a dictionary mapping landmark-IDs onto an Object2D
class CozmoMap:
        def __init__(self, grid, landmarks, targets=None):
            self.grid = grid
            self.landmarks = landmarks
            self.targets = targets
            self.wallTypeMap = TypeMap()

        def addWallType(self, wall: WallType):
            self.wallTypeMap.addType(wall._ID,wall.marker,wall.width,wall.height,20,20)

        def addWall(self, wall: Wall):
            if wall.objectType not in self.wallTypeMap.types:
               raise ValueError("Attempted to add a wall of unknown type {0}".format(wall.objectType))
            self.wallTypeMap.addObject(wall.objectID,wall.objectType)
            gridStep = self.grid.gridStepSize
            # for the moment allow only horizontal or vertical walls.
            while not self.grid.isOccupied(Coord2D(wall.x, wall.y)):
               if wall.theta == 0:
                  self.grid.setOccupied(Coord2DGrid(math.floor((wall.x-self.grid.gridStart.x)/gridStep-self.wallTypeMap.types[wall.objectType].y_size_mm/(2*gridStep)),
                                                    math.floor((wall.y-self.grid.gridStart.y)/gridStep-0.5)),
                                        Coord2DGrid(math.ceil((wall.x-self.grid.gridStart.x)/gridStep+self.wallTypeMap.types[wall.objectType].y_size_mm/(2*gridStep)),
                                                    math.ceil((wall.y-self.grid.gridStart.y)/gridStep+0.5)))
               else:
                  self.grid.setOccupied(Coord2DGrid(math.floor((wall.x-self.grid.gridStart.x)/gridStep-0.5),
                                                    math.floor((wall.y-self.grid.gridStart.y)/gridStep-self.wallTypeMap.types[wall.objectType].y_size_mm/(2*gridStep))),
                                        Coord2DGrid(math.ceil((wall.x-self.grid.gridStart.x)/gridStep+0.5),
                                                    math.ceil((wall.y-self.grid.gridStart.y)/gridStep+self.wallTypeMap.types[wall.objectType].y_size_mm/(2*gridStep))))
            self.landmarks[wall.objectID] = Object2D(Frame2D.fromXYA(wall.x,wall.y,wall.theta),
                                                     10,
                                                     self.wallTypeMap.types[wall.objectType].y_size_mm)

def loadU08520Map():
        # based on a 60cm x 80cm layout
        sizeX = 32
        #sizeY = 42
        sizeY = 44
        grid = OccupancyGrid(Coord2D(-10,-10), 20.0, sizeX, sizeY)
        grid.setOccupied(Coord2DGrid(0,0), Coord2DGrid(sizeX,sizeY))
        grid.setFree(Coord2DGrid(1,1), Coord2DGrid(sizeX-1,sizeY-1))
        grid.setOccupied(Coord2DGrid(16,21), Coord2DGrid(sizeX,23))
        lightCubeDim = 44 # assume 44 mm light cube size
        
        landmarks = {
                cozmo.objects.LightCube1Id : Object2D(Frame2D.fromXYA(540,40,0),lightCubeDim,lightCubeDim),
                cozmo.objects.LightCube2Id : Object2D(Frame2D.fromXYA(40,440,0),lightCubeDim,lightCubeDim),
                cozmo.objects.LightCube3Id : Object2D(Frame2D.fromXYA(540,500,0),lightCubeDim,lightCubeDim) }
        
        targets = [Frame2D.fromXYA(400,760,0)]
        
        return CozmoMap(grid,landmarks,targets)

def loadMap(sizeX_cm, sizeY_cm, gridSize_mm, wall_types, walls, light_cubes, tgts):
        sizeX = 10*sizeX_cm//gridSize_mm+20//gridSize_mm
        sizeY = 10*sizeY_cm//gridSize_mm+20//gridSize_mm
        grid = OccupancyGrid(Coord2D(-10,-10), gridSize_mm, sizeX, sizeY)
        grid.setOccupied(Coord2DGrid(0,0), Coord2DGrid(sizeX,sizeY))
        grid.setFree(Coord2DGrid(1,1), Coord2DGrid(sizeX-1,sizeY-1))
        lightCubeDim = 44
        landmarks = {}
        for cube in light_cubes:
            landmarks[cube] = Object2D(Frame2D.fromXYA(light_cubes[cube][0],light_cubes[cube][1],0),lightCubeDim,lightCubeDim)

        targets = []
        for target in tgts:
            targets.append(Frame2D.fromXYA(target[0],target[1],0))

        map = CozmoMap(grid,landmarks,targets)
        for wType in wall_types:
            map.addWallType(wType)
        for wall in walls:
            map.addWall(wall)

        return map

def plotMap(ax, m : CozmoMap, color="blue"):
        grid = m.grid
        minX = grid.minX()
        maxX = grid.maxX()
        minY = grid.minY()
        maxY = grid.maxY()
        tick = grid.gridStepSize
        numX = grid.gridSizeX
        numY = grid.gridSizeY
        for xIndex in range (0,numX+1):
                x = minX + xIndex*tick
                bold = 0.8 if (xIndex-1)%5==0 else 0.4
                plt.plot([x, x], [minY,maxY], color, alpha=bold, linewidth=bold)
        for yIndex in range (0,numY+1):
                y = minY + yIndex*tick
                bold = 0.8 if (yIndex-1)%5==0 else 0.4
                plt.plot([minX,maxX], [y, y], color, alpha=bold, linewidth=bold)

        for xIndex in range (0,numX):
                for yIndex in range (0,numY):
                        if grid.isOccupiedGrid(Coord2DGrid(xIndex,yIndex)):
                                rect = patches.Rectangle((minX+tick*xIndex,minY+tick*yIndex),tick,tick,linewidth=1,edgecolor='blue',facecolor='blue',zorder=0)
                                ax.add_patch(rect)

        for landmark in m.landmarks.items():
                x = landmark[1].pose.x()
                y = landmark[1].pose.y()
                a = landmark[1].pose.angle()
                '''walls in Cozmo think of the width as the long axis, height as vertical height,
                   and thickness (what should be the 'height' dimension in matplotlib's Rectangle)
                   as fixed at 10mm. Assiging the x dimension to 'height' and y dimension to 'width'
                   is therefore used as a notational convenience. Furthermore Cozmo angles are in radians,
                   whilst matplotlib uses degrees. Hence the conversion to angle below. As if that
                   weren't enough, matplotlib's origin is at the lower left corner, whilst Cozmo's
                   pose origin is at the centre of the object. The old Microsoft 'bounding-box'-style
                   definition of a graphical object once again makes things more complicated than
                   than they need to be.
                '''
                width = landmark[1].ydim
                height = landmark[1].xdim
                angle = 180/math.pi*a
                #halfDiagonal = math.sqrt(width**2+height**2)/2
                mplOrigin = (x+((height/2)*math.sin(a)-(width/2)*math.cos(a)),y-((width/2)*math.sin(a)+(height/2)*math.cos(a)))
                # show cubes in blue
                if (landmark[0] == cozmo.objects.LightCube1Id or
                    landmark[0] == cozmo.objects.LightCube2Id or
                    landmark[0] == cozmo.objects.LightCube3Id):
                    colour = 'blue'
                # and walls in green
                else:
                    colour = 'green'
                rect = patches.Rectangle(mplOrigin,width,height,angle=angle,linewidth=2,edgecolor=colour,facecolor="none",zorder=0)
                ax.add_patch(rect)

        for target in m.targets:
                x = target.x()
                y = target.y()
                size = 20
                ell = patches.Ellipse((x,y),2*size,2*size,linewidth=2,edgecolor='blue',facecolor=[0,0,1,0.5],zorder=0)
                ax.add_patch(ell)
                ell2 = patches.Ellipse((x,y),4*size,4*size,linewidth=2,linestyle=":",edgecolor='blue',facecolor="none",zorder=0)
                ax.add_patch(ell2)

def is_in_map( m : CozmoMap , x, y):
    if x < m.grid.minX() or m.grid.maxX() < x:
        return False
    if y < m.grid.minY() or m.grid.maxY() < y:
        return False
    return True

def round_20(n, base=20):
    return base * round(float(n)/base)

def update_occupancy(current_map, true_pose, measured_pose, vision_cone):
    for i in range(len(current_map)):
        cellx = int(((i - 43) / 44) * 20)
        celly = int(840 - (((43 + (44 * (x / 20))) - i) * 20))

        print("cellx pose: " + str(cellx) + " celly pose: " + str(celly))
        if vision_cone[0] < measured_pose.x() < vision_cone[1]:# and vision_cone[2] < measured_pose.y() < vision_cone[3]:
            p = 1 #p(Mi| Zi, Xi): p(Mi) = P(Zi|Xi,Mi)* P(Xi|Mi) * P(Mi)/ P(Zi|Xi) P(Xi)
            if p >= 0.5:
                current_map.gridData[cell] = current_map.OCCUPIED
            else:
                current_map.gridData[cell] = current_map.FREE

    return current_map