#!/usr/bin/env python3


import asyncio

import cozmo

from frame2d import Frame2D
from cmap import CozmoMap, plotMap, loadMap, Coord2D, Wall, WallType
from matplotlib import pyplot as plt
from cozmo_interface import *
from mcl_tools import *
from cozmo_sim_world import *
from terminal_reader import WaitForChar
from gaussian import Gaussian, GaussianTable, plotGaussian
import math
import numpy as np
import threading
import time


def plotRobot(pos: Frame2D, color="orange", existingPlot=None):
    xy = np.array([[30, 35, 35, 30, -30, -30, 30],
                   [20, 15, -15, -20, -20, 20, 20],
                   [1, 1, 1, 1, 1, 1, 1]])
    xy = np.matmul(pos.mat, xy)
    if existingPlot is not None:
        existingPlot.set_xdata(xy[0, :])
        existingPlot.set_ydata(xy[1, :])
        existingPlot.set_color(color)
        return existingPlot
    else:
        line = plt.plot(xy[0, :], xy[1, :], color)
        return line[0]


def plotCube(pos: Frame2D, color="orange", existingPlot=None):
    xy = np.array([[25, -25, -25, 25, 25],
                   [25, 25, -25, -25, 25],
                   [1, 1, 1, 1, 1]])
    if pos.x() == 0 and pos.y() == 0:
        xy = np.matmul(Frame2D.fromXYA(-1000, -1000, 0).mat, xy)
    else:
        xy = np.matmul(pos.mat, xy)
    if existingPlot is not None:
        existingPlot.set_xdata(xy[0, :])
        existingPlot.set_ydata(xy[1, :])
        existingPlot.set_color(color)
        return existingPlot
    else:
        line = plt.plot(xy[0, :], xy[1, :], color)
        return line[0]


def runPlotLoop(simWorld: CozmoSimWorld, finished):
    # create plot
    plt.ion()
    plt.show()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, aspect=1)

    ax.set_xlim(m.grid.minX(), m.grid.maxX())
    ax.set_ylim(m.grid.minY(), m.grid.maxY())

    plotMap(ax, m)

    robotPlot = plotRobot(simWorld._touch_for_experiments_only__pos())
    cube1Plot = plotCube(simWorld.sense_cube_pose_global(cozmo.objects.LightCube1Id))
    cube2Plot = plotCube(simWorld.sense_cube_pose_global(cozmo.objects.LightCube2Id))
    cube3Plot = plotCube(simWorld.sense_cube_pose_global(cozmo.objects.LightCube3Id))

    # main loop
    t = 0
    while not finished.is_set():
        # update plot

        plotRobot(simWorld._touch_for_experiments_only__pos(), existingPlot=robotPlot)
        plotCube(simWorld.sense_cube_pose_global(cozmo.objects.LightCube1Id), existingPlot=cube1Plot)
        plotCube(simWorld.sense_cube_pose_global(cozmo.objects.LightCube2Id), existingPlot=cube2Plot)
        plotCube(simWorld.sense_cube_pose_global(cozmo.objects.LightCube3Id), existingPlot=cube3Plot)

        plt.draw()
        plt.pause(0.001)

        time.sleep(0.01)


def runCozmoMainLoop(simWorld: CozmoSimWorld, finished):
    time.sleep(5)

    # example of track based driving behavior
    # simWorld.drive_wheel_motors(50,-10)

    # main loop
    # TODO insert driving and navigation behavior HERE
    while not finished.is_set():
        for t in range(10):
            print("Jump", end="\r\n")
            simWorld._touch_for_experiments_only__set_pos(Frame2D.fromXYA(200, 100 + t * 50, math.pi / 2))
            time.sleep(1)


def cozmo_program(simWorld: CozmoSimWorld):
    finished = threading.Event()
    print("Starting simulation. Press Q to exit", end="\r\n")
    threading.Thread(target=runWorld, args=(simWorld, finished)).start()
    threading.Thread(target=WaitForChar, args=(finished, '[Qq]')).start()
    threading.Thread(target=runCozmoMainLoop, args=(simWorld, finished)).start()
    # running the plot loop in a thread is not thread-safe because matplotlib
    # uses tkinter, which in turn has a threading quirk that makes it
    # non-thread-safe outside the python main program.
    # See https://stackoverflow.com/questions/14694408/runtimeerror-main-thread-is-not-in-main-loop

    # threading.Thread(target=runPlotLoop, args=(simWorld,finished)).start()
    runPlotLoop(simWorld, finished)


# ----------------------------------------------Wall Types for Mapping-----------------------------------------
wallType1 = WallType(4, 90, 90, cozmo.objects.CustomObjectMarkers.Circles2)
wallType2 = WallType(5, 80, 80, cozmo.objects.CustomObjectMarkers.Diamonds2)
wallTypes = [wallType1, wallType2]

# -----------------LEVEL 1 EASY Map 1 ----------------
E1_wall1 = Wall(4, 4, 200, 100, False)
E1_wall2 = Wall(5, 5, 500, 50, False)

E1_Walls = [E1_wall1, E1_wall2]

E1lightCubes = {cozmo.objects.LightCube1Id: (300, 200),
                cozmo.objects.LightCube2Id: (400, 100),
                cozmo.objects.LightCube3Id: (100, 100)}

E1_targets = [(300, 700)]

m = loadMap(60, 80, 10, wallTypes, E1_Walls, E1lightCubes, E1_targets)

# -----------------LEVEL 1 EASY Map 2 ----------------
# E2_wall1 = Wall(4, 4, 200, 450, False)
# E2_wall2 = Wall(5, 5, 200, 300, True)

# E2_Walls = [E2_wall1, E2_wall2]

# E2lightCubes = {cozmo.objects.LightCube1Id: (200, 250),
# cozmo.objects.LightCube2Id: (200, 550),
# cozmo.objects.LightCube3Id: (300, 400)}

# E2_targets = [(500, 450)]

# m = loadMap(60, 80, 10, wallTypes, E2_Walls, E2lightCubes, E2_targets)

# --------------------LEVEL 2 MEDIUM Map 1 ------------------
# M1_wall1 = Wall(4, 4, 300, 650, False)
# M1_wall2 = Wall(5, 5, 330, 600, True)

# M1_walls = [M1_wall1, M1_wall2]

# M1_lightCubes = {cozmo.objects.LightCube1Id: (550, 50),
# cozmo.objects.LightCube2Id: (50, 750),
# cozmo.objects.LightCube3Id: (50, 350)}

# M1_targets = [(340, 650)]

# m = loadMap(60, 80, 10, wallTypes, M1_walls, M1_lightCubes, M1_targets)

# --------------------LEVEL 2 MEDIUM Map 2 ------------------

# M2_wall1 = Wall(4, 4, 450, 200, False)
# M2_wall2 = Wall(5, 5, 250, 500, False)
# M2_walls = [M2_wall1, M2_wall2]


# M2lightCubes = {cozmo.objects.LightCube1Id: (500, 40),
# cozmo.objects.LightCube2Id: (50, 400),
# cozmo.objects.LightCube3Id: (400, 400)}

# M2_targets = [(350, 500)]

# m = loadMap(60, 80, 10, wallTypes, M2_walls, M2lightCubes, M2_targets)


# ---------------LEVEL 3  HARD Map 1 --------------------

# H1_wall1 = Wall(4, 4, 500, 100, False)
# H1_wall2 = Wall(5,5, 100, 300, True)

# H1_Walls = [H1_wall1, H1_wall2]

# H1lightCubes = {cozmo.objects.LightCube1Id: (350, 300), cozmo.objects.LightCube2Id: (200, 100),
# cozmo.objects.LightCube3Id: (100, 600)}

# H1_targets = [(400, 700)]

# m = loadMap(60, 80, 10, wallTypes, H1_Walls, H1lightCubes, H1_targets)


# ---------------LEVEL 3  HARD Map 2 --------------------

# H2_wall1 = Wall(4, 4, 450, 200, True)
# H2_wall2 = Wall(5, 5, 200, 450, True)


# H2_Walls= [H2_wall1,H2_wall2]


# H2lightCubes = {cozmo.objects.LightCube1Id: (550, 50), cozmo.objects.LightCube2Id: (50, 500),
# cozmo.objects.LightCube3Id: (500, 650)}

# H2_targets = [(50, 50)]

# m = loadMap(60, 80, 10, wallTypes, H2_Walls, H2lightCubes, H2_targets)


# ---------------LEVEL 3  HARD Map 3 --------------------

# H3_wall1 = Wall(4, 4, 350, 100, True)
# H3_wall2 = Wall(5, 5, 400, 80, False)

# H3_Walls = [H3_wall1, H3_wall2]

# H3lightCubes = {cozmo.objects.LightCube1Id: (360, 200), cozmo.objects.LightCube2Id: (550, 200),
# cozmo.objects.LightCube3Id: (550, 500)}

# H3_targets = [(350, 50)]

# m = loadMap(60, 80, 10, wallTypes, H3_Walls, H3lightCubes, H3_targets)

# --------------------------------------------------------
# create some wall types and walls
# wallType1 = WallType(4, 60, 40, cozmo.objects.CustomObjectMarkers.Circles2)
# wallType2 = WallType(5, 80, 40, cozmo.objects.CustomObjectMarkers.Diamonds2)
# wall1 = Wall(4, 4, 400, 300, True)
# wall2 = Wall(5, 5, 300, 500, False)


# wallTypes = [wallType1, wallType2]
# walls = [wall1, wall2]
# lightCubes = {cozmo.objects.LightCube1Id: (540, 40),
# cozmo.objects.LightCube2Id: (40, 440),
# cozmo.objects.LightCube3Id: (540, 500)}
# targets = [(400, 760)]
# this data structure represents the map
# m = loadMap(60, 80, 10, wallTypes, walls, lightCubes, targets)


# NOTE: this code allows to specify the initial position of Cozmo on the map
simWorld = CozmoSimWorld(m, Frame2D.fromXYA(200, 350, -math.pi / 2))

cozmo_program(simWorld)
