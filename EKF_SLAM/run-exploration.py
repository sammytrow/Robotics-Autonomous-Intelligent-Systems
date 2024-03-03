#!/usr/bin/env python3
from collections import *
import sys
import asyncio
import cozmo
from frame2d import Frame2D
from cmap import CozmoMap, plotMap, loadU08520Map, update_occupancy, OccupancyGrid, Coord2DGrid
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from cozmo_interface import cube_sensor_model, velocity_to_track_speed, motion_model, cliff_sensor_model
from cozmo.util import degrees, distance_mm
import math
import numpy as np
from exploration import *
from cube_exp_means_variance import cube_means, cube_variance
from SLAM import *


avail_landmarks = [
    cozmo.objects.LightCube1Id,
    cozmo.objects.LightCube2Id,
    cozmo.objects.LightCube3Id,
    cozmo.objects.CustomObjectMarkers.Circles2,
    cozmo.objects.CustomObjectMarkers.Circles3,
    cozmo.objects.CustomObjectMarkers.Circles4,
    cozmo.objects.CustomObjectMarkers.Circles5,
    cozmo.objects.CustomObjectMarkers.Diamonds2,
    cozmo.objects.CustomObjectMarkers.Diamonds3,
    cozmo.objects.CustomObjectMarkers.Diamonds4,
    cozmo.objects.CustomObjectMarkers.Diamonds5,
    cozmo.objects.CustomObjectMarkers.Hexagons2,
    cozmo.objects.CustomObjectMarkers.Hexagons3,
    cozmo.objects.CustomObjectMarkers.Hexagons4,
    cozmo.objects.CustomObjectMarkers.Hexagons5,
    cozmo.objects.CustomObjectMarkers.Triangles2,
    cozmo.objects.CustomObjectMarkers.Triangles3,
    cozmo.objects.CustomObjectMarkers.Triangles4,
    cozmo.objects.CustomObjectMarkers.Triangles5
]

def round_20(n, base=20):
    return base * round(float(n) / base)


def printFrameList(frameList, logFile, end="\n"):
    print("[", file=logFile)
    for idx in range(len(frameList)):
        t = frameList[idx][0]
        x = frameList[idx][1].x()
        y = frameList[idx][1].y()
        a = frameList[idx][1].angle()
        print("   (%d, Frame2D.fromXYA(%f,%f,%f))" % (t, x, y, a), end="", file=logFile)
        if idx != len(frameList) - 1:
            print(",", file=logFile)
    print("]", file=logFile, end=end)


def printList(dataList, logFile, end="\n"):
    print("[", file=logFile)
    for idx in range(len(dataList)):
        t = dataList[idx][0]
        v = dataList[idx][1]
        print("(" + str(t) + ", " + str(v) + ")", end="", file=logFile)
        if idx != len(dataList) - 1:
            print(",", file=logFile)
    print("]", file=logFile, end=end)

async def cozmo_program(robot: cozmo.robot.Robot):
    # load map and create plot
    m = loadU08520Map()
    plt.ion()
    plt.show()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, aspect=1)

    ax.set_xlim(m.grid.minX(), m.grid.maxX())
    ax.set_ylim(m.grid.minY(), m.grid.maxY())

    plotMap(ax, m)

    # setup belief grid data structures
    grid = m.grid
    minX = grid.minX()
    maxX = grid.maxX()
    minY = grid.minY()
    maxY = grid.maxY()
    tick = grid.gridStepSize
    numX = grid.gridSizeX
    numY = grid.gridSizeY
    gridXs = []
    gridYs = []
    gridCs = []
    for xIndex in range(0, numX):
        for yIndex in range(0, numY):
            gridXs.append(minX + 0.5 * tick + tick * xIndex)
            gridYs.append(minY + 0.5 * tick + tick * yIndex)
            gridCs.append((1, 1, 1))
    #plt.scatter(gridXs, gridYs, c=gridCs)

    # TODO try me out: choose which robot angles to compute probabilities for
    # gridAs = [0] # just facing one direction # <! sam: single direction or multiple as below \/
    gridAs = np.linspace(0, 2 * math.pi, 11)  # facing different possible directions

    # TODO try me out: choose which cubes are considered
    # cubeIDs = [cozmo.objects.LightCube3Id] # <! sam: single cube or multiple as below \/
    cubeIDs = [cozmo.objects.LightCube1Id, cozmo.objects.LightCube2Id,
               cozmo.objects.LightCube3Id]  # single cube or multiple
    wallIDs = dict.fromkeys([k for k in m.landmarks.keys()
                             if k not in [cozmo.objects.LightCube1Id,
                                          cozmo.objects.LightCube2Id,
                                          cozmo.objects.LightCube3Id]],
                            False)
    #print("print wallids:S " + str(wallIDs))
    # precompute inverse coordinate frames for all x/y/a grid positions

    index = 0
    positionInverseFrames = []  # 3D array of Frame2D objects (inverse transformation of every belief position x/y/a on grid)
    for xIndex in range(0, numX):
        yaArray = []
        x = minX + 0.5 * tick + tick * xIndex
        for yIndex in range(0, numY):
            aArray = []
            y = minY + 0.5 * tick + tick * yIndex
            for a in gridAs:
                aArray.append(Frame2D.fromXYA(x, y, a).inverse())
            yaArray.append(aArray)
        positionInverseFrames.append(yaArray)

    target = Frame2D.fromXYA(280, 700, 0)
    if len(sys.argv) == 2:
        logName = sys.argv[1] + ".py"
    else:
        logName = "sensorLog.py"
    initialize_plot()

    robotPose = Frame2D.fromPose(robot.pose)  # robot pose
    # cubePose = Frame2D.fromPose(robot.world.light_cubes[0].pose)  # cube pose
    start_pose = [100, 100, 0]
    plot_items_vor = []
    plot_vor_counter = 0
    slam_returns_mew = []
    slam_returns_cov = []
    logFile = open(logName, 'w')
    #print("robotFrames = ", file=logFile, end="")
    prev_mew = np.zeros((1, 60))

    covariance = np.zeros((60, 60))
    cube_plotted = []
    seen_landmarks = []
    #print("prev_mew: ", prev_mew)
    await asyncio.sleep(5)
    while True:
        #print("pose: " + str(robotPose))  # print robot pose
        movement, time = explore(robotPose)  # get movement and  next direction to move
        movement = Frame2D.fromXYA(movement[0], movement[1], movement[2])  # convert to frame2d
        #print("movement: " + str(movement))  # print movement
        #print("time: " + str(time))  # print time

        left_speed, right_speed, pose = velocity_to_track_speed(robotPose, movement,
                                                                time)  # get left and right wheel speeds
        #print("speed: " + str(left_speed) + str(right_speed))  # print wheel speeds

        for t in range(int(time)):
            prev_pose = robotPose
            robotPose = Frame2D.fromPose(robot.pose)  # get robot pose
            vel_vector = []
            vel_vector.append(right_speed)  # first entry of vector is the forward velocity
            angular_v = abs(pose[2] - prev_pose.angle()) / (t + 1)
            #if angular_v < 0.5:
            #    angular_v = 0
            vel_vector.append(angular_v)

            #mm_prob_result, mm_cov_mat = motion_model(robotPose, vel_vector, prev_pose, t + 1)
            prev_pose = robotPose
            robot.drive_wheel_motors(left_speed, right_speed)
            # cozmo.run_program(go_to_cube())

            for cubeID in cubeIDs:
                cube = robot.world.get_light_cube(cubeID)
                visible = False
                if cube is not None and cube.is_visible:
                    if cube not in seen_landmarks:
                        seen_landmarks.append(cube)

            #print("vel: ", vel_vector)
            slam_mew, slam_cov = SLAM(prev_mew, covariance, vel_vector, seen_landmarks, avail_landmarks, 1)
            prev_mew = slam_mew
            #print("prev_mew: ", prev_mew)
            covariance = slam_cov

            if robot.is_cliff_detected:
                robot.stop_all_motors()
                t = time
                #print("Cliff detected, rotate 90 degrees to the right and move")
                await asyncio.sleep(1)
                robot.drive_wheel_motors(-80, -80)
                await asyncio.sleep(2)
                robot.drive_wheel_motors(45, -45)
                await asyncio.sleep(3)

            for cubeID in cubeIDs:
                cube = robot.world.get_light_cube(cubeID)
                if cube is not None and cube.is_visible:
                    # go_to_cube(cubePose, robot, cubeIDs)
                    cubePose = m.landmarks[cubeID]
                    relativePose = robotPose.inverse().mult(cubePose)
                    cubePose2y = robotPose.x() + relativePose.x() #start_pose[1]
                    cubePose2x = robotPose.y() + relativePose.y() #start_pose[0]
                    #print("cubePose2x: " + str(cubePose2x) + " | cubePose2y: ", cubePose2y)
                    x = cubePose2y
                    y = cubePose2x

                    #if plot_vor_counter < 3:
                    #    plot_items_vor.append([[cubeID],[x, y]])
                    #    plot_vor_counter += 1

                    #if plot_vor_counter == 3:
                    #    run_voronoi(plot_items_vor)
                    #    plot_vor_counter += 1

                    #print("cube: ", cubeID)
                    plt.scatter(x, y)
                    plt.annotate(str(cubeID), xy=(x, y))
                    plt.draw()
                    plt.pause(0.01)
                    cube_plotted.append(cubeID)

            if robotPose.x() == 300:
                robot.stop_all_motors()  # stop robot
            await asyncio.sleep(0.1)  # sleep for 0.1 seconds
        robot.stop_all_motors()  # stop robot
        slam_returns_mew.append([t, slam_mew])
        slam_returns_cov.append([t, slam_cov])

        cubeVisibility = {}
        cubeRelativeFrames = {}
        for cubeID in cubeIDs:
            cube = robot.world.get_light_cube(cubeID)

            relativePose = Frame2D()
            visible = False
            if cube is not None and cube.is_visible:
                #print("Visible: " + cube.descriptive_name + " (id=" + str(cube.object_id) + ")")
                cubePose = Frame2D.fromPose(cube.pose)
                #print("   pose: " + str(cubePose))
                relativePose = robotPose.inverse().mult(cubePose)
                #print("   relative pose (2D): " + str(relativePose))
                visible = True
            cubeVisibility[cubeID] = visible
            cubeRelativeFrames[cubeID] = relativePose


        # compute position beliefs over grid (and store future visualization colors in gridCs)
        index = 0
        for xIndex in range(0, numX):
            # x = minX+0.5*tick+tick*xIndex
            for yIndex in range(0, numY):
                # y = minY+0.5*tick+tick*yIndex
                maxP = 0
                for aIndex in range(len(gridAs)):
                    invFrame = positionInverseFrames[xIndex][yIndex][aIndex]  # precomputes inverse frames
                    p = 1.  # empty product of probabilities (initial value) is 1.0
                    for cubeID in cubeIDs:
                        # compute pose of cube relative to robot
                        temp = 400  # min(mean_list, key=lambda x: abs(x - m.landmarks[cubeID].x()))

                        relativeTruePose = invFrame.mult(m.landmarks[cubeID])
                        # overall probability is the product of individual probabilities (assumed conditionally independent)

                        if cubeVisibility[cubeID]:
                            p = abs(p * cube_sensor_model(relativeTruePose, cubeVisibility[cubeID],
                                                          cubeRelativeFrames[cubeID]))

                            # if cubeVisibility[cubeID]:
                            # gridCs = update_occupancy(gridCs, relativeTruePose, cubeRelativeFrames[cubeID],
                            #                          vision_cone)
                            # if x_pose:
                            # index_pos = int(43 + (44 * (round_20(x_pose) / 20)) - (round_20(840 - y_pose) / 20))
                            #    gridCs[index_pos] = (0, 0, 0)

                    # maximum probability over different angles is the one visualized in the end
                    if maxP < p:
                        maxP = p
                index = index + 1

        await asyncio.sleep(1)
        plt.savefig('foo.png')
        printList(slam_returns_mew, logFile)
        print(slam_returns_mew)

        printList(slam_returns_cov, logFile)
        print(slam_returns_cov)
    """#print("Y: " + str(plot_y_pose))
    y = [(y * -1) + 100 for y in plot_y_pose]
    x = [x + 100 for x in plot_x_pose]
    #print("Y update: " + str(y))
    plt.plot(y, x)
    plt.scatter(y, x)
    printFrameList(robotFrames, logFile)
    print(robotFrames)
    logFile.close()
    plt.show()
    pop.set_facecolor(gridCs)
    plt.draw()
    plt.pause(0.001)
    plt.savefig('foo.png')
    plt.show()"""


cozmo.robot.Robot.drive_off_charger_on_connect = False
cozmo.run_program(cozmo_program, use_3d_viewer=True, use_viewer=True)