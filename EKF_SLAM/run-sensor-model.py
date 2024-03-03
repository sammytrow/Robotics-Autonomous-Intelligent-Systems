#!/usr/bin/env python3
from collections import *
import sys
import asyncio
import cozmo
from frame2d import Frame2D
from cmap import CozmoMap, plotMap, loadU08520Map, update_occupancy, OccupancyGrid, Coord2DGrid
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from cozmo_interface import cube_sensor_model, velocity_to_track_speed, motion_model
from cozmo.util import degrees, Pose
import math
import numpy as np
from cube_exp_means_variance import cube_means, cube_variance
from localization import EKF_localisation
#from SLAM import SLAM
import math

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


def target_breakdown(target: Frame2D):
    time = []
    pose_change = []
    pose_change.append(Frame2D.fromXYA(0, target.y(), 0))
    time.append(round((target.y() / 100) * 2))
    if target.x() > 0:
        pose_change.append(Frame2D.fromXYA(0, target.y(), math.radians(90)))
        time.append(2)
    elif target.x() < 0:
        pose_change.append(Frame2D.fromXYA(0, target.y(), math.radians(-90)))
        time.append(2)  #
    pose_change.append(Frame2D.fromXYA(target.x(), target.y(), 0))
    time.append(round((target.y() / 100) * 2))
    return pose_change, time

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
    pop = plt.scatter(gridXs, gridYs, c=gridCs)

    # TODO try me out: choose which robot angles to compute probabilities for
    # gridAs = [0] # just facing one direction # <! sam: single direction or multiple as below \/
    gridAs = np.linspace(0, 2 * math.pi, 11)  # facing different possible directions

    # TODO try me out: choose which cubes are considered
    # cubeIDs = [cozmo.objects.LightCube3Id] # <! sam: single cube or multiple as below \/
    cubeIDs = [cozmo.objects.LightCube1Id, cozmo.objects.LightCube2Id, cozmo.objects.LightCube3Id]
    wallIDs = dict.fromkeys([k for k in m.landmarks.keys()
                             if k not in [cozmo.objects.LightCube1Id,
                                          cozmo.objects.LightCube2Id,
                                          cozmo.objects.LightCube3Id]],
                            False)
    print("print wallids:S " + str(wallIDs))
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

    robotFrames = []
    plot_x_pose = []
    plot_y_pose = []
    mm_returns = []
    loc_returns = []
    slam_returns = []
    logFile = open(logName, 'w')
    print("robotFrames = ", file=logFile, end="")
    robotPose = Frame2D.fromPose(robot.pose)

    start = Frame2D.fromXYA(0, 0, 0)
    # end = Frame2D.fromXYA(0, 600, 0)
    end, time = target_breakdown(target)
    # main loop

    seen_landmarks = []
    prev_pose = robotPose
    loc_map = []

    for cubeID in cubeIDs:
        cube = robot.world.get_light_cube(cubeID)
        loc_map.append(cube)
        print("cube: " + str(cubeID))
    for wallID in wallIDs:
        loc_map.append(wallID)
        print("wall: " + str(wallID))
    print("Loc map: " + str(loc_map))
    starting = 1
    updated_mew = [0, 0, 0]
    updated_cov = []

    loc_map = m.landmarks
    while True:
        # for i in range(1000):
        for step in range(len(end)):
            left_speed, right_speed, pose = velocity_to_track_speed(start, end[step], time[step])
            print("speed: " + str(left_speed) + str(right_speed))

            for t in range(int(time[step]) * 10):
                robotPose = Frame2D.fromPose(robot.pose)

                # velocity before being sent into motion model
                vel_vector = []
                vel_vector.append(right_speed)  # first entry of vector is the forward velocity
                angular_v = abs(pose[2] - prev_pose.angle()) / (t+1)
                vel_vector.append(angular_v)

                mm_prob_result, mm_cov_mat = motion_model(robotPose, vel_vector, prev_pose, t+1)  # mm takes in a time argument as well, I think this would be t

                plot_x_pose.append(robotPose.x())
                plot_y_pose.append(robotPose.y())
                mm_returns.append([t, mm_prob_result])

                robot.drive_wheel_motors(left_speed, right_speed)
                start = end[step]

                updated_mew, updated_cov = EKF_localisation(prev_pose, robotPose, mm_cov_mat, vel_vector, seen_landmarks, avail_landmarks, loc_map, 1, updated_mew, starting)
                starting = 0

                for cubeID in cubeIDs:
                    cube = robot.world.get_light_cube(cubeID)
                    visible = False
                    if cube is not None and cube.is_visible:
                        if cube not in seen_landmarks:
                            seen_landmarks.append(cube)

                robotFrames.append((t, robotPose))
                prev_pose = robotPose

                # slam_result =  SLAM(prev_mew, covariance, v_vect, seen_landmarks, avail_landmarks, t)
                # slam_returns.append([t, slam_result])

                if robotPose.x() == 300:
                    robot.stop_all_motors()
                await asyncio.sleep(0.1)
            robot.stop_all_motors()

            vision_cone = [robotPose.x(), (robotPose.x() + 400)]  # , robotPose.y(), (robotPose.y() + 0)]

            cubeVisibility = {}
            cubeRelativeFrames = {}
            for cubeID in cubeIDs:
                cube = robot.world.get_light_cube(cubeID)

                relativePose = Frame2D()
                visible = False
                if cube is not None and cube.is_visible:
                    if cube not in seen_landmarks:
                        seen_landmarks.append(seen_landmarks)
                    print("Visible: " + cube.descriptive_name + " (id=" + str(cube.object_id) + ")")
                    cubePose = Frame2D.fromPose(cube.pose)
                    print("   pose: " + str(cubePose))
                    relativePose = robotPose.inverse().mult(cubePose)
                    print("   relative pose (2D): " + str(relativePose))
                    #basic_map = plot_item(relativeTruePose, "cube", cubeID, basic_map)
                    visible = True
                cubeVisibility[cubeID] = visible
                cubeRelativeFrames[cubeID] = relativePose

            wallVisibility = {}
            wallRelativeFrames = {}
            for wallID in wallIDs:
                wall = robot.world.get_light_wall(wallID)

                relativePose = Frame2D()
                visible = False
                if wall is not None and wall.is_visible:
                    if wall not in seen_landmarks:
                        seen_landmarks.append(seen_landmarks)
                    print("Visible: " + wall.descriptive_name + " (id=" + str(wall.object_id) + ")")
                    wallPose = Frame2D.fromPose(wall.pose)
                    print("   pose: " + str(wallPose))
                    relativePose = robotPose.inverse().mult(wallPose)
                    print("   relative pose (2D): " + str(relativePose))
                    #basic_map = plot_item(relativeTruePose, "wall", wallID, basic_map)
                    visible = True
                wallVisibility[wallID] = visible
                wallRelativeFrames[wallID] = relativePose
            """ Is this how we want to implement it? do we send a position to the kinamatics?
            I think since we need cozmo to explore the environment and make it to a certain location sending through a position
            to move to may be more applicable """

            # compute position beliefs over grid (and store future vis

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

                                if cubeVisibility[cubeID]:
                                    gridCs = update_occupancy(gridCs, relativeTruePose, cubeRelativeFrames[cubeID],
                                                              vision_cone)
                                # if x_pose:
                                # index_pos = int(43 + (44 * (round_20(x_pose) / 20)) - (round_20(840 - y_pose) / 20))
                                #    gridCs[index_pos] = (0, 0, 0)
                        for wallID in wallIDs:
                            # compute pose of wall relative to robot
                            temp = 400  # min(mean_list, key=lambda x: abs(x - m.landmarks[wallID].x()))

                            relativeTruePose = invFrame.mult(m.landmarks[wallID])
                            # overall probability is the product of individual probabilities (assumed conditionally independent)

                            if wallVisibility[wallID]:
                                p = 1  # abs(p * cube_sensor_model(relativeTruePose, wallVisibility[wallID],
                                #                         wallRelativeFrames[wallID]))

                                if wallVisibility[wallID]:
                                    gridCs = update_occupancy(gridCs, relativeTruePose, wallRelativeFrames[wallID],
                                                              vision_cone)
                        # maximum probability over different angles is the one visualized in the end
                        if maxP < p:
                            maxP = p
                    index = index + 1

        print("Y: " + str(plot_y_pose))
        y = [(y * -1) + 100 for y in plot_y_pose]
        x = [x + 100 for x in plot_x_pose]
        plt.plot(y, x)
        plt.scatter(y, x)
        printFrameList(robotFrames, logFile)

        #print(robotFrames)
        print("Motion model results = ", file=logFile, end="")
        printList(mm_returns, logFile)

        #print(mm_returns)
        loc_returns.append([t, updated_mew])
        print("location model results = ", file=logFile, end="")
        printList(loc_returns, logFile)
        #print(loc_returns)

        # printList(slam_returns, logFile)
        # print(slam_returns)

        logFile.close()
        plt.show()
        pop.set_facecolor(gridCs)
        plt.draw()
        plt.pause(0.001)
        plt.savefig('foo.png')

        await asyncio.sleep(100)

cozmo.robot.Robot.drive_off_charger_on_connect = False
cozmo.run_program(cozmo_program, use_3d_viewer=True, use_viewer=True)
