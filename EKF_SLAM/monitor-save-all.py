#!/usr/bin/env python3


import asyncio
import sys
import signal
import cozmo
import time
import cozmo_interface

from cozmo.objects import CustomObject, CustomObjectMarkers, CustomObjectTypes

from frame2d import Frame2D 


timestamps = []
robotFrames = []
cliffSensor = []
trackSpeed = []
cubeFrames = {}
wallFrames = {}

if len(sys.argv) == 2:
	logName = sys.argv[1]+".py"
else:
	logName = "sensorLog.py"



def create_cozmo_walls(robot: cozmo.robot.Robot):
    types = [CustomObjectTypes.CustomType01,
             CustomObjectTypes.CustomType02,
             CustomObjectTypes.CustomType03,
             CustomObjectTypes.CustomType04,
             CustomObjectTypes.CustomType05,
             CustomObjectTypes.CustomType06,
             CustomObjectTypes.CustomType07,
             CustomObjectTypes.CustomType08,
             CustomObjectTypes.CustomType09,
             CustomObjectTypes.CustomType10,
             CustomObjectTypes.CustomType11,
             CustomObjectTypes.CustomType12,
             CustomObjectTypes.CustomType13,
             CustomObjectTypes.CustomType14,
             CustomObjectTypes.CustomType15,
             CustomObjectTypes.CustomType16]
    markers = [CustomObjectMarkers.Circles2,
             CustomObjectMarkers.Diamonds2,
             CustomObjectMarkers.Hexagons2,
             CustomObjectMarkers.Triangles2,
             CustomObjectMarkers.Circles3,
             CustomObjectMarkers.Diamonds3,
             CustomObjectMarkers.Hexagons3,
             CustomObjectMarkers.Triangles3,
             CustomObjectMarkers.Circles4,
             CustomObjectMarkers.Diamonds4,
             CustomObjectMarkers.Hexagons4,
             CustomObjectMarkers.Triangles4,
             CustomObjectMarkers.Circles5,
             CustomObjectMarkers.Diamonds5,
             CustomObjectMarkers.Hexagons5,
             CustomObjectMarkers.Triangles5]
    cozmo_walls = []
    for i in range(0,8):
	    cozmo_walls.append(robot.world.define_custom_wall(types[i],
                                              markers[i],
                                              200, 60,
                                              50, 50, True) )
    for i in range(8,16):
	    cozmo_walls.append(robot.world.define_custom_wall(types[i],
                                              markers[i],
                                              300, 60,
                                              50, 50, True) )
    return cozmo_walls


def printFrameList(frameList, logFile, end="\n"):
	print("[", file=logFile)
	for idx in range(len(frameList)):
		t = frameList[idx][0]
		x = frameList[idx][1].x()
		y = frameList[idx][1].y()
		a = frameList[idx][1].angle()
		print("   (%d, Frame2D.fromXYA(%f,%f,%f))" % (t,x,y,a), end="", file=logFile)
		if idx != len(frameList)-1:
			print(",", file=logFile)
	print("]", file=logFile, end=end)


def printList(dataList, logFile, end="\n"):
	print("[", file=logFile)
	for idx in range(len(dataList)):
		t = dataList[idx][0]
		v = dataList[idx][1]
		print("("+str(t)+", "+str(v)+")", end="", file=logFile)
		if idx != len(dataList)-1:
			print(",", file=logFile)
	print("]", file=logFile, end=end)



visible_walls = []
def handle_object_observed(evt, **kw):
    global visible_walls
    # This will be called whenever an EvtObjectDisappeared is dispatched -
    # whenever an Object goes out of view.
    if isinstance(evt.obj, CustomObject):
        print("Cozmo observed a %s" % str(evt.obj.object_type))
        print(evt.obj)
        if evt.obj not in visible_walls:
                visible_walls.append(evt.obj)



def cozmo_program(robot: cozmo.robot.Robot):
	global visible_walls
	global finished
	
	robot.add_event_handler(cozmo.objects.EvtObjectObserved, handle_object_observed)

	cozmo_walls = create_cozmo_walls(robot)
	
	for t in range(100):
		timestamps.append((t, time.time()))

		robotPose = Frame2D.fromPose(robot.pose)
		robotFrames.append((t,robotPose))

		cliffSensor.append((t,robot.is_cliff_detected))

		cubeIDs = (cozmo.objects.LightCube1Id,cozmo.objects.LightCube2Id,cozmo.objects.LightCube3Id)
		for cubeID in cubeIDs:
			cube = robot.world.get_light_cube(cubeID)
			if cube is not None and cube.is_visible:
				sid = str(cubeID)
				if sid not in cubeFrames:
					cubeFrames[sid] = []
				cubePose2D = Frame2D.fromPose(cube.pose)
				cubeFrames[sid].append((t,cubePose2D))

		visible = visible_walls
		visible_walls = []
		for wall in visible:
			wallID = str(wall.object_type)
			if wallID not in wallFrames:
				wallFrames[wallID] = []
			wallPose2D = Frame2D.fromPose(wall.pose)
			wallFrames[wallID].append((t,wallPose2D))

		time.sleep(0.1)

	logFile = open(logName, 'w')

	print("from frame2d import Frame2D", file=logFile)
	print("from cozmo.objects import CustomObjectTypes", file=logFile)

	print("timestamps = ", file=logFile, end="")
	printList(timestamps,logFile)

	print("robotFrames = ", file=logFile, end="")
	printFrameList(robotFrames,logFile)

	print("cliffSensor = ", file=logFile, end="")
	printList(cliffSensor,logFile)

	print("cubeFrames = {", file=logFile, end="")
	sep=False
	for c in cubeFrames:
		print( ", \n" if sep else "" , file=logFile, end="")
		sep = True
		print(c + str(" : "), file=logFile, end="")
		printFrameList(cubeFrames[c],logFile,end="")
	print("}", file=logFile)


	print("wallFrames = {", file=logFile, end="")
	sep=False
	for w in wallFrames:
		print( ", \n" if sep else "" , file=logFile, end="")
		sep = True
		print(w + str(" : "), file=logFile, end="")
		printFrameList(wallFrames[w],logFile,end="")
	print("}", file=logFile)

	print(cubeFrames)
	print(wallFrames)


cozmo.robot.Robot.drive_off_charger_on_connect = False
cozmo.run_program(cozmo_program, use_3d_viewer=True, use_viewer=False)
