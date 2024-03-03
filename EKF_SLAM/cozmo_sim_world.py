import cozmo

from frame2d import Frame2D

from cmap import CozmoMap, Coord2D, is_in_map
from mcl_tools import *

# Making the sensor models and importing it in here
from cozmo_interface import cliff_sensor_model, cube_sensor_model

from threading import Lock

import math
import numpy as np
import random
import time


class CozmoSimWorld:
    def __init__(self, m, pos):
        self.__dont_touch_maxDockDistance = 1.0  # guess: how close Cozmo has to be to a cube to pick it up
        self.__dont_touch_maxDockOffCentre = 0.2  # guess: how far off dead centre Cozmo can be to a cube to pick it up
        self.__dont_touch_maxDockAngle = math.pi / 32  # guess: maximum off angle Cozmo can be to lift a cube
        self.__dont_touch_safeDockDistance = 0.8  # guess: distance Cozmo has to be to a cube that it is not a factor in being able to pick it up
        self.__dont_touch_safeDockOffCentre = 0.15  # guess: how far off dead centre Cozmo needs to be for it to be a non-factor in picking it up
        self.__dont_touch_safeDockAngle = math.pi / 48  # guess: maximum off angle Cozmo needs to be for angle not to be important in picking it up
        self.map = m
        self.__dont_touch__pos = pos
        self.__pos_mutex = Lock()
        self.__dont_touch__origin = pos  # keep track of the original pose for wall-relative poses.
        self.__dont_touch__s1 = 0
        self.__dont_touch__s2 = 0
        self.__dont_touch__s1Command = 0
        self.__dont_touch__s2Command = 0
        self.__dont_touch__wall_map = {}  # maps cozmo wall IDs to simulation wall IDs
        self.__dont_touch__cliff_sensed = False
        self.__dont_touch__cube_visibility = {
            cozmo.objects.LightCube1Id: False,
            cozmo.objects.LightCube2Id: False,
            cozmo.objects.LightCube3Id: False}
        # initialise actual walls
        # may not be legial to do a conditional comprehenson like this
        self.__dont_touch__wall_visibility = dict.fromkeys([k for k in self.map.landmarks.keys()
                                                            if k not in [cozmo.objects.LightCube1Id,
                                                                         cozmo.objects.LightCube2Id,
                                                                         cozmo.objects.LightCube3Id]],
                                                           False)
        self.__dont_touch__have_cube = None
        # an array of coordinates for the footprint of Cozmo - useful for collision detection
        # this could be a single-dimensional array because the coordinates are built into each index
        # self.__dont_touch__cozmo_footprint = [[Frame2D.fromXYA(x,y,0) for y in range(-20,21)] for x in range (-35,36)]
        self.__dont_touch__cozmo_footprint = [[Frame2D.fromXYA(x, y, 0) for y in range(-20, 21, 5)] for x in
                                              range(-35, 36, 5)]
        self.waitTime = 0.05
        self.gotStuck = False
        self.collision = 0

        self.cube_measured_positions = {
            cozmo.objects.LightCube1Id: [],
            cozmo.objects.LightCube2Id: [],
            cozmo.objects.LightCube3Id: []
        }

    def _touch_for_experiments_only__pos(self):
        return self.__dont_touch__pos

    def _touch_for_experiments_only__set_pos(self, position):
        self.__pos_mutex.acquire()
        self.__dont_touch__pos = position
        self.__pos_mutex.release()

    def _touch_me_cube_sensor_model(self, cubeID):
        measuredP = self.map.landmarks[cubeID].poseRelative(self.__dont_touch__pos)
        # if len(self.cube_measured_positions[cubeID]) == 49:
        #    self.cube_measured_positions[cubeID].pop(0)
        # self.cube_measured_positions[cubeID].append(measuredP)

        p = cube_sensor_model(self.map.landmarks[cubeID].pose, True,
                              self.map.landmarks[cubeID].poseRelative(self.__dont_touch__pos))

        if random.random() < p:
            return True
        return False

    def _touch_me_wall_sensor_model(self, wallID):
        measuredP = self.map.landmarks[wallID].poseRelative(self.__dont_touch__pos)

        """if len(self.cube_measured_positions[wallID]) == 9:
            self.cube_measured_positions[wallID].pop(0)
        self.cube_measured_positions[wallID].append(measuredP)"""
        print("wall position: " + measuredP)
        # p = cube_sensor_model(self.map.landmarks[wallID].pose, True, self.cube_measured_positions[wallID])

        if random.random() < p:
            return True
        return False

    def __dont_touch__sample_cliff_sensor_model(self):
        p = cliff_sensor_model(self.__dont_touch__pos, self.map, True)
        if random.random() < p:
            return True
        return False

    def sample_motion_model(self, leftTrackDistance, rightTrackDistance):  # TODO
        return Frame2D()  # TODO

    def sense_cube_pose_relative(self, cubeID):
        print(self.map.landmarks[cubeID].pose)
        if cubeID == self.__dont_touch__have_cube:
            return Frame2D.fromXYA(0, 0, 0)

        if not self.cube_is_visible(cubeID):
            return Frame2D()

        cubeRelativePose = self.map.landmarks[cubeID].poseRelative(self.__dont_touch__pos)
        cubeDistance = math.sqrt(cubeRelativePose.y() ** 2 + cubeRelativePose.x() ** 2)
        cubeAngle = math.atan2(cubeRelativePose.y(), cubeRelativePose.x())
        cubeRotation = cubeRelativePose.angle()

        # use  self.map.landmarks[cubeID] (of type 'Object2D') to obtain relative position # TODO
        noiseGenerators: list = [np.random, np.random, np.random]

        sigmaS = 10.0
        aParamAngle = 138  # angular variance of ~0.015*pi
        aParamRotation = 34  # rotation variance of ~0.03*pi
        # distance estimates are Gaussian distributed
        noisyDistance = sigmaS = 10.0
        aParamAngle = 138  # angular variance of ~0.015*pi
        aParamRotation = 34  # rotation variance of ~0.03*pi
        # distance estimates are Gaussian distributed
        noisyDistance = noiseGenerators[0].normal(cubeDistance, sigmaS)
        # but angle estimates are beta-distributed because they're limited to the range [-pi/2,pi/2]
        noisyAngle = cubeAngle + math.pi * noiseGenerators[1].beta(aParamAngle, aParamAngle) - math.pi / 2
        noisyRotation = cubeRotation + math.pi * noiseGenerators[2].beta(aParamRotation,
                                                                         aParamRotation) - math.pi / 2
        print(" !!!!!!!!!!!!!!!!!!!!NOISY!!!!!!!!!!!!!!!! " + str(noiseGenerators))
        cubeRelativePose.x = (noisyDistance * math.cos(noisyAngle))
        cubeRelativePose.y = (noisyDistance * math.sin(noisyAngle))
        cubeRelativePose.angle = noisyRotation
        return cubeRelativePose  # TODO model sensor noise

    def dont_touch__step(self):
        if not self.gotStuck:
            x = self.__dont_touch__pos.x()
            y = self.__dont_touch__pos.y()
            if not is_in_map(self.map, x, y):
                self.gotStuck = True
            elif self.map.grid.isHole(Coord2D(x, y)):
                self.gotStuck = True

        f = self.sample_motion_model(self.__dont_touch__s1 * self.waitTime, self.__dont_touch__s2 * self.waitTime)

        if not self.gotStuck:
            # move, but only if we aren't moving into an obstruction
            self.__pos_mutex.acquire()
            newPos = self.__dont_touch__pos.mult(f)
            # if not self.map.grid.isOccupied(Coord2D(newPos.x(),newPos.y())):
            self.collision = self.__dont_touch__detect_collision(newPos)
            # if not self.collision:
            self.__dont_touch__pos = newPos
            self.__pos_mutex.release()

        self.__dont_touch__cliff_sensed = self.__dont_touch__sample_cliff_sensor_model()

        self.__dont_touch__compute_object_visibility  # update all the walls in view

        ''' original cube visibility lines below

                for cubeID in self.__dont_touch__cube_visibility:
                        self.__dont_touch__cube_visibility[cubeID] = self.__dont_touch__compute_cube_is_visible(cubeID)
                '''

        self.__dont_touch__s1 = 0.9 * self.__dont_touch__s1 + 0.1 * self.__dont_touch__s1Command
        self.__dont_touch__s2 = 0.9 * self.__dont_touch__s2 + 0.1 * self.__dont_touch__s2Command

    def __dont_touch__detect_collision(self, pos):
        for x in self.__dont_touch__cozmo_footprint:
            for xy in x:
                cozmoPos = pos.mult(xy)
                if self.map.grid.isOccupiedGrid(self.map.grid.frame2D2Coord2DGrid(cozmoPos)):
                    return time.time()
        return 0

    # need a general object-visibility routine to identify objects more or less in the line of sight
    # now that we have walls to worry about
    def __dont_touch__compute_object_visibility(self):
        # first, test whether each object can be seen at all and gather a list
        landmarkCoords = []
        for landmark in self.map.landmarks.items():
            relativePoseCentre = landmark[1].poseRelative(self.__dont_touch__pos)
            relativePoseRightEdge = landmark[1].rightLOS(self.__dont_touch__pos)
            relativePoseLeftEdge = landmark[1].leftLOS(self.__dont_touch__pos)
            landmarkCoords.append([[relativePoseLeftEdge.x(), relativePoseLeftEdge.y(), relativePoseLeftEdge.angle()],
                                   [relativePoseCentre.x(), relativePoseCentre.y(),
                                    relativePoseCentre.angle()],
                                   [relativePoseRightEdge.x(), relativePoseRightEdge.y(),
                                    relativePoseRightEdge.angle()]])
        landmarkCArray = np.array(landmarkCoords)
        distancePArray = np.transpose(landmarkCArray, (0, 2, 1))
        distancePArray[:, 2] = [0, 0, 0]

        # calculate distance by Pythagoras. (sqrt(x^2+y^2). This will produce an
        # array of distances, where each row is a landmark and the columns are the left,
        # centre, and right distances respectively
        distance = np.sqrt(np.matmul(landmarkCArray, distancePArray))

        # and angle to objects by trigonometry (arctan). This will produce an array
        # of angles, in the same format as the distance array.
        angle = np.arctan2(landmarkCArray[:, :, 1], landmarkCArray[:, :, 0])
        midAngle = 30.0 / 180.0 * math.pi

        # and this gives an array of angles relative to the visible range where rows are landmarks and columns angles
        relativeAngle = np.fabs(angle) / midAngle
        relativeTolerance = 0.1

        # now we can quickly eliminate elements that are completely outside the FOV
        invisible = []  # this will allow us to delete the elements as a bulk row-erase
        visibleIDs = []  # and this lets us know which landmark IDs are still potentially visible
        invisibleIndex = 0
        for landmark in self.map.landmarks.items():
            if not landmark[1].objectVisible(self.__dont_touch__pos):
                invisible.append(invisibleIndex)
            else:
                visibleIDs.append(landmarkID)
            invisibleIndex = invisibleIndex + 1

        # have the list of definitely invisibles; get rid of them in all relevant arrays
        np.delete(distance, invisible, 0)
        np.delete(angle, invisible, 0)
        np.delete(relativeAngle, invisible, 0)

        np.delete(landmarkCArray, invisible, 0)
        # let the fun begin: we now have to test objects against other objects to see which ones are in front of which.
        # There are probably some very efficient game-engine algorithms to do this, but we assume the number of objects
        # left at this point is small and won't kill us computationally.
        occluded = []  # create an occluded indices list

        # go through each candidate visible object, looking at its centre
        for idx in range(angle.shape[0]):
            # overlapping objects must at least obscure half of the centre-point marker
            overlapping = [index for index in range(angle.shape[0]) if index != idx and
                           angle[index, 0] > angle[idx, 1] and
                           angle[index, 2] < angle[idx, 1]]
            # check the distance of the potentially overlapping object at the angle of the test object's marker
            for otherIdx in overlapping:
                if self.map.landmarks[visibleIDs[otherIdx]].distanceAtAngle(self.__dont_touch__pos, angle[idx, 1]) < \
                        distance[idx, 1, 1]:
                    occluded.append[idx]
                    break

        # now we have a comprehensive list of occluded objects. Easiest solution to prune the visible list
        # is to vectorise it in numpy and then bulk delete.
        visibleIDArray = np.array(visibleIDs)
        np.delete(visibleIDArray, occluded)
        np.delete(distance, occluded, 0)
        np.delete(angle, occluded, 0)
        np.delete(relativeAngle, occluded, 0)

        # reset everything to invisible
        for key in self.__dont_touch__cube_visibility.keys():
            self.__dont_touch__cube_visibility[key] = False
        for key in self.__dont_touch__wall_visibility.keys():
            self.__dont_touch__wall_visibility[key] = False

        ''' and then handle the (probably) visible items. This uses the same original code of
                     __dont_touch__compute_cube_is_visible. We assume the 'active area' of the wall is
                     similar to a light cube, which is probably close enough to true given that it will
                     use a marker just like light cubes do.
                '''

        for visibleIdx in range(visibleIDArray.shape[0]):
            if 1 + relativeTolerance < relativeAngle[visibleIdx, 1]:
                angleVisibilityProb = 0.0
            elif 1 - relativeTolerance < relativeAngle[visibleIdx, 1]:
                angleVisibilityProb = 0.5
            else:
                angleVisibilityProb = 0.99

            minDistance = 100
            minTolerance = 50
            maxDistance = 600
            maxTolerance = 100
            if distance[visibleIdx, 1, 1] < minDistance - minTolerance:
                distanceProb = 0.0
            elif distance[visibleIdx, 1, 1] < minDistance + minTolerance:
                distanceProb = 0.99 * (distance[visibleIdx, 1] - (minDistance - minTolerance)) / (2 * minTolerance)
            elif distance[visibleIdx, 1, 1] < maxDistance - maxTolerance:
                distanceProb = 0.99
            elif distance[visibleIdx, 1, 1] < maxDistance + maxTolerance:
                distanceProb = 0.99 - 0.99 * (distance[visibleIdx, 1] - (maxDistance - maxTolerance)) / (
                        2 * maxTolerance)
            else:
                distanceProb = 0.0

            p = angleVisibilityProb * distanceProb

            if random.random() < p:
                if visibleIDArray[visibleIdx] in self.__dont_touch__cube_visibility:
                    self.__dont_touch__cube_visibility[visibleIDArray[visibleIdx]] = True
                else:
                    self.__dont_touch__wall_visibility[visibleIDArray[visibleIdx]] = True

    def cube_is_visible(self, cubeID):
        self.__dont_touch__cube_visibility[cubeID] = self._touch_me_cube_sensor_model(cubeID)
        ### <!- sam: Add probability function?
        return self.__dont_touch__cube_visibility[cubeID]

    def cube_in_range(self, cubeID):
        relativePose = self.map.landmarks[cubeID].poseRelative(self.__dont_touch__pos)
        relativeDistance = math.sqrt(relativePose.x() ** 2 + relativePose.y() ** 2)
        return relativeDistance <= self.__dont_touch__maxDockDistance

    def cube_on_axis(self, cubeID):
        relativePose = self.map.landmarks[cubeID].poseRelative(self.__dont_touch__pos)
        return (abs(relativePose.angle()) <= self.__dont_touch__maxDockAngle) and (
                abs(relativePose.y()) <= self.__dont_touch__maxDockOffCentre)

    def lift_cube(self, cubeID):
        if self.__dont_touch__have_cube is not None:
            return False  # already have a cube. Don't try to lift another.
        if not (self.cube_in_range(cubeID) and self.cube_on_axis(cubeID)):
            self.__dont_touch__have_cube = False  # not in a pose to lift the cube
            return False
        xyaFailureGenerators = [np.random.Generator(), np.random.Generator(), np.random.Generator()]
        relativePose = self.map.landmarks[cubeID].poseRelative(self.__dont_touch__pos)
        # 3 ways to fail, which we assume are conditionally independent given the docking criteria
        liftSuccessful = xyaFailureGenerators[0].binomial(1, 0.99 - (
                0.49 * ((relativePose.x() - self.__dont_touch__safeDockDistance) /
                        (self.__dont_touch__maxDockDistance - self.__dont_touch__safeDockDistance))))
        liftSuccessful = liftSuccessful * xyaFailureGenerators[1].binomial(1, 0.99 - (
                0.49 * ((relativePose.y() - self.__dont_touch__safeDockOffCentre) /
                        (self.__dont_touch__maxDockOffCentre - self.__dont_touch__safeDockOffCentre))))
        liftSuccessful = liftSuccessful * xyaFailureGenerators[1].binomial(1, 0.99 - (
                0.49 * ((relativePose.angle() - self.__dont_touch__safeDockAngle) /
                        (self.__dont_touch__maxDockAngle - self.__dont_touch__safeDockAngle))))
        # successfully lifted, so the cube is now in the same pose as the robot.
        if liftSuccessful == 1:
            self.__dont_touch__have_cube = cubeID
            self.__dont_touch__cube_visibility[cubeID] = False
            self.map.landmarks[cubeID].pose = self.__dont_touch__pos
        # otherwise the cube stays where it is. In reality, it would likely be jostled, but we're not going to try to simulate that!

    def deposit_cube(self):
        if self.__dont_touch__have_cube is None:
            return
        xyaGenerators = [np.random.Generator(), np.random.Generator(), np.random.Generator()]
        '''frightening maths: what we are doing here is computing a gamma-distributed estimate of the
                   drop-off distance (one that peaks a short distance away from the absolute minimum: the front
                   of the robot) and then decays quasi-exponentially with a long, but rapidly diminishing, tail
                   Computation of the scale and shape parameters, which set up the distribution, is complicated
                   because it involves a quadratic, but straightforward. We set the peak to occur at the 'safe'
                   dock distance and the variance to be the difference between safe and max dock distance. So the
                   cube could (improbably) slide forward some long distance, but much more probably, it will be
                   deposited immediately in front of the Cozmo.
                '''
        actualDistanceDelta = self.__dont_touch__safeDockDistance - self.__dont_touch__cozmoFront
        maxDistanceDelta = self.__dont_touch_maxDockDistance - self.__dont_touch__safeDockDistance
        xScale = (actualDistanceDelta + math.sqrt(actualDistanceDelta ** 2 + 4 * maxDistanceDelta ** 2)) / (
                2 * maxDistanceDelta ** 2)
        xShape = xScale * (self.__dont_touch__safeDockDistance - self.__dont_touch__cozmoFront) + 1
        cubeNewX = xyaGenerators[0].gamma(xShape, xScale)
        # lateral position is simpler: just a Gaussian with deviation equivalent to the 'safe' offset. We don't
        # use the 'maximum' because we assume this would be an unlikely shift
        cubeNewY = xyaGenerators[1].normal(0, 2 * self.__dont_touch__safeDockOffCentre)
        # angles are limited to the range [-pi/2,pi/2] so we use a beta distribution with a variance equivalent to
        # the square of the safe docking angle. This computation follows from the formula for the variance of a
        # beta distribution: var(B) = ab/((a+b)^2*(a+b+1)) and using a=b to get a symmetric distribution
        aParam = (math.pi ** 2) / (32 * self.__dont_touch__safeDockAngle ** 2) - 1 / 2
        cubeNewA = xyaGenerators[2].beta(aParam, aParam) * math.pi - math.pi / 2
        # cube is deposited in the determined pose
        depositedCube = self.__dont_touch__have_cube
        self.map.landmarks[depositedCube].pose = self.__dont_touch__pos.mult(
            Frame2D.fromXYA(cubeNewX, cubeNewY, cubeNewA))
        self.__dont_touch__have_cube = None
        self.__dont_touch__cube_visibility[depositedCube] = self.__dont_touch__compute_cube_visibility(depositedCube)

    @property
    def have_cube(self):
        return self.__dont_touch__have_cube

    def is_cliff_detected(self):
        return self.__dont_touch__cliff_sensed

    def wall_is_visible(self, wallID):
        return self.__dont_touch__wall_visibility.get(self.__dont_touch__wall_map.get(wallID, -1), False)

    def collision_avoidance(self):
        if self.collision != 0:
            if time.time() < self.collision + 3:
                self.__dont_touch__s1Command = -20
                self.__dont_touch__s2Command = 20
            elif time.time() < self.collision + 5:
                self.__dont_touch__s1Command = 20
                self.__dont_touch__s2Command = 20
            else:
                self.__dont_touch__s1Command = 0
                self.__dont_touch__s2Command = 0
                self.collision = 0

    def drive_wheel_motors(self, l_wheel_speed, r_wheel_speed):
        if self.collision == 0:
            self.__dont_touch__s1Command = max(-100, min(l_wheel_speed, 100))
            self.__dont_touch__s2Command = max(-100, min(r_wheel_speed, 100))

    @property
    def driving(self):
        return (self.__dont_touch__s1 > 0 or self.__dont_touch__s2 > 0)

    @property
    def wall_types(self):
        return self.map.wallTypeMap.types

    ''' visible_walls returns a dict with a list of Cozmo-relative poses for each visible wall of each type. 
            SLAM algorithms should interrogate visible_walls to get the latest observations. This may or may not 
            emulate the physical Cozmo's object detection on a customObject type.
        '''

    def visible_walls(self):
        visible_poses = {}
        v_walls = dict([w for w in self.__dont_touch_wall_visibility.items() if w[1]])
        noiseGenerators: list = [np.random.Generator(), np.random.Generator(), np.random.Generator()]
        for visible in v_walls.keys():
            wallRelativePose = self.map.landmarks[visible].poseRelative(self.__dont_touch__pos)
            wallDistance = math.sqrt(wallRelativePose.y() ** 2 + wallRelativePose.x() ** 2)
            wallAngle = math.atan2(wallRelativePose.y(), wallRelativePose.x())
            wallRotation = wallRelativePose.angle()
            # picked-out-of-the-air guesses for radial, angular, and rotational variances
            # for the moment this ignores the distance dependence. Later this should be
            # accounted for, especially wrt the rotation which is likely to be much noisier
            # with increased distance
            sigmaS = 10.0
            aParamAngle = 138  # angular variance of ~0.015*pi
            aParamRotation = 34  # rotation variance of ~0.03*pi
            # distance estimates are Gaussian distributed
            noisyDistance = noiseGenerators[0].normal(wallDistance, sigmaS)
            # but angle estimates are beta-distributed because they're limited to the range [-pi/2,pi/2]
            noisyAngle = wallAngle + math.pi * noiseGenerators[1].beta(aParamAngle, aParamAngle) - math.pi / 2
            noisyRotation = wallRotation + math.pi * noiseGenerators[2].beta(aParamRotation,
                                                                             aParamRotation) - math.pi / 2
            wallType = self.map.wallTypeMap.byID[visible]  # look up the type in ID-to-wall-type map
            if wallType in visible_poses:
                # not clear from Cozmo documentation whether we should be returning poses relative
                # to the start reference frame of Cozmo or the current reference frame. The below
                # assumes the latter. Since these are noisy readings, we assume they are NOT accurate.
                visible_poses[wallType].append(cozmo.util.Pose(x=noisyDistance * math.cos(noisyAngle),
                                                               y=noisyDistance * math.sin(noisyAngle),
                                                               z=0, angle_z=noisyRotation,
                                                               is_accurate=False))
            else:
                visible_poses[wallType] = [cozmo.util.Pose(x=noisyDistance * math.cos(noisyAngle),
                                                           y=noisyDistance * math.sin(noisyAngle),
                                                           z=0, angle_z=noisyRotation,
                                                           is_accurate=False)]
        return visible_poses

    def sense_cube_pose_global(self, cubeID):
        # print(self.map.landmarks[cubeID].pose)
        if cubeID == self.__dont_touch__have_cube:
            return self.__dont_touch__pos
        if not self.cube_is_visible(cubeID):
            return Frame2D()
        return self.__dont_touch__pos.mult(self.sense_cube_pose_relative(cubeID))

    def left_wheel_speed(self):
        if random.random() < 0.05:
            return 0
        return float(int(self.__dont_touch__s1))

    def right_wheel_speed(self):
        if random.random() < 0.05:
            return 0
        return float(int(self.__dont_touch__s2))


def runWorld(w: CozmoSimWorld, finished):
    while not finished.is_set():
        t0 = time.time()
        pose = w._touch_for_experiments_only__pos()
        w.dont_touch__step()
        # reflexively turn and then stop if Cozmo collided with an obstacle
        if w.driving and w.collision != 0:
            print("Collision!\n")
            w.collision_avoidance()
        t1 = time.time()
        timeTaken = t1 - t0
        if timeTaken < w.waitTime:
            time.sleep(w.waitTime - timeTaken)
