#!/usr/bin/env python3
import matplotlib.pyplot as plt

from frame2d import Frame2D
from cmap import CozmoMap, is_in_map, Coord2D
import math
import numpy as np
from cube_exp_means_variance import cube_means, cube_variance
import time

# Do we need the means from Y and A experiments?

mean_dist = [dist[0] for dist in cube_means[:][:]]  # (only x axis)
var_dist = [dist[0] for dist in cube_variance[:][:]]  # (only x axis)
print(str(mean_dist))
print(str(var_dist))
newmeans = []
newvars = []


# Forward kinematics: compute coordinate frame update as Frame2D from left/right track speed and time of movement
def track_speed_to_pose_change(pos: Frame2D, left, right, timer, wheel_distance: float):
    # Calculate the change in linear motion
    if left == right:
        angle = pos.angle()
        x = pos.x() + left * np.cos(angle) * timer
        y = pos.y() + left * np.sin(angle) * timer

        # Calculate the change in circular motion
    else:
        radius = wheel_distance / 2.0 * (left + right) / (right - left)

        # Calculate the ability to instantaneously turn both left /right wheels
        turn_x = pos.x() - radius * np.sin(pos.angle())
        turn_y = pos.y() + radius * np.cos(pos.angle())

        # Calculate angular velocity
        angular_velocity = (right - left) / wheel_distance

        # Calculate the change in angular motion
        angular_change = pos.angle() + angular_velocity * timer

        # Forwards kinematics for differential drive
        x = np.cos(angular_change) * (pos.x() - turn_x) - np.sin(angular_change) * (pos.y() - turn_y) + turn_x
        y = np.sin(angular_change) * (pos.x() - turn_x) + np.cos(angular_change) * (pos.y() - turn_y) + turn_y
        angle = angular_change + pos.angle()

    # Returns the new position in FRAME2D
    # return Frame2D(x, y, angle)
    return x, y, angle


# Basic Idea for the Kinematics
# TODO
#   - use the robot's speed and time to compute the change in position
#   - use the change in position to compute the change in frame
#   - use the change in frame to compute the new coordinate frame
#   - use the new coordinate frame to compute the new position
#   - use the new position to compute the new coordinate frame
#   - repeat

def velocity_to_track_speed(start_pose: Frame2D, end_pose: Frame2D, timer):
    # Compute distance between the Cozmo's wheels
    # And also the robot's starting position
    # global pos
    forward_motion = True
    wheel_distance = 35  # cozmo tracks are 35mm apart

    xdist = abs(start_pose.x() - end_pose.x())
    ydist = abs(start_pose.y() - end_pose.y())

    if xdist == 0 and ydist == 0:
        forward_motion = False
    else:
        forward_motion = True

    if forward_motion:  # if forward motion
        if xdist == 0:  # if moving along y axis
            left = right = (ydist / timer)  # set speed using distance/time
            if left < 5:  # check if above min speed, if not then set to min speed of 5
                left = right = 5
        else:  # else must be moving on x axis
            left = right = (xdist / timer)  # set speed
            if left < 5:  # check min speed
                left = right = 5

        x, y, angle = track_speed_to_pose_change(start_pose, left, right, timer, wheel_distance)  # calculate pos
        plt.quiver(x, y, np.cos(angle), np.sin(angle), angles='xyz', scale_units='xy', scale=2)  # plot position
        print("Forward Motion" + "x: " + str(x) + " y: " + str(y) + " angle: " + str(angle))

    else:  # if the left and right track speeds are different i.e. angular motion

        temp = (math.degrees(end_pose.angle() - start_pose.angle()))
        speed = temp / timer

        if temp < 0:
            left = -speed
            right = speed
        else:
            right = - speed
            left = speed

        x, y, angle = track_speed_to_pose_change(start_pose, left, right, timer, wheel_distance)
        plt.quiver(x, y, np.cos(angle), np.sin(angle), angles='xyz', scale_units='xy', scale=2)
        print("Angular Motion" + "x: " + str(x) + " y: " + str(y) + " angle: " + str(angle))

    return left, right, [x, y, angle]


def motion_model(current_pose: Frame2D, velocity, previous_pose: Frame2D, time):
    pose_px = previous_pose.x()
    pose_py = previous_pose.y()
    pose_cx = current_pose.x()
    pose_cy = current_pose.y()
    ydist = pose_cy - pose_py  # distance between y coords of current and previous
    xdist = pose_cx - pose_px  # distance between x coords of current and previous
    r = math.sqrt((xdist) ** 2 + (ydist) ** 2)  # Euclidean distance between current and previous using pythagoras
    cov_mat = [[]]

    print(str("previous: ") + str(previous_pose))
    print(str("current:" + str(current_pose)))

    mew = abs(0.5 * ((xdist * math.cos(previous_pose.angle()) + (ydist * math.sin(previous_pose.angle()))) /
                     (ydist * math.cos(previous_pose.angle()) - (xdist * math.sin(previous_pose.angle())))))

    estimate_x = ((pose_px + pose_cx) / 2) + (mew * (pose_py - pose_cy))
    estimate_y = ((pose_py + pose_cy) / 2) + (mew * (pose_px - pose_cx))
    estimate_dist = math.sqrt((pose_px - estimate_x) ** 2 + (pose_py - estimate_y) ** 2)

    print("Estimate x cord:" + str(estimate_x))
    print("Estimate y cord:" + str(estimate_y))
    print("Estimate dist:" + str(estimate_dist))

    delta_a = math.atan2(pose_cy - estimate_y, pose_cx - estimate_x) - math.atan2(pose_py - estimate_y,
                                                                                  pose_px - estimate_x)

    estimate_fv = (delta_a / time) * estimate_dist
    estimate_av = (delta_a / time)
    gamma_hat = ((current_pose.angle() - previous_pose.angle()) / time) - estimate_av

    # internal system error noise value
    alpha_pred = 0.1
    x_variance = 20.33
    y_variance = 3.6
    a_variance = 0.001

    cov_mat = [[x_variance, 0, 0], [0, y_variance, 0], [0, 0, a_variance]]

    noise = (alpha_pred * velocity[0]) + (alpha_pred * velocity[1])
    # prob(v − vˆ, α1|v| + α2|ω|) (NB: , means p(X and y)

    fv_error = velocity[0] - estimate_fv
    av_error = velocity[1] - estimate_av
    gamma_error = gamma_hat

    # q1 = calc_prob(0, x_variance)
    # q2 = calc_prob(0, y_variance)
    # q3 = calc_prob(0, a_variance)
    # test with different noise equations to see if they make a difference
    q1 = calc_prob(x_variance, noise)
    q2 = calc_prob(y_variance, noise)
    q3 = calc_prob(a_variance, noise)

    # What if we represented the variance (i.e. error) that we take from experiments as a zero centred mean probability.
    # then we implement this value into the fv/av_error values when calculating the final probability??
    print(str(current_pose.angle()))
    print(str((previous_pose.angle() + estimate_av * time + gamma_hat * time)))

    p1 = calc_prob(estimate_fv, fv_error * (q1 + q2))  # I think this has now added noise to it...?
    p2 = calc_prob(estimate_av, av_error * q3)
    p3 = calc_prob(gamma_hat, 0)  # i.e. a zero variances prob

    p = p1 * p2 * p3
    print("Probability: " + str(p))
    return p, cov_mat


def calc_prob(mew, var):
    p = 1

    if var == 0:
        p = math.exp(-0)
    else:
        p = math.exp(-((mew ** 2) / (2 * var)))

    return p


def cliff_sensor_model(robotPose: Frame2D, m: CozmoMap, cliffDetected):
    sensorPose = robotPose.mult(Frame2D.fromXYA(20, 0, 0))
    if not is_in_map(m, robotPose.x(), robotPose.y()):
        return 0
    if not is_in_map(m, sensorPose.x(), sensorPose.y()):
        return 0
    c = Coord2D(sensorPose.x(), sensorPose.y())
    if m.grid.isOccupied(c) == cliffDetected:  # TODO this will not always be exact
        return 0.99
    else:
        return 0.0


# Take a true cube position (relative to robot frame). Compute /probability/ of cube being (i) visible AND being
# detected at a specific measure position (relative to robot frame)
def cube_sensor_model(trueCubePosition: Frame2D, visible, measuredPosition: Frame2D):
    """
    #, robotPose: Frame2D, mean_list, var_list):
    # Loop through the many frames
    # for loop through the cube frames list

    mabs_dist = {}
    abs_angle = {}
    # print(str(trueCubePosition.x()) + " = " + str(robotPose.x()))
    tabs_dist = math.sqrt((trueCubePosition.x() - robotPose.x()) ** 2 + (trueCubePosition.y() - robotPose.y()) ** 2)
    tabs_angle = math.atan2(trueCubePosition.y() - robotPose.y(), trueCubePosition.x() - robotPose.x())

    # for cube_frame in measuredPosition:
    # mabs_dist = math.sqrt((cube_frame.x() - robotPose.x()) ** 2 + (cube_frame.y() - robotPose.y()) ** 2)
    # mabs_angle = math.atan2(trueCubePosition.y() - robotPose.y(), trueCubePosition.x() - robotPose.x())
    # mean_dist = [adist for adist in mabs_dist]
    # print("temp")
    # variancedist = math.sqrt(sum([abs((adist - mean_dist)) ** 2 for adist in mabs_dist]) / len(mabs_dist))

    # except:
    mabs_dist = math.sqrt((measuredPosition.x() - robotPose.x()) ** 2 + (measuredPosition.y() - robotPose.y()) ** 2)
    # mabs_dist = math.sqrt((measuredPosition.x() - robotPose.x()) ** 2 + (measuredPosition.y() - robotPose.y()) ** 2)
    mabs_angle = math.atan2(measuredPosition.y() - robotPose.y(), measuredPosition.x() - robotPose.x())
    #    mean_dist = mabs_dist
    new_mean = [((mean_list[1] * mean_list[0]) + measuredPosition.x()) / (mean_list[0] + 1),
                ((mean_list[2] * mean_list[0]) + measuredPosition.y()) / (mean_list[0] + 1),
                ((mean_list[3] * mean_list[0]) + measuredPosition.angle()) / (mean_list[0] + 1)]
    new_variance = (var_list[0] / (var_list[0] + 1)) * (
            var_list[1] + (((new_mean[0] - measuredPosition.x()) ** 2) / (var_list[0] + 1)))

    # print(str(mabs_dist))
    p = exp(-0.5 * (abs(tabs_dist - mabs_dist) / tabs_dist) * (abs(tabs_angle - mabs_angle) / tabs_angle))
    # p, updated_mean, updated_var = calc_Probability(measuredPosition, trueCubePosition)
    # p = calc_Probability(measuredPosition, trueCubePosition)
    """
    return 1  # , new_mean, new_variance


# Take a true wall position (relative to robot frame). Compute /probability/ of wall being (i) visible AND being
# detected at a specific measure position (relative to robot frame)
def wall_sensor_model(trueWallPosition: Frame2D, visible, measuredPosition: Frame2D):
    return


''' updated sensor model adds walls to list of sensed objects. We assume here that walls will
    be passed in much like cubes, namely as a dict of visible ones and a dict of (left,centre,right) frame
    triplets. We expect the visibility list AND the frame list to be in the Cozmo ID system rather than the
    simulation ID  system (i.e. walls not yet even detected aren't here). There will have to be a
    transformation at the user side from the returned wall poses in visible_walls() to the relative frames
    passed in here (or we could set this up so that visible_walls returns Frames rather than poses)'''


def cozmo_sensor_model(robotPose: Frame2D, m: CozmoMap, cliffDetected, cubeVisibility, cubeRelativeFrames,
                       wallVisibility, wallRelativeFrames):
    p = 1.
    # for cubeID in cubeVisibility:
    # cube_sensor_model(, cubeVisibility, cubeRelativeFrames) # FIXME use result correctly
    # for wallCID in wallVisibility:
    # wall_sensor_model(.....) # FIXME use result correctly
    p = p * cliff_sensor_model(robotPose, m, cliffDetected)
    return p
