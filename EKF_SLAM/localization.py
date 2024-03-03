import numpy as np
import math

from frame2d import Frame2D

def EKF_localisation(est_prev_pos, current_pose, covariance, v_vect, seen_landmarks, map, change_time, prev_result, start):

    pose_px = est_prev_pos.x()
    pose_py = est_prev_pos.y()
    pose_cx = current_pose.x()
    pose_cy = current_pose.y()
    ydist = pose_cy - pose_py  # distance between y coords of current and previous
    xdist = pose_cx - pose_px  # distance between x coords of current and previous

    r = math.sqrt((xdist) ** 2 + (ydist) ** 2)  # Euclidean distance between curlandmarksrent and previous using pythagoras

    systemVariance = [[20.33, 0, 0], [0, 3.6, 0], [0, 0, 0.001]]

    mew_list = prev_result

    est_mew = [((-v_vect[0]/v_vect[1]*math.sin(est_prev_pos.angle())) + (v_vect[0]/v_vect[1]*math.sin(est_prev_pos.angle()+v_vect[1]*change_time))),
                          ((v_vect[0] / v_vect[1] * math.cos(est_prev_pos.angle())) - (v_vect[0] / v_vect[1] * math.cos(
                              est_prev_pos.angle() + v_vect[1] * change_time))), v_vect[1]*change_time]

    taylor_exp = np.array([[1, 0, (((v_vect[0] / v_vect[1]) * math.cos(est_prev_pos.angle())) - (v_vect[0] / v_vect[1]) * math.cos(
        est_prev_pos.angle() + v_vect[1] * change_time))],
                           [0, 1, (((v_vect[0]/v_vect[1])*math.sin(est_prev_pos.angle())) - (v_vect[0]/v_vect[1])*math.sin(est_prev_pos.angle()+v_vect[1]*change_time))],
                           [0, 0, 1]])
    
    
    updated_covariance = np.array(taylor_exp @ covariance @ np.transpose(taylor_exp) + systemVariance) #WHAT IS THIS? (Rt))

    sensorReading = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])


    est_land_dist = []
    matrixK = []
    sensor_jacobian = []

    for landmark in map:

        delta = np.array([[map[landmark].pose.x() - est_mew[0]], [map[landmark].pose.y() - est_mew[1]]])

        trans_dist = np.transpose(delta) @ delta # [0, 0]?

        z_angle = math.atan2(delta[0], delta[1]) - est_mew[-1]
        angle = map[landmark].pose.angle()
        est_land_dist.append(np.array([math.sqrt(trans_dist), z_angle, angle]))

        H = np.array(1/trans_dist * [[math.sqrt(trans_dist) * delta[0], -math.sqrt(trans_dist) * delta[1], 0,],
                                          [delta[1], delta[0], -1],
                                         [0, 0, 0]])

        matrixK.append((updated_covariance @ np.transpose(H)) @ np.linalg.inv(H.astype(float) @ updated_covariance.astype(float) @ np.transpose(H).astype(float) + sensorReading.astype(float)))
        sensor_jacobian.append(H)


    sum_matk = [0, 0, 0]
    for seen in seen_landmarks:
        i = seen.object_id-1
        object = next((j for j in map if j == seen.object_id), None)
        z = np.array([map[object].pose.x(), map[object].pose.y(), map[object].pose.angle()])
        sum_matk = sum_matk + (matrixK[i] @ (z - est_land_dist[i]))

    temp2 = [sum_matk[0], sum_matk[1], sum_matk[2]] # due to array issue
    mew2 = est_mew + temp2

    sum_matk2 = [0, 0, 0]
    for seen in seen_landmarks:
        i = seen.object_id - 1
        sum_matk2 = sum_matk2 + (matrixK[i] * sensor_jacobian[i])

    temp2 = [sum_matk2[0], sum_matk2[1], sum_matk2[2]]
    new_sigma = (np.identity(len(covariance)) - temp2) @ updated_covariance

    mew_list.append(mew2)

    return mew_list, new_sigma