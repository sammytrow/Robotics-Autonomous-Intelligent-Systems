import numpy as np
import math
from frame2d import Frame2D

def SLAM(prev_mew, covariance, v_vect, seen_landmarks, landmarks, change_time):
    F_mask = np.zeros((3, 60))  # 60 as it is 20 x 3

    F_mask[0][0] = 1
    F_mask[1][1] = 1
    F_mask[2][2] = 1

    systemVariance = np.array([[20.33, 0, 0], [0, 3.6, 0], [0, 0, 0.001]])

    temp = np.array([[(-v_vect[0] / v_vect[1] * math.sin(prev_mew[-1][2]) + (
                v_vect[0] / v_vect[1] * math.sin(prev_mew[-1][2] + v_vect[1] * change_time))),
                      ((v_vect[0] / v_vect[1] * math.cos(prev_mew[-1][2])) - (v_vect[0] / v_vect[1] * math.cos(
                          prev_mew[-1][2] + v_vect[1] * change_time))), v_vect[1] * change_time]])

    est_mew = (np.transpose(F_mask) @ temp.T)  # should all be adding the previous mew value (as a  60x1 vector)

    FT = np.transpose(F_mask)
    I = np.identity(len(FT))
    mat_X = np.array(
        [[0, 0, (((v_vect[0] / v_vect[1]) * math.cos(prev_mew[-1][2])) - (v_vect[0] / v_vect[1]) * math.cos(
            prev_mew[-1][2] + v_vect[1] * change_time))],
         [0, 0, (((v_vect[0] / v_vect[1]) * math.sin(prev_mew[-1][2])) - (v_vect[0] / v_vect[1]) * math.sin(
             prev_mew[-1][2] + v_vect[1] * change_time))],
         [0, 0, 0]])

    taylor_exp = I + (FT @ mat_X @ F_mask)


    updated_covariance = taylor_exp @ covariance @ np.transpose(taylor_exp) + np.transpose(F_mask) @ systemVariance @ F_mask

    sensorReading = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    sensor_jacobian = []
    F_mask_k = np.zeros((60, 60))  # 3 times the number of landmarks
    F_mask_k[0][0] = 1
    F_mask_k[1][1] = 1
    F_mask_k[2][2] = 1
    matrixK = []
    est_seen = []

    for seen in seen_landmarks:
        j = landmarks[seen.object_id-1]
        kp = j * 3 + 1

        Pose = Frame2D.fromPose(seen.pose)
        if est_mew[kp][0] == 0:
            est_mew[kp] = Pose.x()
            est_mew[kp + 1] = Pose.y()
            est_mew[kp + 2] = Pose.angle()

        delta = [est_mew[kp] - est_mew[0], est_mew[kp+1] - est_mew[1]]

        q_k = np.transpose(delta) @ delta

        est_seen.append([math.sqrt(q_k), math.atan2(delta[1], delta[0]) - est_mew[2], est_mew[2]])
        F_mask_k[kp][kp] = 1
        F_mask_k[kp + 1][kp + 1] = 1
        F_mask_k[kp + 2][kp + 2] = 1


        grape = [[math.sqrt(q_k) * delta[0], -math.sqrt(q_k) * delta[1], 0, -math.sqrt(q_k) * delta[0],
                  math.sqrt(q_k) * delta[1], 0],
                 [delta[1], delta[0], -1, -delta[1], -delta[0], 0], [0, 0, 0, 0, 0, 1]]

        f_temp = np.concatenate((F_mask_k[0:3], F_mask_k[kp:kp+3]), axis=0)

        H = 1/q_k * (grape @ f_temp)

        matrixK.append(updated_covariance @ np.transpose(H) @ np.linalg.inv(H.astype(float)
                @ updated_covariance.astype(float) @ np.transpose(H).astype(float) + sensorReading))
        sensor_jacobian.append(H)


    i = 0
    sum_matk = [0, 0, 0]
    for seen in seen_landmarks:
        Pose = Frame2D.fromPose(seen.pose)
        z = np.array([Pose.x(), Pose.y(), Pose.angle()])
        sum_matk = matrixK[i] * (z - est_seen[i])
        i += 1
    mew2 = est_mew + sum_matk

    i = 0
    sum_matk2 = np.zeros((60,60))
    for seen in seen_landmarks:
        sum_matk2 = sum_matk2 + matrixK[i] @ sensor_jacobian[i]
        i += 1

    new_sigma = (I - sum_matk2) @ updated_covariance

    return mew2, new_sigma
