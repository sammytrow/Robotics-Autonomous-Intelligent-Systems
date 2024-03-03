import cozmo
import numpy as np
import math
from frame2d import Frame2D
# when we run this in run file, init prev_mew as vector of 3 zeros
# may need to include a culculation for orientation at 3rd row (currently seems to be only X and Y.
def SLAM(prev_mew, covariance, v_vect, seen_landmarks, landmarks, change_time):

    F_mask = np.zeros((3, 57)) # 57 as it is 19 x 3

    F_mask[0][0] = 1
    F_mask[1][1] = 1
    F_mask[2][2] = 1

    systemVariance = np.array([[20.33, 0, 0], [0, 3.6, 0], [0, 0, 0.001]])

    #    est_current_mew = prev_mew
    # make a column vector that we then multipy by F_mask Transpose
    temp = np.array([[(-v_vect[0]/v_vect[1]*math.sin(prev_mew[-1][2]) + (v_vect[0]/v_vect[1] * math.sin(prev_mew[-1][2]+v_vect[1]*change_time))),
                      ((v_vect[0] / v_vect[1] * math.cos(prev_mew[-1][2])) - (v_vect[0] / v_vect[1] * math.cos(
                          prev_mew[-1][2] + v_vect[1] * change_time))), v_vect[1]*change_time]])

    est_mew = (np.transpose(F_mask) @ temp.T) # should all be adding the previous mew value (as a vector)

    est_current_mew = prev_mew[-1] + est_mew

    #est_current_mew.append(prev_mew[-1] + (np.transpose(F_mask) @ ((((-v_vect[0] / v_vect[1]) * math.sin(prev_mew[-1][2])) + (
    #            v_vect[0] / v_vect[1]) * math.sin(prev_mew[-1][2] + v_vect[1] * change_time)),
    #                      (((v_vect[0] / v_vect[1]) * math.cos(prev_mew[-1][2])) - (
    #                                  v_vect[0] / v_vect[1]) * math.cos(
    #                          prev_mew[-1][2] + v_vect[1] * change_time)), v_vect[1] * change_time)))

    I = np.identity(len(covariance))
    FT = np.transpose(F_mask)

    mat_X = np.array(
        [[0, 0, (((v_vect[0] / v_vect[1]) * math.cos(prev_mew[-1][2])) - (v_vect[0] / v_vect[1]) * math.cos(
            prev_mew[-1][2] + v_vect[1] * change_time))],
         [0, 0, (((v_vect[0] / v_vect[1]) * math.sin(prev_mew[-1][2])) - (v_vect[0] / v_vect[1]) * math.sin(
             prev_mew[-1][2] + v_vect[1] * change_time))],
         [0, 0, 0]])

    taylor_exp = I + (FT[0:3, 0:3] @ mat_X @ F_mask[0:3, 0:3])
    # taylor_exp =  np.array([np.identity(len(covariance)) + np.transpose(F_mask) @ [
    #    [0, 0, (((v_vect[0] / v_vect[1]) * math.cos(prev_mew[-1][2])) - (v_vect[0] / v_vect[1]) * math.cos(
    #        prev_mew[-1][2] + v_vect[1] * change_time))]],
    #   [ [0, 0, (((v_vect[0] / v_vect[1]) * math.sin(prev_mew[-1][2])) - (v_vect[0] / v_vect[1]) * math.sin(
    #        prev_mew[-1][2] + v_vect[1] * change_time))]],
    #    [[0, 0, 0]]  @ F_mask])

    updated_covariance = taylor_exp @ covariance @ np.transpose(taylor_exp) + np.transpose(F_mask[0:3, 0:3]) @ systemVariance @ F_mask[0:3, 0:3]

    sensorReading = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    sensor_jacobian = []
    trident = []
    print(str(est_mew))
    F_mask_k= np.zeros((57,57)) # 3 times the number of landmarks
    F_mask_k[0][0] = 1
    F_mask_k[1][1] = 1
    F_mask_k[2][2] = 1
    matrixK = []
    est_seen = []
    done = 0
    for seen in seen_landmarks:
        j = landmarks[seen.object_id-1]
        kp = j * 3 + 1
        print("dimensions: ", np.shape(est_mew))
        print("seen.pose: ", seen.pose)
        Pose = Frame2D.fromPose(seen.pose)
        if est_mew[kp][0] == 0:
            est_mew[kp] = Pose.x()
            est_mew[kp+1] = Pose.y()
            est_mew[kp+2] = Pose.angle()

        print("object position: ", est_mew)
        delta = [est_mew[kp] - est_mew[0], est_mew[kp+1] - est_mew[1]]

        q_k = np.transpose(delta) @ delta

        est_seen.append([math.sqrt(q_k), math.atan2(delta[1], delta[0]) - est_mew[2], est_current_mew[2]])
        F_mask_k[kp][kp] = 1
        F_mask_k[kp+1][kp+1] = 1
        F_mask_k[kp+2][kp+2] = 1

        F_temp = np.zeros((6, 6))
        F_temp[0][0] = 1
        F_temp[1][1] = 1
        F_temp[2][2] = 1
        F_temp[3][3] = 1
        F_temp[4][4] = 1
        F_temp[5][5] = 1

        grape = [[math.sqrt(q_k) * delta[0], -math.sqrt(q_k) * delta[1], 0, -math.sqrt(q_k) * delta[0], math.sqrt(q_k) * delta[1], 0],
                                   [delta[1], delta[0], -1, -delta[1], -delta[0], 0], [0, 0, 0, 0, 0, 1]]

        print(np.shape(grape))
        H2 = grape @ F_temp
        print(H2)
        H = np.transpose(H2 / q_k)
        first_entry = updated_covariance.dot(H.T)
        print(first_entry)
        print(H)
        print(updated_covariance)
        print(str(np.shape(H)))
        print(str(np.shape(updated_covariance)))
        #matrixK = updated_covariance @ np.transpose[sensor_jacobean] @ np.linalg.inv(sensor_jacobean @ updated_covariance @ np.transpose(sensor_jacobean) + sensorReading)

        matrixK.append(first_entry.dot((H.astype(float).dot(updated_covariance.astype(float)).dot(np.transpose(H).astype(float)))))
        sensor_jacobian.append(H)
    done = 1
    sum_matk = [0, 0, 0]

    if done == 1:
        i = 0
        for seen in seen_landmarks:
            #i = seen.object_id - 1
            Pose = Frame2D.fromPose(seen.pose)
            z = np.array([Pose.x(), Pose.y(), Pose.angle()])
            print(np.shape(np.transpose(matrixK[i])))
            qwer = np.transpose(matrixK[i])
            print(np.shape(sum_matk))
            sum_matk = (qwer.dot((z - est_seen[i])))
        i += 1
    mew2 = est_mew + sum_matk


    sum_matk2 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    if done == 1:
        i = 0
        for seen in seen_landmarks:
            sum_matk2 = matrixK[i].dot( sensor_jacobian[i])
            i += 1


    new_sigma = (I - sum_matk2) @ updated_covariance

    return mew2, new_sigma


    # Fx mask

    #first motion model

    #sensor matrix

    #for all observed features
    #j = landmark_index
    #if landmark j is new

    #endif

    #deviation of map (how much change)

    #whats the rest? line 13 - 20

