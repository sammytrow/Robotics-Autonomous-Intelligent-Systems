import math


def calc_prob(mew, var):
    p = 1

    if var == 0:
        p = math.exp(-0)
    else:
        p = math.exp(-((mew ** 2) / (2 * var)))

    return p


def motion_model(current_pose, velocity, previous_pose, time):
    pose_px = previous_pose[0]  # x of previous
    pose_py = previous_pose[1]  # y of previous
    pose_cx = current_pose[0]  # x of current
    pose_cy = current_pose[1]  # y of current
    ydist = pose_cy - pose_py  # distance between y coords of current and previous
    xdist = pose_cx - pose_px  # distance between x coords of current and previous
    r = math.sqrt((xdist) ** 2 + (ydist) ** 2)  # Euclidean distance between current and previous using pythagoras

    print(str("previous: ") + str(previous_pose))
    print(str("current:" + str(current_pose)))

    mew = abs(0.5 * (((pose_px - pose_cx) * math.cos(previous_pose[2]) + (
            (pose_py - pose_cy) * math.sin(previous_pose[2]))) /
                     ((pose_py - pose_cy) * math.cos(previous_pose[2]) - (
                             (pose_px - pose_cx) * math.sin(previous_pose[2])))))

    estimate_x = ((pose_px + pose_cx) / 2) + (mew * (pose_py - pose_cy))
    estimate_y = ((pose_py + pose_cy) / 2) + (mew * (pose_px - pose_cx))
    estimate_dist = math.sqrt((pose_px - estimate_x) ** 2 + (pose_py - estimate_y) ** 2)

    print("Estimate x cord:" + str(estimate_x))
    print("Estimate y cord:" + str(estimate_y))
    print("Estimate dist:" + str(estimate_dist))

    delta_a = math.atan2(pose_cy - estimate_y, pose_cx - estimate_x) - math.atan2(pose_py - estimate_y,
                                                                                  pose_px - estimate_x)

#    print(str(math.atan2(pose_cy - estimate_y, pose_cx - estimate_x)))
  #  print(str(math.atan2(pose_py - estimate_y, pose_px - estimate_x)))
    #print(str(delta_a))

    estimate_fv = (delta_a / time) * estimate_dist  # estimate forward velocity
    estimate_av = (delta_a / time)
    gamma_hat = ((current_pose[2] - previous_pose[2] / time) - estimate_av)

    # temporary noise value calculated for real world experiments
    alpha_pred = 0.1
    x_variance = 20.33  # will need to update
    y_variance = 3.6  # will need to update
    a_variance = 0.001  # will need to update
    # we will use a zero centred mean, i,e. mew = estimate - mean = estimate
    noise = (alpha_pred*velocity[0]) + (alpha_pred*velocity[1])
    print(str(noise))
    p1 = calc_prob(estimate_fv, (velocity[0] - estimate_fv))  # need to do prob for angular_vel and gamma_hat
    p2 = calc_prob(estimate_av, (velocity[1] - estimate_av))
    p3 = calc_prob(gamma_hat, 0)

    p = p1*p2*p3
    print(str(p))
    return p

currentpose = [250.0, 112.0, 0.0]
previouspose = [120.0, 45.0, 1.63]
velocity = [25.0, 10]
time = 50
motion_model(currentpose, velocity, previouspose, time)