import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import random
from frame2d import Frame2D
import math


def explore(robotpose: Frame2D):
    axis = random.choice([0, 1])
    direction = random.choice([0, 1])
    move = random.randrange(100, 400)

    if axis is 0:
        if direction is 1:
            new_pose = [robotpose.x(), robotpose.y(), (robotpose.angle() + -math.pi/2)] #math.radians(-90))]
            time = 2
        else:
            new_pose = [robotpose.x(), robotpose.y(), (robotpose.angle() + math.pi/2)]
            time = 2

    elif (robotpose.y() + move) < 800:
        new_pose = [robotpose.x(), (robotpose.y() + move), robotpose.angle()]
        time = round((new_pose[1] / 100) * 2)
    else:
        new_pose = [robotpose.x(), robotpose.y(), (robotpose.angle() + math.pi / 2)]
        time = 2


    return new_pose, time


def found_item(item):
    if item[0] == "cube":
        size = [[item[1][0] - 25, item[1][1] - 25], [item[1][0] - 25, item[1][1] + 25],
                [item[1][0] + 25, item[1][1] + 25], [item[1][0] + 25, item[1][1] - 25]]
    else:
        pass
    return size


# items = [["cube", [300, 100]], ["cube", [400, 700]], ["cube", [200, 150]]]  # , ["wall", [400,100]]]


def plot_item(item: Frame2D, type, id):
    x = item.x()
    y = item.y()
    plt.scatter(x, y)
    plt.annotate(str(type + id), xy=(x, y))
    plt.draw()
    plt.pause(0.01)


def plot_item2(item: list, type, id):
    x = item[0]
    y = item[1]
    plt.scatter(x, y)
    #plt.annotate(str(type + id), xy=(x, y))
    plt.draw()
    plt.pause(0.01)


def initialize_plot():
    plt.ion()
    plt.ylim(0, 840)
    plt.xlim(0, 640)


def run_voronoi(landmarks):
    points = []
    for j in landmarks:
        object = [["cube"][j[1][0], j[1][1]]]
        points.append(found_item(object))
    vor = Voronoi(points, incremental=True)
    voronoi_plot_2d(vor)
    # plt.plot(vor)
    plt.show()

# plt.plot()
