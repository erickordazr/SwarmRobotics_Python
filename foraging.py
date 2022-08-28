# -*- coding: utf-8 -*-
"""
Swarm Robotics - Foraging v0.1
Author: Erick de Jesús Ordaz Rivas
Email: erick.ordazrv@uanl.edu.mx
"""
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
import math
import random
import time



def normalize(r, lb, ub):
    return (r - lb) / ((ub - lb) + 0.000001)


def denormalize(n, lb, ub):
    return n * (ub - lb) + lb

# Dynamic Model
def model(c, t, u):
    # Robot parameters
    m = 0.38  # Robot mass (kg)
    Im = 0.005  # Moment of inertia
    d = 0.02  # Distance from centroid to wheel axis (m)
    r = 0.03  # Wheel radius (m)
    R = 0.05  # Distance to wheel-center (m)

    M = np.matrix([[m, 0], [0, Im + m * d ** 2]])  # Inertia matrix
    H = np.array([[-m * d * c[5] ** 2], [m * d * c[4] * c[5]]])  # Coriolis matrix
    B = np.matrix([[1 / r, 1 / r], [R / r, -R / r]])  # Conversion matrix torque-wheel-movil force
    A = np.matrix([[r / 2, r / 2], [r / (2 * R), -r / (2 * R)]])
    Ts = np.matrix([[0.434, 0], [0, 0.434]])
    Ks = np.matrix([[2.745, 0], [0, 2.745]])
    Kl = np.matrix([[1460.2705, 0], [0, 1460.2705]])

    dxdt = np.array(np.concatenate((
        np.asarray(np.matrix([[np.cos(c[3]), -d * np.sin(c[3])], [np.sin(c[3]), d * np.cos(c[3])]]) * np.array(
            [[c[4]], [c[5]]])),
        np.array([[c[4]], [c[5]]]),
        np.linalg.inv(M + B * np.linalg.inv(Kl) * Ts * np.linalg.inv(A)) * (B * np.linalg.inv(Kl) * Ks * u - (
                H + B * np.linalg.inv(Kl) * np.linalg.inv(A) * np.array([[c[4]], [c[5]]])))
    ), axis=0))

    dxdt.reshape(6, 1)
    return np.squeeze(np.asarray(dxdt))


def movement(ci, u):
    initial_time = 0
    final_time = 1
    t = np.linspace(initial_time, final_time, 11)
    c = odeint(model, ci, t, args=(u,))

    rows = len(c)
    c_ = c[rows - 1, :]
    return c_


def foraging(objects, individuals, r_r, o_r, a_r, animation):
    # [report, ob_ep, ob_ip, collectedObjects, optimization_functions] = foraging(objects, individuals, r_r, o_r, a_r, animation)
    cs = np.zeros((individuals, 6))  # Initial individuals states
    c = np.zeros((individuals, 6))  # Individuals states
    iterations = 0  # Iterations
    report = np.zeros((100000, individuals, 4))  # States report
    collectedObjects = np.zeros((individuals, 4))  # Delivery time,Search time,collected objects
    area_limits = 10  # Area limit
    nest_radius = 4  # Maximum distance of influence (nest)
    box_radius = 2.5  # Maximum distance of influence (objects box)
    wn = random.random() * 0.01  # White noise
    explore = np.zeros((individuals, 1))
    state_detected = np.zeros((100000, individuals))
    
    # Objective functions
    optimization_functions = np.zeros((6,1))
    f1 = 0  # execution time
    f2 = 0  # energy used
    f3 = 0  # number of members of the swarm
    f4 = 0  # swarm efficiency
    f5 = 0  # task balance
    f6 = 0 # uncollected objects

    if animation:
        plt.figure(figsize=(10, 10), dpi=80)
        # ax = plt.gca()  # Nest full (end task)

    desired_voltage = np.zeros((individuals, 2))
    repulsion_voltage = 2  # 15 cm/s
    orientation_voltage = 2.7  # 20 cm/s
    attraction_voltage = 3.7  # 30 cm/s
    influence_voltage = orientation_voltage

    repulsion_radius = 0.075 + r_r
    orientation_radius = 0.075 + o_r
    attraction_radius = 0.075 + a_r

    gripState = np.zeros((individuals, 1))  # Open grip/Close grip
    nestFull = objects

    # Objects box
    box_center = 0.75
    box_limits = 0.2
    objectbox = [box_center * area_limits, box_center * area_limits]
    ob_ip = np.zeros((objects, 2))
    ob_ep = np.zeros((objects, 2))  # Initial position of objects

    objects_location = np.zeros((objects, 2))  # Objects location
    if objects == 0:
        obv = np.zeros((1, 1))  # Objects vector
        goi = np.zeros((1, 1))  # Objects gripped by individual
    else:
        obv = np.zeros((objects, 1))  # Objects vector
        goi = np.zeros((objects, 1))  # Objects gripped by individual

    # Random objects position
    for o in range(objects):
        obRand1 = denormalize(random.random(), box_center - (box_limits / 2), box_center + (box_limits / 2))
        obRand2 = denormalize(random.random(), box_center - (box_limits / 2), box_center + (box_limits / 2))
        objects_location[o, 0] = area_limits * obRand1
        objects_location[o, 1] = area_limits * obRand2
        ob_ip[o, 0] = objects_location[o, 0]
        ob_ip[o, 1] = objects_location[o, 1]
        obv[o, 0] = 1  # Search mode

    # Nest
    nest_arealimits = 0.2
    nest_dot = np.zeros(2) + area_limits * (nest_arealimits / 2)  # Nest dot
    nest_location = [nest_dot[0], nest_dot[1]]  # Nest location (Dotted line)
    nest_influence = np.zeros(individuals)  # Influence of nest activated by individual

    # individual initial conditions
    for i in range(individuals):
        if i == 0:
            c[i, 0] = area_limits * random.uniform(0, nest_arealimits)  # x position
            c[i, 1] = area_limits * random.uniform(0, nest_arealimits)  # y position
        else:
            while True:
                safe = 0
                c[i, 0] = area_limits * random.uniform(0, 0.2)  # x position
                c[i, 1] = area_limits * random.uniform(0, 0.2)  # y position
                for j in range(i):
                    d_other = math.sqrt((c[i, 0] - c[j, 0]) ** 2 + (c[i, 1] - c[j, 1]) ** 2)
                    if d_other > 0.3:
                        safe = safe + 1
                if safe == i:
                    break
        c[i, 2] = 0  # Movement
        c[i, 3] = random.uniform(0, 2 * math.pi)  # Orientation
        c[i, 4] = 0  # Speed
        c[i, 5] = 0  # Angular speed
    dirExp = c[:, 3]

    # Finish task when nest is full
    while nestFull != 0 and iterations<6000:
        iterations = iterations + 1

        for i in range(individuals):

            desired_voltage[i, :] = [orientation_voltage + wn, orientation_voltage + wn]

            # Elements detected
            repulsion_walls = np.zeros(individuals)

            repulsion_detected = np.zeros(individuals)
            orientation_detected = np.zeros(individuals)
            attraction_detected = np.zeros(individuals)
            elements_rx = []
            elements_ry = []
            elements_ox = []
            elements_oy = []
            elements_ax = []
            elements_ay = []

            # perception range
            repulsion_range = 6.28319
            orientation_range = 0.5235988
            attraction_range = 0.5235988
            influence_range = 0.5235988
            nest_range = 3.14159
            objectbox_range = 3.14159

            # Verify each sensor for repulsion of walls
            for w in range(5):
                if w == 0:
                    dirObs = c[i, 3] - 3.83972
                else:
                    dirObs = dirObs + 1.91986
                if dirObs < 0:
                    dirObs = dirObs + (2 * math.pi)
                elif dirObs > (2 * math.pi):
                    dirObs = dirObs - (2 * math.pi)
                Dir = [math.cos(dirObs), math.sin(dirObs)]
                limitX = c[i, 0] + (Dir[0] * repulsion_radius)
                limitY = c[i, 1] + (Dir[1] * repulsion_radius)

                # Resulting direction due exploration
                if limitX > area_limits or limitX < 0 or limitY > area_limits or limitY < 0:
                    explore[i] = 0
                    dirExp[i] = dirObs + (3 * math.pi / 4) + (random.uniform(0, 1) * math.pi / 2)
                    repulsion_walls[i] = repulsion_walls[i] + 1
                    if dirExp[i] > (2 * math.pi):
                        dirExp[i] = dirExp[i] - (2 * math.pi)
                    elif dirExp[i] < 0:
                        dirExp[i] = dirExp[i] + (2 * math.pi)

            # Resulting direction due object box
            objectbox_angle = math.atan2(objectbox[1] - c[i, 1], objectbox[0] - c[i, 0])
            if objectbox_angle < 0:
                objectbox_angle = objectbox_angle + (2 * math.pi)
            elif objectbox_angle > (2 * math.pi):
                objectbox_angle = objectbox_angle - (2 * math.pi)

            # Calculation of influence angles by object box
            ob_Beta = objectbox_angle - c[i, 3]
            if ob_Beta < 0:
                ob_Beta = ob_Beta + (2 * math.pi)

            ob_Gamma = c[i, 3] - objectbox_angle
            if ob_Gamma < 0:
                ob_Gamma = ob_Gamma + (2 * math.pi)

            if ob_Gamma < ob_Beta:
                ob_Delta = ob_Gamma
            else:
                ob_Delta = ob_Beta

            # Calculated distance between the robots and the object zone
            if ob_Delta < objectbox_range / 2:
                objectbox_distance = math.sqrt((c[i, 0] - objectbox[0]) ** 2 + (c[i, 1] - objectbox[1]) ** 2)
            else:
                objectbox_distance = math.inf

            ob_n = normalize(objectbox_distance, 0, box_radius)
            influence_voltage = denormalize(ob_n, repulsion_voltage, attraction_voltage)

            for j in range(individuals):
                if i != j:
                    # Angle of the individual with respect to other members of the swarm
                    neighbors_angle = math.atan2((c[j, 1] - c[i, 1]), (c[j, 0] - c[i, 0]))
                    if neighbors_angle > (2 * math.pi):
                        neighbors_angle = neighbors_angle - (2 * math.pi)
                    elif neighbors_angle < 0:
                        neighbors_angle = neighbors_angle + (2 * math.pi)

                    # Calculation of angles of repulsion and attraction with respect to other individuals
                    Beta = neighbors_angle - c[i, 3]
                    if Beta < 0:
                        Beta = Beta + (2 * math.pi)

                    Gamma = c[i, 3] - neighbors_angle
                    if Gamma < 0:
                        Gamma = Gamma + (2 * math.pi)

                    if Gamma < Beta:
                        Delta = Gamma
                    else:
                        Delta = Beta

                    # Calculation of the repulsion distance with respect to other individuals
                    if Delta < repulsion_range / 2:
                        repulsion_distance = math.sqrt((c[i, 0] - c[j, 0]) ** 2 + (c[i, 1] - c[j, 1]) ** 2)
                    else:
                        repulsion_distance = math.inf

                    # Calculation of the attraction distance with respect to other individuals
                    if Delta < attraction_range / 2:
                        attraction_distance = math.sqrt((c[i, 0] - c[j, 0]) ** 2 + (c[i, 1] - c[j, 1]) ** 2)
                    else:
                        attraction_distance = math.inf

                    # Calculation of the orientation distance with respect to other individuals
                    if Delta < orientation_range / 2:
                        orientation_distance = math.sqrt((c[i, 0] - c[j, 0]) ** 2 + (c[i, 1] - c[j, 1]) ** 2)
                    else:
                        orientation_distance = math.inf

                    # Number of individuals detected in the radius of repulsion, orientation and attraction
                    if repulsion_distance <= repulsion_radius:
                        elements_rx.append(math.cos(neighbors_angle))
                        elements_ry.append(math.sin(neighbors_angle))
                        repulsion_detected[i] = repulsion_detected[i] + 1

                    if orientation_radius < attraction_distance <= attraction_radius and \
                            repulsion_detected[i] == 0 and repulsion_walls[i] == 0:
                        elements_ax.append(math.cos(neighbors_angle))
                        elements_ay.append(math.sin(neighbors_angle))
                        attraction_detected[i] = attraction_detected[i] + 1

                    if repulsion_radius < orientation_distance <= orientation_radius and \
                            repulsion_detected[i] == 0 and attraction_detected[i] == 0 and repulsion_walls[i] == 0:
                        elements_ox.append(math.cos(c[j, 3]))
                        elements_oy.append(math.sin(c[j, 3]))
                        orientation_detected[i] = orientation_detected[i] + 1

            for o in range(objects):
                # Search object
                if obv[o, 0] == 1 and gripState[i, 0] == 0:

                    # Angle of objects
                    object_angle = math.atan2((objects_location[o, 1] - c[i, 1]), (objects_location[o, 0] - c[i, 0]))
                    if object_angle < 0:
                        object_angle = object_angle + (2 * math.pi)
                    elif object_angle > (2 * math.pi):
                        object_angle = object_angle - (2 * math.pi)

                    # Calculation of object angles
                    o_Beta = object_angle - c[i, 3]
                    if o_Beta < 0:
                        o_Beta = o_Beta + (2 * math.pi)

                    o_Gamma = c[i, 3] - object_angle
                    if o_Gamma < 0:
                        o_Gamma = o_Gamma + (2 * math.pi)

                    if o_Gamma < o_Beta:
                        o_Delta = o_Gamma
                    else:
                        o_Delta = o_Beta

                    # Noise for distance value in influence (standard distribution) of 5%
                    ds1 = 0
                    ds2 = 0
                    for n in range(12):
                        ds1 = ds1 + random.uniform(-1, 1)
                    for m in range(6):
                        ds2 = ds2 + random.uniform(-1, 1)
                    object_noise = (ds1 - ds2) * 0.05

                    # Calculated influence distance
                    if o_Delta < influence_range / 2:
                        object_distance = math.sqrt((c[i, 0] - objects_location[o, 0]) ** 2 + (
                                c[i, 1] - objects_location[o, 1]) ** 2) + object_noise
                    else:
                        object_distance = math.inf

                    # Distance between objects and robot
                    object_limit = 0.2
                    if object_distance <= object_limit:
                        nest_influence[i] = 1
                        goi[o] = i
                        collectedObjects[i, 2] = collectedObjects[i, 2] + 1
                        gripState[i, 0] = 1  # Close grip
                        obv[o, 0] = 0  # Object taken by robot

                # Nest delivery
                if obv[o, 0] == 0 and nest_influence[i] == 1:

                    # Angle respect to nest
                    nest_angle = math.atan2((nest_location[1] - c[i, 1]), (nest_location[0] - c[i, 0]))
                    if nest_angle < 0:
                        nest_angle = nest_angle + (2 * math.pi)
                    elif nest_angle > (2 * math.pi):
                        nest_angle = nest_angle - (2 * math.pi)

                    # Calculation of nest angles
                    n_Beta = nest_angle - c[i, 3]
                    if n_Beta < 0:
                        n_Beta = n_Beta + (2 * math.pi)

                    n_Gamma = c[i, 3] - nest_angle
                    if n_Gamma < 0:
                        n_Gamma = n_Gamma + (2 * math.pi)

                    if n_Gamma < n_Beta:
                        n_Delta = n_Gamma
                    else:
                        n_Delta = n_Beta

                    # Calculated nest distance
                    if n_Delta < nest_range / 2:
                        nest_distance = math.sqrt((nest_location[0] - c[i, 0]) ** 2 + (nest_location[1] - c[i, 1]) ** 2)
                    else:
                        nest_distance = math.inf

                    # Distance between nest and robot
                    nest_limit = 0.2
                    if nest_distance <= nest_limit:
                        objects_location[o, 0] = nest_location[0]
                        objects_location[o, 1] = nest_location[1]
                        nest_dot[0] = nest_dot[0] + random.uniform(-0.1, 0.1)
                        nest_dot[1] = nest_dot[1] + random.uniform(-0.1, 0.1)
                        nest_location[0] = nest_dot[0]
                        nest_location[1] = nest_dot[1]
                        nestFull = nestFull - 1
                        gripState[i, 0] = 0  # Open grip
                        nest_influence[i] = 0

                    nest_n = normalize(nest_distance, nest_limit, nest_radius)
                    influence_voltage = denormalize(nest_n, repulsion_voltage, attraction_voltage)

                    if gripState[int(goi[o, 0]), 0] == 1:
                        objects_location[o, 0] = c[int(goi[o, 0]), 0]
                        objects_location[o, 1] = c[int(goi[o, 0]), 1]

                ob_ep[o, 0] = objects_location[o, 0]
                ob_ep[o, 1] = objects_location[o, 1]

            # Average of detected elements
            if repulsion_detected[i] > 0:
                repulsion_direction = math.atan2((-np.sum(elements_ry)), (-np.sum(elements_rx)))
                if repulsion_direction < 0:
                    repulsion_direction = repulsion_direction + (2 * math.pi)
                elif repulsion_direction > (2 * math.pi):
                    repulsion_direction = repulsion_direction - (2 * math.pi)

            if orientation_detected[i] > 0:
                orientation_direction = math.atan2((np.sum(elements_oy)), (np.sum(elements_ox)))
                if orientation_direction < 0:
                    orientation_direction = orientation_direction + (2 * math.pi)
                elif orientation_direction > (2 * math.pi):
                    orientation_direction = orientation_direction - (2 * math.pi)

            if attraction_detected[i] > 0:
                attraction_direction = math.atan2((np.sum(elements_ay)), (np.sum(elements_ax)))
                if attraction_direction < 0:
                    attraction_direction = attraction_direction + (2 * math.pi)
                elif attraction_direction > (2 * math.pi):
                    attraction_direction = attraction_direction - (2 * math.pi)

            # Behavior Policies
            # Repulsion rules
            if repulsion_walls[i] > 0:
                state_detected[iterations, i] = 1

            if repulsion_detected[i] > 0:
                state_detected[iterations, i] = 1
                if nest_influence[i] == 0:
                    if objectbox_distance < box_radius:
                        desired_voltage[i, :] = [repulsion_voltage + wn, repulsion_voltage + wn]
                        xT = 0.5 * math.cos(c[i, 3]) + 0.4 * math.cos(repulsion_direction) + 0.1 * math.cos(
                            objectbox_angle)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.4 * math.sin(repulsion_direction) + 0.1 * math.sin(
                            objectbox_angle)
                        c[i, 3] = math.atan2(yT, xT)
                        # dirExp[i] = c[i,3]
                        explore[i] = 1
                    else:
                        desired_voltage[i, :] = [repulsion_voltage + wn, repulsion_voltage + wn]
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(repulsion_direction)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(repulsion_direction)
                        c[i, 3] = math.atan2(yT, xT)
                        dirExp[i] = repulsion_direction
                else:
                    if nest_distance < nest_radius:
                        desired_voltage[i, :] = [repulsion_voltage + wn, repulsion_voltage + wn]
                        xT = 0.5 * math.cos(c[i, 3]) + 0.4 * math.cos(repulsion_direction) + 0.1 * math.cos(nest_angle)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.4 * math.sin(repulsion_direction) + 0.1 * math.sin(nest_angle)
                        c[i, 3] = math.atan2(yT, xT)
                        # dirExp[i] = c[i,3]
                    else:
                        desired_voltage[i, :] = [repulsion_voltage + wn, repulsion_voltage + wn]
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(repulsion_direction)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(repulsion_direction)
                        c[i, 3] = math.atan2(yT, xT)
                        dirExp[i] = repulsion_direction

            # Orientation rules
            if orientation_detected[i] > 0 and repulsion_detected[i] == 0 and attraction_detected[i] == 0 and repulsion_walls[i] == 0:
                if nest_influence[i] == 0:
                    if objectbox_distance < box_radius:
                        state_detected[iterations, i] = 4
                        desired_voltage[i, :] = [orientation_voltage + wn, orientation_voltage + wn]
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(objectbox_angle)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(objectbox_angle)
                        c[i, 3] = math.atan2(yT, xT)
                        explore[i] = 1
                        # dirExp[i] = orientation_direction
                    else:
                        state_detected[iterations, i] = 2
                        desired_voltage[i, :] = [orientation_voltage + wn, orientation_voltage + wn]
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(orientation_direction)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(orientation_direction)
                        c[i, 3] = math.atan2(yT, xT)
                        dirExp[i] = orientation_direction
                else:
                    if nest_distance < nest_radius:
                        state_detected[iterations, i] = 4
                        desired_voltage[i, :] = [orientation_voltage + wn, orientation_voltage + wn]
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(nest_angle)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(nest_angle)
                        c[i, 3] = math.atan2(yT, xT)
                        # dirExp[i] = orientation_direction
                    else:
                        state_detected[iterations, i] = 2
                        desired_voltage[i, :] = [orientation_voltage + wn, orientation_voltage + wn]
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(orientation_direction)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(orientation_direction)
                        c[i, 3] = math.atan2(yT, xT)
                        dirExp[i] = orientation_direction

            # Attraction rules
            if attraction_detected[i] > 0 and repulsion_detected[i] == 0 and orientation_detected[i] == 0 and repulsion_walls[i] == 0:
                if nest_influence[i] == 0:
                    if objectbox_distance < box_radius:
                        state_detected[iterations, i] = 4
                        desired_voltage[i, :] = [orientation_voltage + wn, orientation_voltage + wn]
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(objectbox_angle)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(objectbox_angle)
                        c[i, 3] = math.atan2(yT, xT)
                        explore[i] = 1
                        # dirExp[i] = attraction_direction
                    else:
                        state_detected[iterations, i] = 3
                        desired_voltage[i, :] = [attraction_voltage + wn, attraction_voltage + wn]
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(attraction_direction)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(attraction_direction)
                        c[i, 3] = math.atan2(yT, xT)
                        dirExp[i] = attraction_direction
                else:
                    if nest_distance < nest_radius:
                        state_detected[iterations, i] = 4
                        desired_voltage[i, :] = [orientation_voltage + wn, orientation_voltage + wn]
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(nest_angle)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(nest_angle)
                        c[i, 3] = math.atan2(yT, xT)
                        # dirExp[i] = attraction_direction
                    else:
                        state_detected[iterations, i] = 3
                        desired_voltage[i, :] = [attraction_voltage + wn, attraction_voltage + wn]
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(attraction_direction)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(attraction_direction)
                        c[i, 3] = math.atan2(yT, xT)
                        dirExp[i] = attraction_direction

            # Orientation and Attraction rules
            if orientation_detected[i] > 0 and attraction_detected[i] > 0 and repulsion_detected[i] == 0 and repulsion_walls[i] == 0:
                if nest_influence[i] == 0:
                    if objectbox_distance < box_radius:
                        state_detected[iterations, i] = 4
                        desired_voltage[i, :] = [orientation_voltage + wn, orientation_voltage + wn]
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(objectbox_angle)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(objectbox_angle)
                        c[i, 3] = math.atan2(yT, xT)
                        explore[i] = 1
                        # dirExp[i] = attraction_direction
                    else:
                        state_detected[iterations, i] = 5
                        desired_voltage[i, :] = [orientation_voltage + wn, orientation_voltage + wn]
                        xT = 0.5 * math.cos(c[i, 3]) + 0.25 * math.cos(orientation_direction) + 0.25 * math.cos(
                            attraction_direction)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.25 * math.sin(orientation_direction) + 0.25 * math.sin(
                            attraction_direction)
                        c[i, 3] = math.atan2(yT, xT)
                        dirExp[i] = math.atan2((math.sin(orientation_direction) + math.sin(attraction_direction)),
                                                (math.cos(orientation_direction) + math.cos(attraction_direction)))
                else:
                    if nest_distance < nest_radius:
                        state_detected[iterations, i] = 4
                        desired_voltage[i, :] = [orientation_voltage + wn, orientation_voltage + wn]
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(nest_angle)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(nest_angle)
                        c[i, 3] = math.atan2(yT, xT)
                        # dirExp[i] = attraction_direction
                    else:
                        state_detected[iterations, i] = 5
                        desired_voltage[i, :] = [orientation_voltage + wn, orientation_voltage + wn]
                        xT = 0.5 * math.cos(c[i, 3]) + 0.25 * math.cos(orientation_direction) + 0.25 * math.cos(
                            attraction_direction)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.25 * math.sin(orientation_direction) + 0.25 * math.sin(
                            attraction_direction)
                        c[i, 3] = math.atan2(yT, xT)
                        dirExp[i] = math.atan2((math.sin(orientation_direction) + math.sin(attraction_direction)),
                                                (math.cos(orientation_direction) + math.cos(attraction_direction)))

            # Out of range
            if attraction_detected[i] == 0 and repulsion_detected[i] == 0 and orientation_detected[i] == 0 and repulsion_walls[i] == 0:
                if nest_influence[i] == 0:
                    if objectbox_distance < box_radius:
                        state_detected[iterations, i] = 4
                        desired_voltage[i, :] = [orientation_voltage + wn, orientation_voltage + wn]
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(objectbox_angle)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(objectbox_angle)
                        c[i, 3] = math.atan2(yT, xT)
                        explore[i] = 1
                    else:
                        state_detected[iterations, i] = 0
                        desired_voltage[i, :] = [orientation_voltage + wn, orientation_voltage + wn]
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(dirExp[i])
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(dirExp[i])
                        c[i, 3] = math.atan2(yT, xT)
                else:
                    if nest_distance < nest_radius:
                        state_detected[iterations, i] = 4
                        desired_voltage[i, :] = [orientation_voltage + wn, orientation_voltage + wn]
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(nest_angle)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(nest_angle)
                        c[i, 3] = math.atan2(yT, xT)
                    else:
                        state_detected[iterations, i] = 0
                        desired_voltage[i, :] = [orientation_voltage + wn, orientation_voltage + wn]
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(dirExp[i])
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(dirExp[i])
                        c[i, 3] = math.atan2(yT, xT)

            if explore[i] == 1 and objectbox_distance > box_radius and random.random() < 0.1:
                explore[i] = 0
                dirExp[i] = dirExp[i] + (3 * math.pi / 4) + (random.random() * math.pi / 2)
                if dirExp[i] < 0:
                    dirExp[i] = dirExp[i] + (2 * math.pi)
                elif dirExp[i] > (2 * math.pi):
                    dirExp[i] = dirExp[i] - (2 * math.pi)

            report[iterations, i, 0] = c[i, 0]
            report[iterations, i, 1] = c[i, 1]
            report[iterations, i, 2] = c[i, 2]
            collectedObjects[i, 3] = c[i, 2]
            report[iterations, i, 3] = c[i, 3]

            Cpast = c[i, :]
            cs[i, :] = movement(c[i, :], desired_voltage[i, :].reshape(2, 1))
            c[i, :] = cs[i, :]

            if c[i, 0] < (0 + (repulsion_radius / 2)) or c[i, 0] > (area_limits - (repulsion_radius / 2)) or c[i, 1] < (
                    0 + (repulsion_radius / 2)) or c[i, 1] > (area_limits - (repulsion_radius / 2)):
                c[i, :] = Cpast

            # this avoids an infinite increment of radians
            if c[i, 3] > (2 * math.pi):
                c[i, 3] = c[i, 3] - (2 * math.pi)
            elif c[i, 3] < 0:
                c[i, 3] = c[i, 3] + (2 * math.pi)

            # Delivery time
            if gripState[i, 0] == 1:  # Grip close
                collectedObjects[i, 0] = collectedObjects[i, 0] + 1

        # Simulation
        if animation:

            # plt.figure(figsize=(7, 7), dpi=80)
            ax = plt.gca()
            x = report[iterations, :, 0]
            y = report[iterations, :, 1]
            vx = np.cos(report[iterations, :, 3])
            vy = np.sin(report[iterations, :, 3])
            box_circle = plt.Circle((objectbox[0], objectbox[1]), box_radius, color='blue', alpha=0.6, fill=False)
            box_rectangle = plt.Rectangle(
                ((box_center - (box_limits / 2)) * area_limits, (box_center - (box_limits / 2)) * area_limits),
                box_limits * area_limits, box_limits * area_limits, color='blue', alpha=0.6, fill=False)
            nest_circle = plt.Circle((nest_location[0], nest_location[1]), nest_radius, color='red', alpha=0.6,
                                        fill=False)
            nest_rectangle = plt.Rectangle((0, 0), nest_arealimits * area_limits, nest_arealimits * area_limits,
                                            color='red',
                                            alpha=0.6, fill=False)

            plt.cla()
            ax.add_patch(box_circle)
            ax.add_patch(box_rectangle)
            ax.add_patch(nest_circle)
            ax.add_patch(nest_rectangle)
            for i in range(individuals):
                if state_detected[iterations, i] == 0:  # Out of range
                    plt.quiver(x[i], y[i], vx[i], vy[i], color='dimgray')
                elif state_detected[iterations, i] == 1:  # Repulsion
                    plt.quiver(x[i], y[i], vx[i], vy[i], color='red')
                elif state_detected[iterations, i] == 2:  # Orientation
                    plt.quiver(x[i], y[i], vx[i], vy[i], color='dimgray')
                elif state_detected[iterations, i] == 3:  # Attraction
                    plt.quiver(x[i], y[i], vx[i], vy[i], color='dimgray')
                elif state_detected[iterations, i] == 4:  # Influence
                    plt.quiver(x[i], y[i], vx[i], vy[i], color='dimgray')
                elif state_detected[iterations, i] == 5:  # Attraction-Orientation
                    plt.quiver(x[i], y[i], vx[i], vy[i], color='dimgray')
                for o in range(objects):
                    obplt = plt.Circle((objects_location[o, 0], objects_location[o, 1]), 0.1, color='#00BB2D')
                    ax.add_patch(obplt)

            ax.set(xlim=(0, area_limits), ylim=(0, area_limits))
            ax.set_aspect('equal')
            plt.pause(0.000001)

    collectedObjects[:, 1] = iterations - collectedObjects[:, 0]
    np.save('report', report)
    np.save('collectedObjects', collectedObjects)
    
    f1 = iterations
    f2 = sum(collectedObjects[:, 3])
    f3 = individuals
    f4 = sum(collectedObjects[:, 1]) / (iterations*individuals)
    f5 = np.std(collectedObjects[:, 2])
    f6 = nestFull
    optimization_functions[0, 0] = f1
    optimization_functions[1, 0] = f2
    optimization_functions[2, 0] = f3
    optimization_functions[3, 0] = f4
    optimization_functions[4, 0] = f5
    optimization_functions[5, 0] = f6
    
    return report, ob_ep, ob_ip, collectedObjects, optimization_functions


def main_simulation():
    objects = int(input("Enter the number of objects: "))
    individuals = int(input("Enter the number of individuals: "))
    r_r = float(input("Enter the repulsion radius (m): "))
    o_r = float(input("Enter the orientation radius (m): "))
    a_r = float(input("Enter the attraction radius (m): "))

    while True:
        answer_ = input("Do you want to see the animation?? (YES/NO): ")
        if answer_.upper() == "YES":
            animation = True
            break
        elif answer_.upper() == "NO":
            animation = False
            break

    initial_time = time.time()
    [report, ob_ep, ob_ip, collectedObjects, optimization_functions] = foraging(objects, individuals, r_r, o_r, a_r, animation)
    final_time = time.time()
    print('\nRuntime: ', final_time - initial_time)
    np.save('report', report)
    np.save('collectedObjects', collectedObjects)
    np.save('optimization_functions', optimization_functions)   
    return report, ob_ep, ob_ip, collectedObjects, optimization_functions


def simulation_mean(replicas):
    #[optimization_functions_report, optimization_functions_mean] = foraging_mean(replicas)
    optimization_functions_report = np.zeros((replicas, 6))
    optimization_functions_mean = np.zeros((replicas, 6))
    animation = False
    
    objects = int(input("Enter the number of objects: "))
    individuals = int(input("Enter the number of individuals: "))
    r_r = float(input("Enter the repulsion radius (m): "))
    o_r = float(input("Enter the orientation radius (m): "))
    a_r = float(input("Enter the attraction radius (m): "))

    initial_time = time.time()
    percentage = np.linspace(0, 100, replicas + 1)
    print("Progress: ", percentage[0], "%")

    for r in range(replicas):
        [report, ob_ep, ob_ip, collectedObjects, optimization_functions] = foraging(objects, individuals, r_r, o_r, a_r, animation)
        print("Progress: ", round(percentage[r + 1], 2), "%")

        for f in range(6):
            optimization_functions_report[r, f] = optimization_functions[f, 0]
            
    for r in range(replicas):
        optimization_functions_mean[r, :] = optimization_functions_report[r, :]
        for f in range(6):
            optimization_functions_mean[r, f] = sum(optimization_functions_mean[0:r + 1, f]) / (r + 1)

    final_time = time.time()
    print('\nAverage runtime: ', (final_time - initial_time) / replicas)
    print('Runtime: ', final_time - initial_time)    

    return optimization_functions_report, optimization_functions_mean


if __name__ == '__main__':
    menu = """
    Aggregation task in robot swarms \n
    1.- Simple simulation
    2.- Multiple simulations
    3.- Exit
    """
    
    print(menu)
    while True:
        answer = int(input("Choose an option: "))
        if answer == 1:
            [report, ob_ep, ob_ip, collectedObjects, optimization_functions] = main_simulation()
            break
        elif answer == 2:
            replicas = int(input("Enter number of replicas: "))
            [optimization_functions_report, optimization_functions_mean] = simulation_mean(replicas)
            break
        elif answer == 3:
            break
        else:
            print("\nError")
            print(menu)    
    
