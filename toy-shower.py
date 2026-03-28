#!/usr/bin/env python3
# an oversimplified (QED-like) parton shower
# for Zuoz lectures (2016) by Gavin P. Salam
from math import exp, log, pi, sqrt
from random import random

import matplotlib.pyplot as plt
import numpy as np

ptHigh = 100.0
ptCut = 1.0
alphas = 0.12
CA = 3
CF = 4 / 3
c_charge = 3

num_events = np.arange(1, 10, 1)


def get_rotation_matrix(parent_vec):
    """Returns a matrix that rotates the z-unit vector [0, 0, 1] to align with the parent_vec"""
    norm = np.linalg.norm(parent_vec)
    if norm == 0:
        return np.eye(3)

    unit_p = parent_vec / norm
    z_axis = np.array([0, 0, 1])

    if np.allclose(unit_p, z_axis):
        return np.eye(3)
    if np.allclose(unit_p, -z_axis):
        return -1 * np.eye(3)  # 180 deg flip

    v = np.cross(z_axis, unit_p)
    s = np.linalg.norm(v)
    c = np.dot(z_axis, unit_p)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    return np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s**2))


def main():
    # list of averages per event
    averages = []
    for events in num_events:
        # total number of emissions during the event
        total_emissions = 0

        for iev in range(0, events):
            emissions_in_event = event(c_charge)
            total_emissions += emissions_in_event[0]
        # calculate the average created number of particles

        average = total_emissions / events

        # append the averages to the list
        averages.append(average)

    # from the plot we see that the
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    # Plot
    ax.plot(num_events, averages)

    # Make plot nicer
    ax.grid(True)
    ax.set_xlabel(r"$N_{events}$", fontsize=12)
    ax.set_ylabel(r"$\left< N \right>_{created} $", fontsize=12)
    ax.set_title("The average number of particles created per gluon jet")

    plt.show()


def event(c_charge):
    # counter to count the gluon jets
    count = 0
    # start with maximum possible value of Sudakov
    sudakov = 1

    # part a: save the momenta
    pts_in_event = []
    z_values = []

    # part 4:

    parent_p4 = [ptHigh, 0, 0, ptHigh]  # Starting momentum parent
    all_p4 = []  # Save momenta childs

    while True:
        # scale it by a random number
        sudakov *= random()
        # deduce the corresponding pt
        pt = ptFromSudakov(sudakov, c_charge)
        # if pt falls below the cutoff, event is finished
        if pt < ptCut:
            break
        # increment counter if pt greater than cutoff.
        count += 1
        pts_in_event.append(pt)

        # Part (b): calculate z

        # define z limits: z_min = (pt**2)/(Q**2)   z_max = 1 - z_min
        z_min = (pt**2) / (ptHigh**2)
        z_max = 1 - z_min

        # create sampling list (I use linspace instead of arange)
        z_grid = np.linspace(z_min, z_max, 100)

        # calculate the splitting function
        if c_charge == CA:
            p_vals = 2 * c_charge * ((1 - z_grid) / z_grid + z_grid / (1 - z_grid))
        elif c_charge == CF:
            p_vals = c_charge * (1 + z_grid**2) / (1 - z_grid)
        else:
            raise ValueError(
                f"Color charge of {c_charge} not defined. use 3 (gluons) or 4/3 (quarks)"
            )

        # normalize
        p_norm = p_vals / np.sum(p_vals)

        # sample z-values
        z_sample = np.random.choice(z_grid, p=p_norm)
        z_values.append(z_sample)

        ## Part 4: rotation

        phi = random() * 2 * np.pi  # sample random angle between  0 and 2*pi

        E_p = parent_p4[0]
        p1_loc = np.array(
            [pt * np.cos(phi), pt * np.sin(phi), z_sample * E_p]
        )  # local momentum first child
        p2_loc = np.array(
            [-pt * np.cos(phi), pt * np.sin(phi), (1 - z_sample) * E_p]
        )  # local momentum second child

        R = get_rotation_matrix(parent_p4[1:])  # get the rotation matrix

        p1 = np.dot(R, p1_loc)  # rotated momentum p1
        p2 = np.dot(R, p2_loc)  # rotated momentum p2

        # Update the parent and samve emitted gluon
        parent_p4 = np.array([np.linalg.norm(p1), *p1])
        p2_p4 = np.array([np.linalg.norm(p2), *p2])

        all_p4.append(p2_p4)

    return count, all_p4


def ptFromSudakov(sudakovValue, c_charge):
    """Returns the pt value that solves the relation
    Sudakov = sudakovValue (for 0 < sudakovValue < 1)
    """

    norm = c_charge / pi
    if c_charge == CA:
        norm *= 2
    elif c_charge == CF:
        norm = norm
    else:
        raise ValueError(
            f"Color charge of {c_charge} not defined. use 3 (gluons) or 4/3 (quarks)"
        )
    # r = Sudakov = exp(-alphas * norm * L^2)
    # --> log(r) = -alphas * norm * L^2
    # --> L^2 = log(r)/(-alphas*norm)
    L2 = log(sudakovValue) / (-alphas * norm)
    pt = ptHigh * exp(-sqrt(L2))
    return pt


if __name__ == "__main__":
    main()
