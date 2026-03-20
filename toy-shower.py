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

num_events = np.arange(1, 4000, 20)


def main():
    # list of averages per event
    averages = []

    for events in num_events:
        # total number of emissions during the event
        total_emissions = 0

        for iev in range(0, events):
            emissions_in_event = event()
            total_emissions += emissions_in_event

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

    # Print the average number of created particles by averaging from the converged regime ~400 events
    average_jets = np.mean(averages[int(400 / 50) :])
    print(f"Average jets per event: {average_jets:.2f}")


def event():
    # counter to count the gluon jets
    count = 0
    # start with maximum possible value of Sudakov
    sudakov = 1
    while True:
        # scale it by a random number
        sudakov *= random()
        # deduce the corresponding pt
        pt = ptFromSudakov(sudakov)
        # if pt falls below the cutoff, event is finished
        if pt < ptCut:
            break
        # increment counter if pt greater than cutoff.
        count += 1
    return count


def ptFromSudakov(sudakovValue):
    """Returns the pt value that solves the relation
    Sudakov = sudakovValue (for 0 < sudakovValue < 1)
    """
    norm = CF / pi  # CF = 4/3
    # r = Sudakov = exp(-alphas * norm * L^2)
    # --> log(r) = -alphas * norm * L^2
    # --> L^2 = log(r)/(-alphas*norm)
    L2 = log(sudakovValue) / (-alphas * norm)
    pt = ptHigh * exp(-sqrt(L2))
    return pt


main()
