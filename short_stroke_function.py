import numpy as np


def compute_time(time):  # s
    driving_time = len(time)
    return driving_time


def compute_average_speed(speed):  # km/h
    average_speed = sum(speed) / len(speed)
    speed_np = np.array(speed)
    speed_standard_deviation = np.std(speed_np, ddof=1)
    return average_speed, speed_standard_deviation


def compute_average_driving_speed(speed):
    s = 0
    count = 0
    n = len(speed)
    for i in range(n):
        if speed[i] > 0:
            s += speed[i]
            count += 1
    average_driving_speed = s / count
    return average_driving_speed


def compute_max_speed(speed):  # km/h
    return max(speed)


def compute_max_acc(time, speed):  # m/s2
    diff = [(speed[i+1] - speed[i])/(time[i+1] - time[i]) for i in range(len(speed) - 1)]
    return max(diff) / 3.6


def compute_min_decc(time, speed):  # m/s2
    diff = [(speed[i+1] - speed[i])/(time[i+1] - time[i]) for i in range(len(speed) - 1)]
    return min(diff) / 3.6


def average_acc(time, speed):
    diff = [(speed[i+1] - speed[i])/(time[i+1] - time[i]) for i in range(len(speed) - 1)]
    acc = 0
    count = 0
    for i in range(len(diff)):
        if diff[i] / 3.6 > 0.1 and speed[i+1] >= 10:
            acc += diff[i]
            count += 1
    if count == 0:
        avr_acc = 0
    else:
        avr_acc = acc / count
    Pa = count / len(speed)
    acc_np = np.array(diff)
    acc_standard_deviation = np.std(acc_np, ddof=1)
    return avr_acc, Pa, acc_standard_deviation


def average_decc(time, speed):
    diff = [(speed[i+1] - speed[i])/(time[i+1] - time[i]) for i in range(len(speed) - 1)]
    decc = 0
    count = 0
    for i in range(len(diff)):
        if diff[i] / 3.6 < -0.1 and speed[i+1] >= 10:
            decc += diff[i]
            count += 1
    if count == 0:
        avr_decc = 0
    else:
        avr_decc = decc / count
    Pd = count / len(speed)
    return avr_decc, Pd


def idle_speed_per(speed):
    num = 0
    for i in range(len(speed)):
        if speed[i] < 10:
            num += 1
    Pidle = num / len(speed)
    return Pidle
