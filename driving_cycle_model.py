import numpy as np
import xlrd
import xlwt
import matplotlib.pyplot as plt



path1 = './Representative_data/11.xlsx'

def data_read(path1):
    data = xlrd.open_workbook(path1)
    table = data.sheets()[0]
    nrows = table.nrows #行数
    ncols = table.ncols #列数

    point_data = []

    for i in range(ncols):
        value = table.col_values(i)
        point_data.append(value)
    point_data_ = np.array(point_data)
    point_data_.reshape(ncols, 1)

    return point_data_


def compute_average_speed_(speed):  # km/h
    speed_ = np.array(speed)
    average_speed = sum(speed_) / len(speed)
    return average_speed


def compute_average_driving_speed_(speed):
    s = 0
    count = 0
    n = len(speed)
    for i in range(n):
        if speed[i] > 0:
            s += speed[i]
            count += 1
    average_driving_speed = s / count
    return average_driving_speed

def average_acc_(speed):
    diff = [speed[i+1] - speed[i] for i in range(len(speed) - 1)]
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
    return Pa


def average_decc_(speed):
    diff = [speed[i+1] - speed[i] for i in range(len(speed) - 1)]
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
    return Pd


def idle_speed_per_(speed):
    num = 0
    for i in range(len(speed)):
        if speed[i] < 10:
            num += 1
    Pidle = num / len(speed)
    return Pidle


point_data = data_read(path1)
#print(data_read(path1))
time = len(point_data)
average_speed = compute_average_speed_(point_data)
average_driving_speed = compute_average_driving_speed_(point_data)
max_speed = max(point_data)
Pa = average_acc_(point_data)
Pd = average_decc_(point_data)
Pi = idle_speed_per_(point_data)
Pc = 1 - Pa - Pd - Pi
print(time, average_speed, average_driving_speed, max_speed, Pi, Pc, Pa, Pd)

plt.figure()
plt.plot(point_data)
plt.title('Vehicle driving cycle curve')

plt.figure()
plt.plot(point_data[0:320])
plt.title('low speed section')

plt.figure()
plt.plot(point_data[320:830])
plt.title('medium speed section')

plt.figure()
plt.plot(point_data[830:])
plt.title('high speed section')

plt.show()



# print(data_read(path1))