import numpy as np
import xlrd
import xlwt
import matplotlib.pyplot as plt
import short_stroke_function

path1 = './original data/1.xlsx'
path2 = './processed data/111.txt'


def data_read(path1):
    data = xlrd.open_workbook(path1)
    table = data.sheets()[0]
    nrows = table.nrows #行数
    ncols = table.ncols #列数

    start = 0  # 开始的行
    end = nrows  # 结束的行

    rows = end - start

    date_list = []
    time_list = []
    speed_list = []
    longtitude_list = []
    latitude_list = []
    engine_speed_list = []
    torque_per_list = []
    fuel_cons_list = []
    pedal_degree_list = []
    air_fuel_ratio_list = []
    engine_load_list = []
    intake_flow_list = []

    for x in range(1, end):
        values = []
        row = table.row_values(x)
        value_split = row[0].split(' ', 1)
        date_split = value_split[0].split('/', 2)
        # date_split[2]
        time_split = value_split[1].split(':', 2)
        time_split2 = time_split[2].split('.', 1)
        # time_split2[0]
        # print(date_split[2])
        # print(time_split2[0])
        date = date_split[2]
        detal = int(date) - 18
        time = detal * 3600 * 24 + int(time_split[0]) * 3600 + int(time_split[1]) * 60 + int(time_split2[0])
        date_list.append(date)
        time_list.append(time)
        speed_list.append(row[1])
        longtitude_list.append(row[5])
        latitude_list.append(row[6])
        engine_speed_list.append(row[7])
        torque_per_list.append(row[8])
        fuel_cons_list.append(row[9])
        pedal_degree_list.append(row[10])
        air_fuel_ratio_list.append(row[11])
        engine_load_list.append(row[12])
        intake_flow_list.append(row[13])

    return time_list, speed_list, longtitude_list, latitude_list, engine_speed_list, \
           torque_per_list, fuel_cons_list, pedal_degree_list, air_fuel_ratio_list, \
           engine_load_list, intake_flow_list


times, speed, longtitude, latitude, engine_speed,\
               torque_per, fuel_cons, pedal_degree, air_fuel_ratio,\
               engine_load, intake_flow = data_read(path1)

# plt.plot(times[:1000], speed[:1000])
# plt.show()
# 处理数据
'''
index_flag = [0]
conti_time_slice = ([0])
for i in range(1, len(times)):
    diff = times[i] - times[i-1]
    if diff == 1:
        continue
    else:
        conti_time_slice += (times[index_flag[-1]:i])
        index_flag.append(i)
print(conti_time_slice)
print(index_flag)'''
# 去除GPS信号丢失数据
n = len(longtitude)
i = 0
while i < n:
    if longtitude[i] == 0:
        times.pop(i)
        speed.pop(i)
        longtitude.pop(i)
        latitude.pop(i)
        engine_speed.pop(i)
        torque_per.pop(i)
        fuel_cons.pop(i)
        pedal_degree.pop(i)
        air_fuel_ratio.pop(i)
        engine_load.pop(i)
        intake_flow.pop(i)

    n = len(longtitude)
    i += 1

# 加速度异常数据
n1 = len(speed)
j = 1
while j < n1:
    if speed[j] - speed[j-1] > 14 or speed[j] - speed[j-1] < -7.5:
        times.pop(j)
        speed.pop(j)
        longtitude.pop(j)
        latitude.pop(j)
        engine_speed.pop(j)
        torque_per.pop(j)
        fuel_cons.pop(j)
        pedal_degree.pop(j)
        air_fuel_ratio.pop(j)
        engine_load.pop(j)
        intake_flow.pop(j)

    n1 = len(speed)
    j += 1

# 给怠速状态打标签
idle_speed = []
count = 0
n_of_speed = len(speed)
k = 0
while k < n_of_speed:
    # if times[k+1] - times[k] < 10:
    if speed[k] < 10:
        idle_speed.append(1)
        count += 1
    else:
        idle_speed.append(0)
        # 怠速应持续10s以上
        if count < 40:  # 怠速最低10s
            zero_list = []
            for j in range(count):
                zero_list.append(0)
            # print(zero_list)
            # print(idle_speed[i-count, i])
            idle_speed[k-count: k] = zero_list  # 还原持续不超过10s的怠速片段
            count = 0
        # 怠速不超过180
        '''elif count > 180:
            del speed[k-count:k]
            del times[k-count:k]
            del idle_speed[k-count:k]
            k = k - count
            count = 0
            n_of_speed = len(speed)'''

    k += 1

# print(idle_speed)

# 统计怠速起始点  10~
# 提取运动学片段
short_stroke = []  # 怠速起始点的索引列表
orig_processed_data = []
temp = [0] * 14
short_stroke_feature = np.array(temp)
# print(short_stroke_feature)
if idle_speed[0] == 1:
    short_stroke.append(0)
for i in range(1, len(idle_speed)):
    if idle_speed[i-1] == 0 and idle_speed[i] == 1:
        short_stroke.append(i)
# print(short_stroke)
num_of_data = 0
short_stroke_num = 0
for i in range(1, len(short_stroke)):
    short_stroke_speed = speed[short_stroke[i-1]:short_stroke[i]]
    short_stroke_time = times[short_stroke[i-1]:short_stroke[i]]
    short_stroke_flag = idle_speed[short_stroke[i-1]:short_stroke[i]]
    print(short_stroke_speed)
    print(short_stroke_flag)
    # short_stroke_idle_speed = idle_speed[short_stroke[i-1]:short_stroke[i]]

    if len(short_stroke_speed) < 50:  # 短行程的时间小于20的去掉
        continue
    distance = sum(short_stroke_speed) / 3.6
    if distance < 30:  # 短行程的行驶距离小于10m的去掉
        continue
    else:
        short_stroke_feature_i = []
        short_stroke_num += 1
        # num_of_data += len(short_stroke_speed)
        '''if short_stroke_num == 67:
            point_data_90 = np.vstack((np.array(short_stroke_speed), np.array(short_stroke_time)))
            np.savetxt('./Representative_data/point_data_67.csv', point_data_90, delimiter=',', fmt='%.4f')
        if short_stroke_num == 79:
            point_data_171 = np.vstack((np.array(short_stroke_speed), np.array(short_stroke_time)))
            np.savetxt('./Representative_data/point_data_79.csv', point_data_171, delimiter=',', fmt='%.4f')'''
        # 处理短行程，计算特性并存储在short_stroke_feature = []中
        Pidel = short_stroke_function.idle_speed_per(short_stroke_speed)
        # 怠速不超过180
        time_of_idel = len(short_stroke_speed) * Pidel
        if int(time_of_idel) > 180:
            del short_stroke_speed[0:(int(time_of_idel) - 180)]
            del short_stroke_time[0:(int(time_of_idel) - 180)]
            del short_stroke_flag[0:(int(time_of_idel) - 180)]
            Pidel = short_stroke_function.idle_speed_per(short_stroke_speed)
        num_of_data += len(short_stroke_speed)

        orig_processed_data.extend(short_stroke_speed)

        driving_time = short_stroke_function.compute_time(short_stroke_time)
        average_speed, speed_standard_deviation = short_stroke_function.compute_average_speed(short_stroke_speed)
        average_driving_speed = short_stroke_function.compute_average_driving_speed(short_stroke_speed)
        max_speed = short_stroke_function.compute_max_speed(short_stroke_speed)
        max_acc = short_stroke_function.compute_max_acc(short_stroke_time, short_stroke_speed)
        min_decc = short_stroke_function.compute_min_decc(short_stroke_time, short_stroke_speed)
        average_acc, Pa, acc_standard_deviation = short_stroke_function.average_acc(short_stroke_time, short_stroke_speed)
        average_decc, Pd = short_stroke_function.average_decc(short_stroke_time, short_stroke_speed)
        Pc = 1 - Pa - Pd - Pidel

        '''if len(short_stroke_speed) == 168:
            point_data_1_0 = np.vstack((np.array(short_stroke_speed), np.array(short_stroke_time)))
            if 4<average_speed < 10:
                np.savetxt('./Representative_data/point_data_1_low1.csv', point_data_1_0, delimiter=',', fmt='%.4f')'''
        '''if len(short_stroke_speed) == 180:
            point_data_1_0 = np.vstack((np.array(short_stroke_speed), np.array(short_stroke_time)))
            if 4< average_speed < 10:
                np.savetxt('./Representative_data/point_data_1_low2.csv', point_data_1_0, delimiter=',', fmt='%.4f')'''
        '''if len(short_stroke_speed) == 76:
            point_data_1_0 = np.vstack((np.array(short_stroke_speed), np.array(short_stroke_time)))
            if 4< average_speed < 10:
                np.savetxt('./Representative_data/point_data_1_low3.csv', point_data_1_0, delimiter=',', fmt='%.4f')'''
        short_stroke_feature_i.append(driving_time)
        short_stroke_feature_i.append(average_speed)
        short_stroke_feature_i.append(average_driving_speed)
        short_stroke_feature_i.append(max_speed)
        short_stroke_feature_i.append(max_acc)
        short_stroke_feature_i.append(min_decc)
        short_stroke_feature_i.append(average_acc)
        short_stroke_feature_i.append(average_decc)
        short_stroke_feature_i.append(Pidel)
        short_stroke_feature_i.append(Pc)
        short_stroke_feature_i.append(Pa)
        short_stroke_feature_i.append(Pd)
        short_stroke_feature_i.append(speed_standard_deviation)
        short_stroke_feature_i.append(acc_standard_deviation)

        short_stroke_feature_sub = np.array(short_stroke_feature_i)
        # print(short_stroke_feature_i)
        # print(short_stroke_feature)
        short_stroke_feature = np.vstack((short_stroke_feature, short_stroke_feature_sub))

short_stroke_feature = np.delete(short_stroke_feature, 0, axis=0)
# save_data = np.around(short_stroke_feature, decimals=4)
# np.set_printoptions(suppress=True)
# print(save_data)
print(num_of_data)

orig_processed_data_np = np.array(orig_processed_data)
orig_processed_data_np.reshape(num_of_data, 1)
np.savetxt('./new_data2/orig_processed_data.csv', orig_processed_data_np, delimiter=',', fmt='%.4f')
# 保存特征数据14个
# np.savetxt('./new_data2/data_feature3.csv', short_stroke_feature, delimiter=',', fmt='%.4f')

'''def save(data, path):
    f = xlwt.Workbook()  # 创建工作簿
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
    [h, l] = data.shape  # h为行数，l为列数
    for i in range(h):
        for j in range(l):
            sheet1.write(i, j, data[i, j])
    f.save(path)'''

print(short_stroke_num)








