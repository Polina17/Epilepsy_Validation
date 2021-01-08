# Скрипт для отрисовки биоданных по сенсорам

import scipy.io
import matplotlib.pyplot as plt


mat = scipy.io.loadmat('/home/polina/диплом/эпилепсия_данные_био/2011 may 03 P32 BCX rust/2011_05_03_0021.mat', squeeze_me=True)

print(mat.keys())
data = mat['lfp']
print(data.shape)

values = [i+1 for i in range(2000)] # миллисекунды, по оси x
print(values)

lfp = []     # lfp, j - сенсоры, i - данные
for j in range(15):   # 16
    result = []
    for i in range(2000):
        result.append(data[i,j,30])  # № записи
    lfp.append(result)

count = 1
for elem in lfp:
    print(count, elem)
    print(min(elem), max(elem))
    count += 1

print(len(lfp))


fig = plt.figure(1) #первое окно с графиками
for i in range(15):  # 16
    plt.subplot(15,1,i+1)  # 16
    plt.plot(values, lfp[i], linewidth=1.0)
    if i == 0:
        plt.title('День 1. Эксперимент 21. Запись 30')
    if i == 7:
        plt.ylabel('Потенциал локального поля, мВ')
    if i == 14:  # 15
        plt.xlabel('Время, мс')
plt.show()


