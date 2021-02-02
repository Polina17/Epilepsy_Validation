# Скрипт для нахождения p_value между био и био данными
# при применении косинусного ядра к исходным данным с построением графиков

import math
import matplotlib.pyplot as plt
import scipy.integrate
from numpy import inf, exp
import scipy.io


# функция для нахождения суммы значений косинусного ядра в данной точке
def kernel_cos_sum(array, x, h):   # array выборка, x - знач., в котором ищем плотность, h - сглаживающий параметр
    sum = 0
    for elem in array:
        u = (x - elem)/h
        if (math.fabs(u) <= 1):
            K = (math.pi/4) * math.cos(u*math.pi/2)
        else:
            K = 0
        sum +=K
    return sum


# функция для нахождения значений плотности вероятности в данных точках
# array - выборка
# h - сглаживающий параметр
# values - точки, в которых ищем плотность
# type - тип ядра
def density_estim(array, h, values, type):
    result = []
    for elem in values:
        if (type == 'cos'):
            result.append(kernel_cos_sum(array, elem, h) * (1/(len(array)*h)))
    return result


# функция для нахождения внутрипараметрического различия
# биологических данных для нахождения тестовой статистики
def difference_bio(bio, h_bio):
    sum = 0
    for elem in bio:
        for el in bio:
            u = (elem-el) / h_bio
            if (math.fabs(u) <= 1):
                K = (math.pi / 4) * math.cos(u * math.pi / 2)
            else:
                K = 0
            sum += K
    result = sum / (math.pow(len(bio), 2) * h_bio)
    return result


# функция для нахождения внутрипараметрического различия
# симуляционных данных для нахождения тестовой статистики
def difference_neuron(neuron, h_neuron):
    sum = 0
    for elem in neuron:
        for el in neuron:
            u = (elem - el) / h_neuron
            if (math.fabs(u) <= 1):
                K = (math.pi / 4) * math.cos(u * math.pi / 2)
            else:
                K = 0
            sum += K
    result = sum / (math.pow(len(neuron), 2) * h_neuron)
    return result


# функция для нахождения межпараметрического различия симуляционных и биологических данных
# при сглаживающем параметре для био
# для нахождения тестовой статистики
def difference_bio_neuron_h_bio(bio, neuron, h_bio):
    sum = 0
    for elem in bio:
        for el in neuron:
            u = (elem - el) / h_bio
            if (math.fabs(u) <= 1):
                K = (math.pi / 4) * math.cos(u * math.pi / 2)
            else:
                K = 0
            sum += K
    result = sum / (len(bio) * len(neuron) * h_bio)
    return result


# функция для нахождения межпараметрического различия симуляционных и биологических данных
# при сглаживающем параметре для симуляционных данных
# для нахождения тестовой статистики
def difference_bio_neuron_h_neuron(bio, neuron, h_neuron):
    sum = 0
    for elem in bio:
        for el in neuron:
            u = (elem - el) / h_neuron
            if (math.fabs(u) <= 1):
                K = (math.pi / 4) * math.cos(u * math.pi / 2)
            else:
                K = 0
            sum += K
    result = sum / (len(bio) * len(neuron) * h_neuron)
    return result


# функция для нахождения тестовой статистики при заданных сглаживающих параметрах
def statistics(bio, neuron, h_bio, h_neuron):
    T = difference_bio(bio, h_bio) + difference_neuron(neuron, h_neuron) - difference_bio_neuron_h_bio(bio, neuron, h_bio) - difference_bio_neuron_h_neuron(bio, neuron, h_neuron)
    return T


# функция для нахождения значения p_value при заданных сглаживающих параметрах
def p_value(bio, neuron, h_bio, h_neuron):
    T = statistics(bio, neuron, h_bio, h_neuron)
    f = lambda x: exp((-x ** 2)/2)
    integral = scipy.integrate.quad(f, -inf, T)
    result = 1/(math.sqrt(2*math.pi)) * integral[0]
    return result


# функция для формирования био данных данной записи по сенсорам
def bio(data, rec):
    bio_data = []
    for j in range(15):
        result = []
        for i in range(2000):
            result.append(data[i, j, rec])
        bio_data.append(result)
    return bio_data

# Получаем био данные
mat = scipy.io.loadmat('/home/polina/диплом/эпилепсия_данные_био/2011 may 03 P32 BCX rust/2011_05_03_0010.mat', squeeze_me=True)
data = mat['lfp']
print(data.shape)

mat1 = scipy.io.loadmat('/home/polina/диплом/эпилепсия_данные_био/2011 may 03 P32 BCX rust/2011_05_03_0023.mat', squeeze_me=True)
data1 = mat1['lfp']
print(data1.shape)

bio_d = bio(data, 41)
bio_data = bio(data1, 40)

density_estim_bio = []
density_estim_bio1 = []
T = []
p = []

h = [10, 10]  # h_bio, h_bio_1

#min_v = min(min(bio_d[0]), min(bio_data[0]))
#max_v = max(max(bio_d[0]), max(bio_data[0]))
min_v = -1000
max_v = 1000

print(min_v, max_v)

values = []  # формируем массив значений, в которых будем искать плотность
step = (max_v - min_v) / 2500  # число зависит от объёма выборки
for j in range(2501):
    value = min_v + (step * j)
    values.append(value)  # массив значений для рассчёта плотности bio


for i in range(0, 15):  # 0,5  5,10  10,15
    bio_d[i].sort()
    bio_data[i].sort()

    print()

    density_estim_bio.append(density_estim(bio_d[i], h[0], values, 'cos'))  # h[i-5][0]
    density_estim_bio1.append(density_estim(bio_data[i], h[1], values, 'cos'))  #h[i-5][1]

#   print(density_estim_bio)
#   print(density_estim_neuron)
    print('Сенсор ', i+1)
    T.append(statistics(bio_d[i], bio_data[i], h[0], h[1]))
    print('h_bio ', h[0], 'h_sim ', h[1])
    print('значение статистики ', T[i])

    p.append(p_value(bio_d[i], bio_data[i], h[0], h[1]))
    print('p-value ', p[i])
    print()


# Отрисовка графиков
fig, ax = plt.subplots(nrows=8, ncols=2)
plt.suptitle('Эксперимент 10. Запись 41 / Эксперимент 23. Запись 40. Исходные данные. h_bio = %.1f, h_bio_1 = %.1f' % (h[0], h[1]))
ax[0, 0].set_title('Cенсоры 1-8')
ax[0, 0].plot(values, density_estim_bio[0], 'g', label='bio', linewidth=0.8)
ax[0, 0].plot(values, density_estim_bio1[0], 'b', label='bio_1', linewidth=0.8)
ax[1, 0].plot(values, density_estim_bio[1], 'g', label='bio', linewidth=0.8)
ax[1, 0].plot(values, density_estim_bio1[1], 'b', label='bio_1', linewidth=0.8)
ax[2, 0].plot(values, density_estim_bio[2], 'g', label='bio', linewidth=0.8)
ax[2, 0].plot(values, density_estim_bio1[2], 'b', label='bio_1', linewidth=0.8)
ax[3, 0].plot(values, density_estim_bio[3], 'g', label='bio', linewidth=0.8)
ax[3, 0].plot(values, density_estim_bio1[3], 'b', label='bio_1', linewidth=0.8)
ax[4, 0].plot(values, density_estim_bio[4], 'g', label='bio', linewidth=0.8)
ax[4, 0].plot(values, density_estim_bio1[4], 'b', label='bio_1', linewidth=0.8)
ax[4, 0].set_ylabel('Значение плотности вероятности')
ax[5, 0].plot(values, density_estim_bio[5], 'g', label='bio', linewidth=0.8)
ax[5, 0].plot(values, density_estim_bio1[5], 'b', label='bio_1', linewidth=0.8)
ax[6, 0].plot(values, density_estim_bio[6], 'g', label='bio', linewidth=0.8)
ax[6, 0].plot(values, density_estim_bio1[6], 'b', label='bio_1', linewidth=0.8)
ax[7, 0].plot(values, density_estim_bio[7], 'g', label='bio', linewidth=0.8)
ax[7, 0].plot(values, density_estim_bio1[7], 'b', label='bio_1', linewidth=0.8)
ax[7, 0].set_xlabel('Потенциал локального поля, мВ')
ax[0, 1].set_title('Cенсоры 9-15')
ax[0, 1].plot(values, density_estim_bio[8], 'g', label='bio', linewidth=0.8)
ax[0, 1].plot(values, density_estim_bio1[8], 'b', label='bio_1', linewidth=0.8)
ax[1, 1].plot(values, density_estim_bio[9], 'g', label='bio', linewidth=0.8)
ax[1, 1].plot(values, density_estim_bio1[9], 'b', label='bio_1', linewidth=0.8)
ax[2, 1].plot(values, density_estim_bio[10], 'g', label='bio', linewidth=0.8)
ax[2, 1].plot(values, density_estim_bio1[10], 'b', label='bio_1', linewidth=0.8)
ax[3, 1].plot(values, density_estim_bio[11], 'g', label='bio', linewidth=0.8)
ax[3, 1].plot(values, density_estim_bio1[11], 'b', label='bio_1', linewidth=0.8)
ax[4, 1].plot(values, density_estim_bio[12], 'g', label='bio', linewidth=0.8)
ax[4, 1].plot(values, density_estim_bio1[12], 'b', label='bio_1', linewidth=0.8)
ax[5, 1].plot(values, density_estim_bio[13], 'g', label='bio', linewidth=0.8)
ax[5, 1].plot(values, density_estim_bio1[13], 'b', label='bio_1', linewidth=0.8)
ax[6, 1].plot(values, density_estim_bio[14], 'g', label='bio', linewidth=0.8)
ax[6, 1].plot(values, density_estim_bio1[14], 'b', label='bio_1',linewidth=0.8)
ax[7, 1].get_xaxis().set_visible(False)
ax[7, 1].get_yaxis().set_visible(False)

for i in range(7):
    for j in range(2):
        ax[i, j].legend(loc=2)
ax[7, 0].legend(loc=2)
plt.show()

for j in range(len(T)):
    print(T[j])
    print(p[j])