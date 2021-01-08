# Скрипт для нахождения p_value между симуляционными и симуляционными данными
# при применении нормального ядра к исходным данным с построением графиков

import math
import matplotlib.pyplot as plt
import scipy.integrate
from numpy import inf, exp
import scipy.io


# функция для нахождения суммы значений нормального ядра в данной точке
# array - выборка, x - значение, в котором ищем плотность, h - сглаживающий параметр
def kernel_normal_sum(array, x, h):
    sum = 0
    for elem in array:
        u = (x-elem)/h
        K = math.pow(math.pi * 2, -0.5) * math.pow(math.e, -0.5 * math.pow(u, 2))
        sum += K
    return sum


# функция для нахождения значений плотности вероятности в данных точках
# array - выборка
# h - сглаживающий параметр
# values - точки, в которых ищем плотность
# type - тип ядра
def density_estim(array, h, values, type):
    result = []
    for elem in values:
        if (type == 'normal'):
            result.append(kernel_normal_sum(array, elem, h) * (1/(len(array)*h)))
    return result


# функция для нахождения внутрипараметрического различия
# биологических данных для нахождения тестовой статистики
def difference_bio(bio, h_bio):
    sum = 0
    for elem in bio:
        for el in bio:
            u = (elem-el) / h_bio
            K = math.pow(math.pi * 2, -0.5) * math.pow(math.e, -0.5 * math.pow(u, 2))
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
            K = math.pow(math.pi * 2, -0.5) * math.pow(math.e, -0.5 * math.pow(u, 2))
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
            K = math.pow(math.pi * 2, -0.5) * math.pow(math.e, -0.5 * math.pow(u, 2))
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
            K = math.pow(math.pi * 2, -0.5) * math.pow(math.e, -0.5 * math.pow(u, 2))
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


# функция для формирования симуляционных данных по сенсорам
def sim(f):
    sim_data = []
    elem = []
    for line in f:
        elem.append(float(line))

    for i in range(15):  # 0-600, 601-1201, ...
        sim_data.append(elem[i*601+1 : (i*601)+601 : 10])  # получили sim data по сенсорам 0.1 мс (по мс - :10)

    return sim_data

# Получаем симуляционные данные
f = open('/home/polina/диплом/эпилепсия_данные_sim/sim_data_prepare1', 'r')
file = open('/home/polina/диплом/эпилепсия_данные_sim/sim_data_prepare2', 'r')

sim_data = sim(f)
sim_data1 = sim(file)

density_estim_sim = []
density_estim_sim1 = []
T = []
p = []

#min_v = min(min(sim_data[0]), min(sim_data1[0]))
#max_v = max(max(sim_data[0]), max(sim_data1[0]))
min_v = -0.008
max_v = 0.004
print(min_v, max_v)

values = []  # формируем массив значений, в которых будем искать плотность
step = (max_v - min_v) / 200  # число зависит от объёма выборки
#    print(step)
for j in range(201):
    value = min_v + (step * j)
    values.append(value)  # массив значений для рассчёта плотности вероятности

h = [0.0007, 0.0007]  # h_sim, h_sim_1

for i in range(0, 15):  # 0,5  5,10  10,15
    sim_data[i].sort()
    sim_data1[i].sort()

    print()

    density_estim_sim.append(density_estim(sim_data[i], h[0], values, 'normal'))  # h[i-5][0]
    density_estim_sim1.append(density_estim(sim_data1[i], h[1], values, 'normal'))

    T.append(statistics(sim_data[i], sim_data1[i], h[0], h[1]))
    print('h_bio ', h[0], 'h_sim ', h[1])
    print('значение статистики ', T[i])

    p.append(p_value(sim_data[i], sim_data1[i], h[0], h[1]))
    print('p-value ', p[i])
    print()

# Отрисовка графиков
fig, ax = plt.subplots(nrows=8, ncols=2)
plt.suptitle('Симуляция 1 / Симуляция 2 (по мс). h_sim = %.5f, h_sim_1 = %.5f' % (h[0], h[1]))
ax[0, 0].set_title('Cенсоры 1-8')
ax[0, 0].plot(values, density_estim_sim[0], 'g', label='sim', linewidth=0.8)
ax[0, 0].plot(values, density_estim_sim1[0], 'b', label='sim_1', linewidth=0.8)
ax[1, 0].plot(values, density_estim_sim[1], 'g', label='sim', linewidth=0.8)
ax[1, 0].plot(values, density_estim_sim1[1], 'b', label='sim_1', linewidth=0.8)
ax[2, 0].plot(values, density_estim_sim[2], 'g', label='sim', linewidth=0.8)
ax[2, 0].plot(values, density_estim_sim1[2], 'b', label='sim_1', linewidth=0.8)
ax[3, 0].plot(values, density_estim_sim[3], 'g', label='sim', linewidth=0.8)
ax[3, 0].plot(values, density_estim_sim1[3], 'b', label='sim_1', linewidth=0.8)
ax[4, 0].plot(values, density_estim_sim[4], 'g', label='sim', linewidth=0.8)
ax[4, 0].plot(values, density_estim_sim1[4], 'b', label='sim_1', linewidth=0.8)
ax[4, 0].set_ylabel('Значение плотности вероятности')
ax[5, 0].plot(values, density_estim_sim[5], 'g', label='sim', linewidth=0.8)
ax[5, 0].plot(values, density_estim_sim1[5], 'b', label='sim_1', linewidth=0.8)
ax[6, 0].plot(values, density_estim_sim[6], 'g', label='sim', linewidth=0.8)
ax[6, 0].plot(values, density_estim_sim1[6], 'b', label='sim_1', linewidth=0.8)
ax[7, 0].plot(values, density_estim_sim[7], 'g', label='sim', linewidth=0.8)
ax[7, 0].plot(values, density_estim_sim1[7], 'b', label='sim_1', linewidth=0.8)
ax[7, 0].set_xlabel('Потенциал локального поля, мВ')
ax[0, 1].set_title('Cенсоры 9-15')
ax[0, 1].plot(values, density_estim_sim[8], 'g', label='sim', linewidth=0.8)
ax[0, 1].plot(values, density_estim_sim1[8], 'b', label='sim_1', linewidth=0.8)
ax[1, 1].plot(values, density_estim_sim[9], 'g', label='sim', linewidth=0.8)
ax[1, 1].plot(values, density_estim_sim1[9], 'b', label='sim_1', linewidth=0.8)
ax[2, 1].plot(values, density_estim_sim[10], 'g', label='sim', linewidth=0.8)
ax[2, 1].plot(values, density_estim_sim1[10], 'b', label='sim_1', linewidth=0.8)
ax[3, 1].plot(values, density_estim_sim[11], 'g', label='sim', linewidth=0.8)
ax[3, 1].plot(values, density_estim_sim1[11], 'b', label='sim_1', linewidth=0.8)
ax[4, 1].plot(values, density_estim_sim[12], 'g', label='sim', linewidth=0.8)
ax[4, 1].plot(values, density_estim_sim1[12], 'b', label='sim_1', linewidth=0.8)
ax[5, 1].plot(values, density_estim_sim[13], 'g', label='sim', linewidth=0.8)
ax[5, 1].plot(values, density_estim_sim1[13], 'b', label='sim_1', linewidth=0.8)
ax[6, 1].plot(values, density_estim_sim[14], 'g', label='sim', linewidth=0.8)
ax[6, 1].plot(values, density_estim_sim1[14], 'b', label='sim_1',linewidth=0.8)
ax[7, 1].get_xaxis().set_visible(False)
ax[7, 1].get_yaxis().set_visible(False)

for i in range(8):
    for j in range(2):
        ax[i, j].legend(loc=2)

plt.show()

for j in range(len(T)):
    print(T[j])
    print(p[j])

