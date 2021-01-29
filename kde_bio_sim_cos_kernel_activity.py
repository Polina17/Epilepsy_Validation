# Скрипт для нахождения p_value между био и симуляционными данными
# при применении косинусного ядра к фильрованным нормализованным данным
# с построением графиков и сравнением различных фильтров

import math
import matplotlib.pyplot as plt
import scipy.integrate
from numpy import inf, exp
import scipy.io


# функция для нахождения среднего значения выборки
def mean(array):
    sum = 0
    for elem in array:
        sum += elem
    result = sum/len(array)
    return result


# функция для нахождения среднего квадратического отклонения выборки
def standart_deviation(array):
    sqr_sum = 0
    for elem in array:
        sqr_sum += math.pow(elem, 2)
    result = sqr_sum/len(array) - math.pow(mean(array), 2)
    return math.sqrt(result)


# функция для нормализации данных (минус среднее / стандартное отклонение)
def normal(array):   # нормализуем исходный массив
    result = []
    mean_value = mean(array)
    standart_dev_value = standart_deviation(array)
    for elem in array:
        e = (elem-mean_value)/standart_dev_value
        result.append(e)
    return result


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
            result.append(kernel_cos_sum(array, elem, h) * (1 / (len(array) * h)))
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


# Функция для нахождения массива значений, в которых будем искать значение плотности
def values(bio_test, sim_test):
    min_v = min(min(bio_test), min(sim_test))
    max_v = max(max(bio_test), max(sim_test))

    values = []  # формируем массив значений, в которых будем искать плотность
    step = (max_v - min_v) / 3000  # число зависит от объёма выборки

    for j in range(3001):
        value = min_v + (step * j)
        values.append(value)  # массив значений для рассчёта плотности bio

    return values

sim_data = []
f = open('/home/polina/диплом/эпилепсия_данные_sim/sim_data_prepare2', 'r')

elem = []
for line in f:
    elem.append(float(line))

# берём каждое 10-е значение, т.к. био данные - по мс, сим - по 0.1 мс
for i in range(15):  # 0-600, 601-1201
    sim_data.append(elem[i*601+1 : (i*601)+601 : 10])  # получили sim data по сенсорам
print('длина sim ', len(sim_data))

sim_data_full = []
for elem in sim_data:
    for el in elem:
        sim_data_full.append(el)

# получаем данные био
mat = scipy.io.loadmat('/home/polina/диплом/эпилепсия_данные_био/2011 may 03 P32 BCX rust/2011_05_03_0021.mat', squeeze_me=True)
data = mat['lfp']
print(data.shape)

bio_data = []
for j in range(15):
    result = []
    for i in range(2000):
        result.append(data[i, j, 30])  # № записи
    bio_data.append(result)

bio_data_full = []
for elem in bio_data:
    for el in elem:
        bio_data_full.append(el)

density_estim_bio_cos_0_1 = []
density_estim_sim_cos_0_1 = []
density_estim_bio_cos_1 = []
density_estim_sim_cos_1 = []

p_cos_0_1 = []
p_cos_1 = []

values_array_0_1 = []
values_array_1 = []

h_0_1 = [0.1, 0.2]  # h_bio, h_sim
h_1 = [0.07, 0.15]

for i in range(10, 15):  # 5,10  10, 15
    bio_data_sort = sorted(normal(bio_data[i]), key=float)  # Сортируем нормализованные данные
    sim_data_sort = sorted(normal(sim_data[i]), key=float)

    print(bio_data_sort)
    print(sim_data_sort)

    bio_test_0_1 = []
    bio_test_1 = []
    for elem in bio_data_sort:
        if elem>1 or elem <-1:   # Фильтруем
            bio_test_1.append(elem)
        if elem>0.1 or elem<-0.1:
            bio_test_0_1.append(elem)

    sim_test_0_1 = []
    sim_test_1 = []
    for elem in sim_data_sort:
        if elem > 1 or elem < -1:
            sim_test_1.append(elem)
        if elem > 0.1 or elem < -0.1:
            sim_test_0_1.append(elem)

    val_0_1 = values(bio_test_0_1, sim_test_0_1)
    val_1 = values(bio_test_1, sim_test_1)
    values_array_0_1.append(val_0_1)
    values_array_1.append(val_1)

    density_estim_bio_cos_0_1.append(density_estim(bio_test_0_1, h_0_1[0], val_0_1, 'cos')) # h[i-5][0], h[i-10][0]
    density_estim_bio_cos_1.append(density_estim(bio_test_1, h_1[0], val_1, 'cos'))
    density_estim_sim_cos_0_1.append(density_estim(sim_test_0_1, h_0_1[1], val_0_1, 'cos'))
    density_estim_sim_cos_1.append(density_estim(sim_test_1, h_1[1], val_1, 'cos'))

    p_cos_0_1.append(p_value(bio_test_0_1, sim_test_0_1, h_0_1[0], h_0_1[1]))
    p_cos_1.append(p_value(bio_test_1, sim_test_1, h_1[0], h_1[1]))
    print('p-value ', p_cos_0_1, p_cos_1)

# Отрисовка графиков
fig, ax = plt.subplots(nrows=5, ncols=2)
plt.suptitle('Симуляция 2 / Эксперимент 21. Запись 30. Сенсоры 11-15. Нормализованные данные, косинусное ядро')
ax[0, 0].set_title('Фильтрация [-0.1; 0.1]. h_bio = %.2f, h_sim = %.2f' % (h_0_1[0], h_0_1[1]))
ax[0, 0].plot(values_array_0_1[0], density_estim_bio_cos_0_1[0], 'g', label='bio', linewidth=0.8)
ax[0, 0].plot(values_array_0_1[0], density_estim_sim_cos_0_1[0], 'b', label='sim, p=%.5f' % p_cos_0_1[0], linewidth=0.8)
ax[1, 0].plot(values_array_0_1[1], density_estim_bio_cos_0_1[1], 'g', label='bio', linewidth=0.8)
ax[1, 0].plot(values_array_0_1[1], density_estim_sim_cos_0_1[1], 'b', label='sim, p=%.5f' % p_cos_0_1[1], linewidth=0.8)
ax[2, 0].plot(values_array_0_1[2], density_estim_bio_cos_0_1[2], 'g', label='bio', linewidth=0.8)
ax[2, 0].plot(values_array_0_1[2], density_estim_sim_cos_0_1[2], 'b', label='sim, p=%.5f' % p_cos_0_1[2], linewidth=0.8)
ax[3, 0].set_ylabel('Значение плотности вероятности')
ax[3, 0].plot(values_array_0_1[3], density_estim_bio_cos_0_1[3], 'g', label='bio', linewidth=0.8)
ax[3, 0].plot(values_array_0_1[3], density_estim_sim_cos_0_1[3], 'b', label='sim, p=%.5f' % p_cos_0_1[3], linewidth=0.8)
ax[4, 0].plot(values_array_0_1[4], density_estim_bio_cos_0_1[4], 'g', label='bio', linewidth=0.8)
ax[4, 0].plot(values_array_0_1[4], density_estim_sim_cos_0_1[4], 'b', label='sim, p=%.5f' % p_cos_0_1[4], linewidth=0.8)
ax[4, 0].set_xlabel('Потенциал локального поля, мВ')

ax[0, 1].set_title('Фильтрация [-1; 1]. h_bio = %.2f, h_sim = %.2f' % (h_1[0], h_1[1]))
ax[0, 1].plot(values_array_1[0], density_estim_bio_cos_1[0], 'g', label='bio', linewidth=0.8)
ax[0, 1].plot(values_array_1[0], density_estim_sim_cos_1[0], 'b', label='sim, p=%.5f' % p_cos_1[0], linewidth=0.8)
ax[1, 1].plot(values_array_1[1], density_estim_bio_cos_1[1], 'g', label='bio', linewidth=0.8)
ax[1, 1].plot(values_array_1[1], density_estim_sim_cos_1[1], 'b', label='sim, p=%.5f' % p_cos_1[1], linewidth=0.8)
ax[2, 1].plot(values_array_1[2], density_estim_bio_cos_1[2], 'g', label='bio', linewidth=0.8)
ax[2, 1].plot(values_array_1[2], density_estim_sim_cos_1[2], 'b', label='sim, p=%.5f' % p_cos_1[2], linewidth=0.8)
ax[3, 1].plot(values_array_1[3], density_estim_bio_cos_1[3], 'g', label='bio', linewidth=0.8)
ax[3, 1].plot(values_array_1[3], density_estim_sim_cos_1[3], 'b', label='sim, p=%.5f' % p_cos_1[3], linewidth=0.8)
ax[4, 1].plot(values_array_1[4], density_estim_bio_cos_1[4], 'g', label='bio', linewidth=0.8)
ax[4, 1].plot(values_array_1[4], density_estim_sim_cos_1[4], 'b', label='sim, p=%.5f' % p_cos_1[4], linewidth=0.8)

for i in range(5):
    for j in range(2):
        ax[i, j].legend(loc=2, fontsize=9)
    #ax[i, 1].legend(loc=2, fontsize=8)
#ax[0, 1].legend(loc=1, fontsize=9)
#ax[1, 1].legend(loc=1, fontsize=9)
#ax[1, 1].legend(loc=1, fontsize=9)
#ax[4, 1].legend(loc=1, fontsize=9)
plt.show()