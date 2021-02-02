# Скрипт для нахождения p_value между био и симуляционными данными
# при применении косинусного ядра к нормализованным и нормализованным фильтрованным
# данным с построением графиков

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


# Функция для получения обобщённых данных
def full(sim_data):
    result = []
    for elem in sim_data:
        for el in elem:
            result.append(el)
    return result


# Функция для нахождения массива значений, в которых будем искать значение плотности
def values_f(bio_test, sim_test):
    min_v = min(min(bio_test), min(sim_test))
    max_v = max(max(bio_test), max(sim_test))

    values = []  # формируем массив значений, в которых будем искать плотность
    step = (max_v - min_v) / 3000  # число зависит от объёма выборки

    for j in range(3001):
        value = min_v + (step * j)
        values.append(value)  # массив значений для рассчёта плотности bio

    return values


# Функция для фильтрации данных
def test_f(sim_data, filter):
    sim_data_sort = sorted(sim_data, key=float)
    sim_test = []

    for elem in sim_data_sort:
        if elem > filter or elem < filter*(-1):  # Фильтруем
            sim_test.append(elem)

    return sim_test


# Получаем симуляционные данные
sim_data = []
f = open('/home/polina/диплом/эпилепсия_данные_sim/sim_data_prepare2', 'r')

elem = []
for line in f:
    elem.append(float(line))

# берём каждое 10-е значение, т.к. био данные - по мс, сим - по 0.1 мс
for i in range(15):  # 0-600, 601-1201
    sim_data.append(elem[i*601+1 : (i*601)+601 : 10])  # получили sim data по сенсорам
print('длина sim ', len(sim_data))

# получаем данные био
mat = scipy.io.loadmat('/home/polina/диплом/эпилепсия_данные_био/2011 may 03 P32 BCX rust/2011_05_03_0023.mat', squeeze_me=True)
data = mat['lfp']
print(data.shape)

bio_data = []
for j in range(15):
    result = []
    for i in range(2000):
        result.append(data[i, j, 40])  # № записи
    bio_data.append(result)

print('длина bio ', len(bio_data))  # получили lfp био по сенсорам - 15 массивов по 2000 данных

print()

fig = plt.figure(1) #первое окно с графиками

#density_estim_bio_cos = []  # Для отрисовки по сенсорам
#density_estim_sim_cos = []
#density_estim_bio_cos_0_1 = []
#density_estim_sim_cos_0_1 = []
#density_estim_bio_cos_1 = []
#density_estim_sim_cos_1 = []

T_cos = []
T_cos_0_1 = []
T_cos_1 = []

p_cos = []
p_cos_0_1 = []
p_cos_1 = []

#values_array = []
#values_array_0_1 = []
#values_array_1 = []

h = [0.1, 0.1]  # h_bio, h_sim
h_0_1 = [0.1, 0.2]
h_1 = [0.07, 0.15]

for i in range(0, 15):  # 0,5  5,10  10,15
    bio_test = normal(bio_data[i])
    sim_test = normal(sim_data[i])

    bio_test_0_1 = test_f(normal(bio_data[i]), 0.1)
    sim_test_0_1 = test_f(normal(sim_data[i]), 0.1)

    bio_test_1 = test_f(normal(bio_data[i]), 1)
    sim_test_1 = test_f(normal(sim_data[i]), 1)

#    values = values_f(bio_test, sim_test)  # Для отрисовки по сенсорам
#    values_array.append(values)
#    values_0_1 = values_f(bio_test_0_1, sim_test_0_1)
#    values_array_0_1.append(values_0_1)
#    values_1 = values_f(bio_test_1, sim_test_1)
#    values_array_1.append(values_1)

#    density_estim_bio_cos.append(density_estim(bio_test, h[0], values, 'cos'))
#    density_estim_sim_cos.append(density_estim(sim_test, h[1], values, 'cos'))
#    density_estim_bio_cos_0_1.append(density_estim(bio_test_0_1, h_0_1[0], values_0_1, 'cos'))  # h[i-5][0], h[i-10][0]
#    density_estim_sim_cos_0_1.append(density_estim(sim_test_0_1, h_0_1[1], values_0_1, 'cos'))
#    density_estim_bio_cos_1.append(density_estim(bio_test_1, h_1[0], values_1, 'cos'))
#    density_estim_sim_cos_1.append(density_estim(sim_test_1, h_1[1], values_1, 'cos'))

    T_cos.append(statistics(bio_test, sim_test, h[0], h[1]))
    T_cos_0_1.append(statistics(bio_test, sim_test, h_0_1[0], h_0_1[1]))
    T_cos_1.append(statistics(bio_test, sim_test, h_1[0], h_1[1]))

    p_cos.append(p_value(bio_test, sim_test, h[0], h[1]))
    p_cos_0_1.append(p_value(bio_test_0_1, sim_test_0_1, h_0_1[0], h_0_1[1]))
    p_cos_1.append(p_value(bio_test_1, sim_test_1, h_1[0], h_1[1]))
    print('Сенсор ', i+1)
    print('p-value ', p_cos, p_cos_0_1, p_cos_1)

bio_full = normal(full(bio_data))
sim_full = normal(full(sim_data))

bio_full_0_1 = test_f(normal(full(bio_data)), 0.1)
sim_full_0_1 = test_f(normal(full(sim_data)), 0.1)

bio_full_1 = test_f(normal(full(bio_data)), 1)
sim_full_1 = test_f(normal(full(sim_data)), 1)

values = values_f(bio_full, sim_full)
values_0_1 = values_f(bio_full_0_1, sim_full_0_1)
values_1 = values_f(bio_full_1, sim_full_1)

density_estim_bio_full = density_estim(bio_full, h[0], values, 'cos')
density_estim_sim_full = density_estim(sim_full, h[1], values, 'cos')
density_estim_bio_full_0_1 = density_estim(bio_full_0_1, h_0_1[0], values_0_1, 'cos')
density_estim_sim_full_0_1 = density_estim(sim_full_0_1, h_0_1[1], values_0_1, 'cos')
density_estim_bio_full_1 = density_estim(bio_full_1, h_1[0], values_1, 'cos')
density_estim_sim_full_1 = density_estim(sim_full_1, h_1[1], values_1, 'cos')

T_full = statistics(bio_full, sim_full, h[0], h[1])
T_full_0_1 = statistics(bio_full_0_1, sim_full_0_1, h_0_1[0], h_0_1[1])
T_full_1 = statistics(bio_full_1, sim_full_1, h_1[0], h_1[1])

p_full = p_value(bio_full, sim_full, h[0], h[1])
p_full_0_1 = p_value(bio_full_0_1, sim_full_0_1, h_0_1[0], h_0_1[1])
p_full_1 = p_value(bio_full_1, sim_full_1, h_1[0], h_1[1])

print('По исходным нормализованным данным')
for i in range(15):
    print(T_cos[i])
    print(p_cos[i])

print()
print('По нормализованным фильтрованным [-0.1; 0.1] данным')
for i in range(15):
    print(T_cos_0_1[i])
    print(p_cos_0_1[i])

print()
print('По нормализованным фильтрованным [-1; 1] данным')
for i in range(15):
    print(T_cos_1[i])
    print(p_cos_1[i])

print()
print('По обобщённым нормализованным данным')
print(T_full)
print(p_full)

print()
print('По обобщённым нормализованным фильтрованным [-0.1; 0.1] данным')
print(T_full_0_1)
print(p_full_0_1)

print()
print('По обобщённым нормализованным фильтрованным [-1; 1] данным')
print(T_full_1)
print(p_full_1)

plt.subplot(3, 1, 1)
plt.title('Симуляция 2 / Эксперимент 23. Запись 40. Обобщённые нормализованные данные, cos ядро')
plt.plot(values, density_estim_bio_full, 'g', label='bio. h_bio = %.2f, h_sim = %.2f' % (h[0], h[1]), linewidth=0.8)
plt.plot(values, density_estim_sim_full, 'b', label='sim, p=%.5f' % p_full, linewidth=0.8)
plt.xticks([])
plt.legend(loc=2, fontsize=9)

plt.subplot(3, 1, 2)
plt.title('Обобщённые нормализованные данные. Фильтрация [-0.1; 0.1]')  # pad=0
plt.plot(values_0_1, density_estim_bio_full_0_1, 'g', label='bio. h_bio = %.2f, h_sim = %.2f' % (h_0_1[0], h_0_1[1]), linewidth=0.8)
plt.plot(values_0_1, density_estim_sim_full_0_1, 'b', label='sim, p=%.5f' % p_full_0_1, linewidth=0.8)
plt.ylabel('Значение плотности вероятности')
plt.xticks([])  # Сделать ось x невидимой
plt.legend(loc=2, fontsize=9)

plt.subplot(3, 1, 3)
plt.title('Обобщённые нормализованные данные. Фильтрация [-1; 1]')
plt.plot(values_1, density_estim_bio_full_1, 'g', label='bio. h_bio = %.2f, h_sim = %.2f' % (h_1[0], h_1[1]), linewidth=0.8)
plt.plot(values_1, density_estim_sim_full_1, 'b', label='sim, p=%.5f' % p_full_1, linewidth=0.8)
plt.xlabel('Потенциал локального поля, мВ')
plt.legend(loc=2, fontsize=9)

plt.show()
