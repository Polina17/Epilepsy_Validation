#Скрипт для сравнения валидации биологических и симуляционных данных с симуляцией 1 и симуляцией 2
# при использовании симуляции 1 и симуляции 2
# при применении нормального ядра к нормализованным данным
# с одинаковыми сглаживающими параметрами

# Валидация биоданных с симуляцией 1 и симуляцией 2 даёт схожие результаты

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

    for i in range(15):  # 0-600, 601-1201
        sim_data.append(elem[i*601+1 : (i*601)+601 : 10])  # получили sim data по сенсорам мс 0)

    return sim_data


# функция для формирования биоданных заданной записи по сенсорам
def bio(data, rec):
    bio_data = []
    for j in range(15):
        result = []
        for i in range(2000):
            result.append(data[i, j, rec])
        bio_data.append(result)
    return bio_data

# Получаем симуляционные данные
f = open('/home/polina/диплом/эпилепсия_данные_sim/sim_data_prepare1', 'r')
file = open('/home/polina/диплом/эпилепсия_данные_sim/sim_data_prepare2', 'r')

sim_data = sim(f)
sim_data1 = sim(file)

# Получаем биологические данные
mat = scipy.io.loadmat('/home/polina/диплом/эпилепсия_данные_био/2011 may 03 P32 BCX rust/2011_05_03_0021.mat', squeeze_me=True)
data = mat['lfp']
print(data.shape)

bio_data = bio(data, 30)  # номер записи

values_array = []
density_estim_bio = []
density_estim_sim = []
density_estim_sim1 = []
T_sim = []
T_sim1 = []
p_sim = []
p_sim1 = []

h = [0.1, 0.15]  # h_bio, h_sim

for i in range(2, 15, 3):  # сенсоры 3, 6, 9, 12, 15
    normal(bio_data[i]).sort()
    normal(sim_data[i]).sort()

    min_v = min(min(normal(bio_data[i])), min(normal(sim_data[i])))
    max_v = max(max(normal(bio_data[i])), max(normal(sim_data[i])))

    print(min_v, max_v)

    values = []  # формируем массив значений, в которых будем искать плотность
    step = (max_v - min_v) / 3000  # число зависит от объёма выборки
    for j in range(3001):
        value = min_v + (step * j)
        values.append(value)  # массив значений для рассчёта плотности bio

    values_array.append(values)
    print()

    density_estim_bio.append(density_estim(normal(bio_data[i]), h[0], values, 'normal'))  # h[i-5][0], h[i-10][0]
    density_estim_sim.append(density_estim(normal(sim_data[i]), h[1], values, 'normal'))
    density_estim_sim1.append(density_estim(normal(sim_data1[i]), h[1], values, 'normal'))

    T_sim.append(statistics(normal(bio_data[i]), normal(sim_data[i]), h[0], h[1]))
    T_sim1.append(statistics(normal(bio_data[i]), normal(sim_data1[i]), h[0], h[1]))
    print('h_bio ', h[0], 'h_sim ', h[1])
    print('значение статистики c sim ', T_sim)
    print('значение статистики c sim1 ', T_sim1)

    p_sim.append(p_value(normal(bio_data[i]), normal(sim_data[i]), h[0], h[1]))
    p_sim1.append(p_value(normal(bio_data[i]), normal(sim_data1[i]), h[0], h[1]))
    print('p-value с sim ', p_sim)
    print('p-value с sim1 ', p_sim1)
    print()

# Отрисовка графиков
fig, ax = plt.subplots(nrows=5, ncols=2)
plt.suptitle('Эксперимент 21. Запись 30. Сенсоры сенсоры 3, 6, 9, 12, 15. h_bio = %.2f, h_sim = %.2f' % (h[0], h[1]))
ax[0, 0].set_title('Сравнение с симуляцией 1')
ax[0, 0].plot(values_array[0], density_estim_bio[0], 'g', label='bio', linewidth=0.8)
ax[0, 0].plot(values_array[0], density_estim_sim[0], 'b', label='sim, p=%.5f' % p_sim[0], linewidth=0.8)
ax[1, 0].plot(values_array[1], density_estim_bio[1], 'g', label='bio', linewidth=0.8)
ax[1, 0].plot(values_array[1], density_estim_sim[1], 'b', label='sim, p=%.5f' % p_sim[1], linewidth=0.8)
ax[2, 0].plot(values_array[2], density_estim_bio[2], 'g', label='bio', linewidth=0.8)
ax[2, 0].plot(values_array[2], density_estim_sim[2], 'b', label='sim, p=%.5f' % p_sim[2], linewidth=0.8)
ax[3, 0].plot(values_array[3], density_estim_bio[3], 'g', label='bio', linewidth=0.8)
ax[3, 0].plot(values_array[3], density_estim_sim[3], 'b', label='sim, p=%.5f' % p_sim[3], linewidth=0.8)
ax[3, 0].set_ylabel('Значение плотности вероятности')
ax[4, 0].plot(values_array[4], density_estim_bio[4], 'g', label='bio', linewidth=0.8)
ax[4, 0].plot(values_array[4], density_estim_sim[4], 'b', label='sim, p=%.5f' % p_sim[4], linewidth=0.8)
ax[4, 0].set_xlabel('Потенциал локального поля, мВ')
ax[0, 1].set_title('Сравнение с симуляцией 2')
ax[0, 1].plot(values_array[0], density_estim_bio[0], 'g', label='bio', linewidth=0.8)
ax[0, 1].plot(values_array[0], density_estim_sim1[0], 'b', label='sim_1, p=%.5f' % p_sim1[0], linewidth=0.8)
ax[1, 1].plot(values_array[1], density_estim_bio[1], 'g', label='bio', linewidth=0.8)
ax[1, 1].plot(values_array[1], density_estim_sim1[1], 'b', label='sim_1, p=%.5f' % p_sim1[1], linewidth=0.8)
ax[2, 1].plot(values_array[2], density_estim_bio[2], 'g', label='bio', linewidth=0.8)
ax[2, 1].plot(values_array[2], density_estim_sim1[2], 'b', label='sim_1, p=%.5f' % p_sim1[2], linewidth=0.8)
ax[3, 1].plot(values_array[3], density_estim_bio[3], 'g', label='bio', linewidth=0.8)
ax[3, 1].plot(values_array[3], density_estim_sim1[3], 'b', label='sim_1, p=%.5f' % p_sim1[3], linewidth=0.8)
ax[4, 1].plot(values_array[4], density_estim_bio[4], 'g', label='bio', linewidth=0.8)
ax[4, 1].plot(values_array[4], density_estim_sim1[4], 'b', label='sim_1, p=%.5f' % p_sim1[4], linewidth=0.8)

for i in range(5):
    for j in range(2):
        ax[i, j].legend(loc=2)

plt.show()
