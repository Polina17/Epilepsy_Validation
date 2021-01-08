# Скрипт для вычисления p_value между симуляционнмыми и симуляционными
# данными с использованием критерия Смирнова

import math


# функция для нахождения накопленной частоты варианты value в выборке array
def cumulative_frequency(value, array):
    if value in array:
        array.sort()
        result = array.index(value) / len(array)
    else:
        array.append(value)
        array.sort()
        result = array.index(value) / (len(array) - 1)
        array.remove(value)
    return result


# Нахождение эмпирических функций распределения выборок
# для каждой варианты в выборках её накопленная частота в 1-й выборке и 2-й
def empirical_function(bio, neuron):
    function = dict()
    for i in bio:
        function[i] = [cumulative_frequency(i, bio), cumulative_frequency(i, neuron)]
    for j in neuron:
        if j not in bio:
            function[j] = [cumulative_frequency(j, bio), cumulative_frequency(j, neuron)]
    return function


# функция для вычисления значения тестовой статистики критерия Смирнова
def statistics_value(bio, neuron):
    func = empirical_function(bio, neuron)
    difference = []
    for i in func:
        difference.append(abs(func[i][0] - func[i][1]))
    D = max(difference)
    n = len(bio)
    m = len(neuron)
    result = D * math.sqrt((n*m)/(n+m))
    return result


# функция для вычисления значения p_value критерия Смирнова
def p_val(statistic):
    Kolm_func = 1 - 2 * (math.e ** (-2 * (statistic ** 2)))
    p = 1 - Kolm_func
    return p


# функция для формирования симуляционных данных по сенсорам
def sim(f):
    sim_data = []
    elem = []
    for line in f:
        elem.append(float(line))

    for i in range(15):  # 0-600, 601-1201
        sim_data.append(elem[i*601+1 : (i*601)+601 : 10])  # получили sim data по сенсорам по мс

    return sim_data

# Получаем симуляционные данные
f = open('/home/polina/диплом/эпилепсия_данные_sim/sim_data_prepare1', 'r')
file = open('/home/polina/диплом/эпилепсия_данные_sim/sim_data_prepare2', 'r')

sim_data = sim(f)
sim_data1 = sim(file)

sim_data_full = []  # обобщённые симуляционные данные
for elem in sim_data:
    for el in elem:
        sim_data_full.append(el)

sim_data_full1 = []  # обобщённые симуляционные данные 1
for elem in sim_data1:
    for el in elem:
        sim_data_full1.append(el)

for i in range(15):
    statistics = statistics_value(sim_data[i], sim_data1[i])
    print(statistics)
    p = 2 * (math.e ** (-2 * (statistics ** 2)))
    print(p)

print()
print('Обобщённые данные:')
statistics = statistics_value(sim_data_full, sim_data_full1)
print(statistics)
p = 2 * (math.e ** (-2 * (statistics ** 2)))
print(p)