# Скрипт для обработки исходных симуляционных данных
# и построения графиков по сенсорам

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

outdir = os.path.abspath('tests/937_tW')

points= [-650,-300,-100,150,600] #5
fig = go.Figure()
df = pd.read_csv('/home/polina/диплом/эпилепсия_данные_sim/extr_all4.csv')

Time = df['t'].unique()
data =  [[] for i in range(15)]
def dist(x2,y2,z2,z1, x1=100,y1=100):
    r=np.sqrt((10**(-6*2)) * ((x1-x2)**2+(y1-y2)**2+(z1-z2)**2))
    return r

THLM=[]
for step in Time:  # step - каждый момент времени в симуляции, всего 600
    filter_t = df['t'] == step
    x=df.loc[filter_t]['x'].values.tolist()  # все x точек в данный момент времени
    # print(x)
    y=df.loc[filter_t]['y'].values.tolist()
    z=df.loc[filter_t]['z'].values.tolist()
    v=df.loc[filter_t]['v'].values.tolist()   # voltage каждой точки в данный момент времени
    id = df.loc[filter_t]['id']

    v_dist = [[] for i in range(15)]

    lenn=15
    lenV=len(v)
    thlm =[]
    for i in range(len(v)):
        if z[i] > 1000:
            thlm.append(v[i] / dist(x[i], y[i], z[i], 1150))
    for list in range(lenn):
        for i in range(lenV):
            if 850 >= z[i] >= -850:
                v_dist[list].append(v[i] /dist(x[i],y[i],z[i],-850+(list*113)))

    for i in range(lenn):
        data[i].append(sum(v_dist[i]))
    THLM.append(sum(thlm))

for elem in data:   # запись в файл
    print(elem)
    for el in elem:
        f = open('/home/polina/диплом/эпилепсия_данные_sim/sim_data_prepare1', 'a+')
        f.write(str(el) + '\n')
        f.close()
    print(len(elem))

# data[i] - каждый участок (сенсор), содержит 601 данных (в каждый момент времени)

time = [0.1*i for i in range(0, 601)]

# Отрисовка графиков симуляционных данных
fig = plt.figure(1) #первое окно с графиками
for i in range(15):
    plt.subplot(15,1,i+1)
    plt.plot(time, data[i], linewidth=1.0)
    if i == 0:
        plt.title('Симуляция 1')
    if i == 7:
        plt.ylabel('Потенциал локального поля, мВ')
    if i == 14:
        plt.xlabel('Время, мс')
plt.show()