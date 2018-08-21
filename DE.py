# -*- coding: utf-8 -*- 
import numpy as np 
import random 
from math import cos 
 
pop_size = 500  # 种群数量 
G = 3000  # 迭代次数 
D = 2   # 问题的维数 
F = 0.5  # 变异算子 
CR = 0.5  # 交换概率 
fitness = np.zeros(pop_size)  # 适应度 
fitness_best=[]
fitness_mean = [] #本代平均适应度 
g=0
P_best = []  # 每次迭代的所获得的最优解 
P = []   #原始种群
V = np.zeros((pop_size,D),dtype=np.float64)  #差分进化后得到的临时种群
U = np.zeros((pop_size,D),dtype=np.float64) #V变异后的临时种群
    
P.append(np.random.rand(pop_size,D)) #生成一个初始种群P【0】
 
#定义适应度函数 
def fitnessf(X): 
    x=X[0]
    y=X[1]
    value =  (1*cos((1+1)*x+1)+2*cos((2+1)*x+2)+3*cos((3+1)*x+3)+4*cos((4+1)*x+4)+5*cos((5+1)*x+5))*(1*cos((1+1)*y+1)+2*cos((2+1)*y+2)+3*cos((3+1)*y+3)+4*cos((4+1)*y+4)+5*cos((5+1)*y+5))
    return value 
     
#定义适应度计算函数 
def calculate_fitness(): 
    global fitness
    for j in range(pop_size): 
        y=P[g] 
        fitness[j]=fitnessf(y[j]) 
    global fitness_mean
    fitness_mean.append(sum(fitness)/pop_size)
    j=0 
    maxc=0 
    for j in range (pop_size): 
        if (fitness[j]==max(fitness)): 
            maxc=j 
    global P_best
    global fitness_best 
    P_best.append(P[g][maxc]) 
    fitness_best.append(max(fitness))
    
def mutation(X):
    global V
    for i in range (pop_size):
        r1=random.randint(0,pop_size-1)
        r2=r1
        while (r2==r1):
            r2=random.randint(0,pop_size-1)
        V[i]=X[i]+F*(X[r1]-X[r2]) 
            #差分变异
    
def crossover():    #交叉互换
    global U
    for i in range (pop_size):
        if (random.random()<=CR):
            r=random.randint(0,D-1)
            U[i]=V[i]
            for j in range (r,D-1):
                U[i][j]=P[g][i][j]
        
def select():   #选择操作
    W = U
    global P
    for i in range (pop_size):
        if (fitnessf(W[i])<fitness[i]):
            W[i]=P[g][i]
    P.append(W)        

calculate_fitness()#计算适应值
while (g<=G):     
    print (g ,"x=",P_best[g],fitness_best[g]) 
    mutation(P[g])
    crossover()
    select()
    g=g+1
    calculate_fitness() 
    
print ("最优解为：x=",P_best[fitness_best.index(max(fitness_best))])
print ("此时适应度为：z=",max(fitness_best))    