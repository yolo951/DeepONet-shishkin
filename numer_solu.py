import numpy as np
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import gaussian_process as gp
import math
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D  # 绘制3D图案
import generate_data_2d
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 17:03:46 2022

@author: Ye Li
"""


def p(x):
    return 1+np.sin(2*np.pi*x)  # p(x)>=alpha=1


def q(x):
    return 1

epsilon=0.0001
S=2**7+1
alpha = 1
sigma = min(1 / 2, epsilon * np.log(S) / alpha)
gridS = np.hstack(
        (np.linspace(0, 1 - sigma, int((S - 1) / 2) + 1), np.linspace(1 - sigma, 1, int((S - 1) / 2) + 1)[1:])).reshape((-1,1))
f=1/(1+np.square(gridS.reshape((1,-1))))

N = f.shape[-1]
Nd = f.shape[0]

grid = np.linspace(0, 1, N)
gridS = np.hstack(
    (np.linspace(0, 1 - sigma, int((N - 1) / 2) + 1), np.linspace(1 - sigma, 1, int((N - 1) / 2) + 1)[1:]))
fS = interpolate.interp1d(np.linspace(0, 1, N), f)(gridS)
h1 = (1 - sigma) / ((N - 1) / 2)
h2 = sigma / ((N - 1) / 2)
yS = np.zeros((Nd, N))

for k in range(Nd):
    U = np.zeros((N - 2, N - 2))
    U[0, :2] = np.array([2 * epsilon / (h1 ** 2) + p(h1) / h1 + q(h1), -epsilon / (h1 ** 2)])
    U[-1, -2:] = np.array(
        [-epsilon / (h2 ** 2) - p(1 - h2) / h2, 2 * epsilon / (h2 ** 2) + p(1 - h2) / h2 + q(1 - h2)])

    Nm = int((N - 3) / 2)  # here N must be odd
    for i in range(1, Nm):
        x_i = (i + 1) * h1
        p1 = -epsilon / (h1 ** 2) - p(x_i) / h1
        r1 = 2 * epsilon / (h1 ** 2) + p(x_i) / h1 + q(x_i)
        q1 = -epsilon / (h1 ** 2)
        U[i, i - 1:i + 2] = np.array([p1, r1, q1])
    x_i = (Nm + 1) * h1
    U[Nm, Nm - 1:Nm + 2] = np.array(
        [-2 * epsilon / (h1 * (h1 + h2)) - p(x_i) / h1, 2 * epsilon / (h1 * h2) + p(x_i) / h1 + q(x_i),
         -2 * epsilon / (h2 * (h1 + h2))])
    for i in range(Nm + 1, N - 3):
        x_i = (Nm + 1) * h1 + (i - Nm) * h2
        p2 = -epsilon / (h2 ** 2) - p(x_i) / h2
        r2 = 2 * epsilon / (h2 ** 2) + p(x_i) / h2 + q(x_i)
        q2 = -epsilon / (h2 ** 2)
        U[i, i - 1:i + 2] = np.array([p2, r2, q2])
B = np.zeros(N - 2)
B[:] = fS[0,1:-1]
u=np.linalg.solve(U, B).flatten()
y = np.zeros(N)
y[1:-1]=u
plt.plot(gridS,y,linestyle='solid', linewidth=1, color='blue',label='reference solution')
plt.legend()
plt.show()
'''N=11
length_scale=1
x = np.linspace(0, 1, num=N)[:, None]
K = gp.kernels.RBF(length_scale=length_scale)
K = K(x)
L = np.linalg.cholesky(K + 1e-13 * np.eye(N))
u = np.random.randn(N, 1)

z=np.dot(L, u).T
plt.plot(x.T,z)
plt.show()'''

'''
x = np.arange(-5.01, 5.01, 0.25)
y = np.arange(-5.01, 5.01, 0.25)
xx, yy = np.meshgrid(x, y)
z = np.sin(xx**2+yy**2)
f = interpolate.interp2d(x, y, z, kind='linear')	#这里是一维的输入


x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
x_, y_ = np.meshgrid(x, y, indexing='ij')
z_ = f(x,y)  # 画图所要表现出来的主函数
fig = plt.figure(figsize=(10, 10), facecolor='white')  # 创建图片
sub = fig.add_subplot(111, projection='3d')  # 添加子图，
surf = sub.plot_surface(x_, y_, z_, cmap=plt.cm.brg)  # 绘制曲面,cmap=plt.cm.brg并设置颜色cmap
cb = fig.colorbar(surf, shrink=0.8, aspect=15)  # 设置颜色棒

sub.set_xlabel(r"x axis")
sub.set_ylabel(r"y axis")
sub.set_zlabel(r"z axis")
plt.show()
'''

'''N = 100
h = np.pi*2/N
l = np.append(np.array([-2 / h ** 2, 1 / h ** 2]), np.zeros(N - 3))
c = np.append(np.array([-2 / h ** 2, 1 / h ** 2]), np.zeros(N - 3))
A = toeplitz(c, l)
x0 = np.linspace(0, 2 * np.pi, N+1)[:, None]
K = gp.kernels.RBF(length_scale=1)
K = K(x0)  # 自协方差矩阵？
L = np.linalg.cholesky(K + 1e-13 * np.eye(101))  # np.linalg.cholesky输出cholesky分解L：A=LL^T
u = np.random.randn(101, 2)  # 返回服从标准正态分布的随机采样值，规格为N*n
F = np.dot(L, u).T
x= np.linspace(0+h, np.pi*2-h, N-1)
u = np.dot(np.linalg.inv(A), F.T[1:-1])
plt.plot(x,u[:,0],x,u[:,1])
plt.legend(['u[0]','u[1]'])
plt.show()'''
'''
F1=np.fromiter(map(lambda x: math.sin(2*math.pi*x), x),dtype=float)
u1 = np.dot(np.linalg.inv(A), F1)
um1=np.array(np.append(np.zeros(1),u[:u.shape[0]-1]))
up1=np.append(u[1:],np.zeros(1))
list = np.hstack([um1[:,None], F,up1[:,None],u])

name = ['u_i-1', 'f','u_i+1','u_i']
test = pd.DataFrame(columns=name, data=list)
test.to_csv('D:\\pycharm\\pycharm_project\\test1.csv')

fig1, p1 = plt.subplots()
p1.plot(x, u, ls='-', lw=2, color='purple')
p1.legend()
p1.set_xlabel('x')
p1.set_ylabel('u')
fig1.savefig('u.eps')
fig2, p2 = plt.subplots()

p2.plot(x, F, ls='-', lw=2, color='red')
p2.legend()
p2.set_xlabel('x')
p2.set_ylabel('f')
fig2.savefig('random_f.eps')
fig3, p3 = plt.subplots()
p3.plot(x, u1, ls='-', lw=2, color='blue')
p3.legend()
p3.set_xlabel('x')
p3.set_ylabel('u0')
fig3.savefig('u0.eps')
'''
