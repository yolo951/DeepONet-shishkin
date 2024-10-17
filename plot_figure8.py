
import generate_data_1d
import importlib
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
importlib.reload(generate_data_1d)

N=1001
EP=0.01
alpha=1
sigma=min(1 / 2, 2*EP * np.log(N) / alpha)
grid=np.linspace(0,1,N)
gridS = np.hstack((np.linspace(0, 1 - sigma, int((N - 1) / 2) + 1),
                               np.linspace(1 - sigma, 1, int((N - 1) / 2) + 1)[1:]))
f1= generate_data_1d.generate(samples=1, out_dim=N)
f2= generate_data_1d.generate(samples=1, out_dim=N)
f3= generate_data_1d.generate(samples=1, out_dim=N)
f4= generate_data_1d.generate(samples=1, out_dim=N)
f5= generate_data_1d.generate(samples=1, out_dim=N)
y1= generate_data_1d.FD_AD_1d(f1, EP) / 100
y2= generate_data_1d.FD_AD_1d(f2, EP) / 100
y3= generate_data_1d.FD_AD_1d(f3, EP) / 100
y4= generate_data_1d.FD_AD_1d(f4, EP) / 100
y5= generate_data_1d.FD_AD_1d(f5, EP) / 100
plt.figure()
plt.plot(gridS, interpolate.interp1d(grid, f1)(gridS).reshape((-1,1)), label='$F_1$', color="blue",linewidth=1.5)
plt.plot(gridS, interpolate.interp1d(grid, f2)(gridS).reshape((-1,1)), label='$F_2$', color="red",linewidth=1.5)
plt.plot(gridS, interpolate.interp1d(grid, f3)(gridS).reshape((-1,1)), label='$F_3$', color="#367244",linewidth=1.5)
plt.plot(gridS, interpolate.interp1d(grid, f4)(gridS).reshape((-1,1)), label='$F_4$', color="#A8301C",linewidth=1.5)
plt.plot(gridS, interpolate.interp1d(grid, f5)(gridS).reshape((-1,1)), label='$F_5$', color="#371650",linewidth=1.5)


plt.legend(fontsize=11)
plt.savefig(r"D:\mypaper\eajam\random_f.eps")

plt.figure()
plt.plot(gridS, y1.reshape((-1,1)), label='$u_1$', color="blue",linewidth=1.5)
plt.plot(gridS, y2.reshape((-1,1)), label='$u_2$',color="red",linewidth=1.5)
plt.plot(gridS, y3.reshape((-1,1)), label='$u_3$',color="#367244",linewidth=1.5)
plt.plot(gridS, y4.reshape((-1,1)), label='$u_4$',color="#A8301C",linewidth=1.5)
plt.plot(gridS, y5.reshape((-1,1)), label='$u_5$',color="#371650",linewidth=1.5)


plt.legend(fontsize=11)
plt.savefig(r"D:\mypaper\eajam\u.eps")
plt.show()