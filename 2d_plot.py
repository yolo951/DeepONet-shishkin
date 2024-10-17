
import torch
import deeponet
from scipy import interpolate
import importlib
import generate_data_2d
importlib.reload(generate_data_2d)
importlib.reload(deeponet)
import numpy as np
import matplotlib.pyplot as plt

i_sample = 88

EP = 0.001
NS = 65
N_max = (NS-1)*8 + 1
alpha = 1
meshtype = "Shishkin"

sigma = min(1 / 2, EP * np.log(NS) / alpha)
sigma_h = min(1 / 2, EP * np.log(N_max) / alpha)
gridS = np.hstack((np.linspace(0, 1 - sigma, int((NS - 1) / 2) + 1),
                   np.linspace(1 - sigma, 1, int((NS - 1) / 2) + 1)[1:]))
grid_h = np.hstack((np.linspace(0, 1 - sigma_h, int((N_max - 1) / 2) + 1),
                    np.linspace(1 - sigma_h, 1, int((N_max - 1) / 2) + 1)[1:]))
X1, X2 = np.meshgrid(grid_h, grid_h)

f_test = np.load('2d_f_test.npy')
u_test = np.load('2d_u_test.npy')
model = torch.load('2d_model_shishkin_ep001.pt').cpu()
model_equal = torch.load('2d_model_equal_ep001.pt').cpu()


f_test = np.array([generate_data_2d.grid_to_vec(y, NS) for y in f_test])

dim = NS  # N_max
grid_vec = np.zeros((dim ** 2, 2))
for j in range(dim):
    for i in range(dim):
        grid_vec[j * dim + i] = [gridS[i], gridS[j]]  # [grid_h[i], grid_h[j]]

u_test_h0 = np.array([generate_data_2d.grid_to_vec(interpolate.interp2d(gridS, gridS, y)(grid_h, grid_h), N_max) for y in u_test])


Yt = generate_data_2d.vec_to_grid(u_test_h0[i_sample].flatten(), N_max)
x = torch.unsqueeze(torch.Tensor(f_test[i_sample, :, 0]), 0)
l = torch.unsqueeze(torch.Tensor(grid_vec), 0)
Yp = generate_data_2d.vec_to_grid(model(x, l).detach().numpy().flatten(), NS)
Yp = interpolate.interp2d(gridS, gridS, Yp)(grid_h, grid_h) 
Yp_equal = generate_data_2d.vec_to_grid(model_equal(x, l).detach().numpy().flatten(), NS)
Yp_equal = interpolate.interp2d(gridS, gridS, Yp_equal)(grid_h, grid_h) 
Yt = Yt/100
Yp = Yp/100
Yp_equal = Yp_equal/100

# plt.figure()
# ax3 = plt.axes(projection='3d')
# ax3.plot_surface(X1,X2,Yt,cmap='rainbow')
# plt.title('True')
# plt.figure()
# ax3 = plt.axes(projection='3d')
# ax3.plot_surface(X1,X2,Yp,cmap='rainbow')
# plt.title('DeepONet Shishkin mesh')
# plt.figure()
# ax3 = plt.axes(projection='3d')
# ax3.plot_surface(X1,X2,Yp_equal,cmap='rainbow')
# plt.title('DeepONet Equal mesh')
# plt.figure()
# ax3 = plt.axes(projection='3d')
# ax3.plot_surface(X1,X2,np.abs(Yt-Yp),cmap='rainbow')
# plt.title('Shishkin error')
# plt.figure()
# ax3 = plt.axes(projection='3d')
# ax3.plot_surface(X1,X2,np.abs(Yt-Yp_equal),cmap='rainbow')
# plt.title('Equal error')

plt.figure()
plt.pcolor(X1, X2, Yt, cmap='jet')
plt.colorbar()
plt.savefig(r"D:\mypaper\eajam\6_true.png", dpi=80)
plt.figure()
plt.pcolor(X1, X2, Yp, cmap='jet')
plt.colorbar()
plt.savefig(r"D:\mypaper\eajam\6_shishkin.png", dpi=80)
plt.figure()
plt.pcolor(X1, X2, Yp_equal, cmap='jet')
plt.colorbar()
plt.savefig(r"D:\mypaper\eajam\6_equal.png", dpi=80)
plt.figure()
plt.pcolor(X1, X2, np.abs(Yt-Yp), cmap='jet')
plt.colorbar()
plt.savefig(r"D:\mypaper\eajam\6_shishkin_error.png", dpi=80)
plt.figure()
plt.pcolor(X1, X2, np.abs(Yt-Yp_equal), cmap='jet')
plt.colorbar()
plt.savefig(r"D:\mypaper\eajam\6_equal_error.png", dpi=80)
plt.figure()
plt.pcolor(X1[int(N_max/2):, int(N_max/2):], X2[int(N_max/2):, int(N_max/2):], np.abs(Yt[int(N_max/2):, int(N_max/2):]-Yp[int(N_max/2):, int(N_max/2):]), cmap='jet')
plt.colorbar()
plt.savefig(r"D:\mypaper\eajam\6_shishkin_sub.png", dpi=80)
plt.figure()
plt.pcolor(X1[int(N_max/2):, int(N_max/2):], X2[int(N_max/2):, int(N_max/2):], np.abs(Yt[int(N_max/2):, int(N_max/2):]-Yp_equal[int(N_max/2):, int(N_max/2):]), cmap='jet')
plt.colorbar()
plt.savefig(r"D:\mypaper\eajam\6_equal_sub.png", dpi=80)

plt.show()

