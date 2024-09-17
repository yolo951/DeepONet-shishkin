
# #
# import numpy as np
# import generate_data_1d
# import importlib
#
#
#
# importlib.reload(generate_data_1d)
# meshtype = "Shishkin"  #"Shishkin" or "Equal"
# n_test = 2**12
# EP = 1/2**7#, 1/2**7, 1/2**8, 1/2**9, 1/2**10, 1/2**11]
# N_max = 2**9+1
#
#
#
# f_test_h = np.load('f_test.npy')
#
# # f_test_h = generate_data_1d.generate(samples=n_test,out_dim=N_max) #equally sampled.f_train_h.shape=(ntrain,N_max)
# # np.save('f_test.npy',f_test_h)
#
#
# u_test_h = generate_data_1d.FD_AD_1d(f_test_h,EP,meshtype)
# np.save('u_test.npy',u_test_h)
# # u_test = generate_data_1d.FD_AD_1d(f_train_h,EP,"Shishkin")
# # np.save('u_test.npy', u_test)
# print('end')



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



# import generate_data_1d
# import matplotlib.pyplot as plt
# import importlib
# import torch
# import deeponet
# import numpy as np
# importlib.reload(generate_data_1d)
# importlib.reload(deeponet)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # torch.save(model.state_dict(), "model_ep6.pt")
# state_dict = torch.load("model_1d_shishkin_ep001.pt")
# model = deeponet.DeepONet(2**8+1, 1).to(device)
# model.load_state_dict(state_dict)
#
#
# epsilon=0.001
# alpha=1
# N=2**8+1
# S=2**8+1
# samples=10
# def u_exact(x):
#     return x-(np.exp(-(1-x)/epsilon)-np.exp(-1/epsilon))/(1-np.exp(-1/epsilon))
# sigma_S = min(1 / 2, 2*epsilon * np.log(S) / alpha)
# sigma_N = min(1 / 2, 2*epsilon * np.log(N) / alpha)
# gridN = np.hstack(
#         (np.linspace(0, 1 - sigma_N, int((N - 1) / 2) + 1), np.linspace(1 - sigma_N, 1, int((N - 1) / 2) + 1)[1:]))
# gridS = np.hstack(
#         (np.linspace(0, 1 - sigma_S, int((S - 1) / 2) + 1), np.linspace(1 - sigma_S, 1, int((S - 1) / 2) + 1)[1:])).reshape((-1,1))
# # f0=np.exp(np.linspace(0,1,N)).reshape((1,-1))
# fig,axs=plt.subplots(1,3,figsize=(13,4))
# f0=generate_data_1d.generate(samples=samples,out_dim=N,length_scale=0.2)
# grid=np.linspace(0,1,f0.shape[-1])
# for i in range(10):
#     axs[0].plot(grid,f0[i])
# axs[0].set_title('$f$ samples($L$=0.2)')
# ff=np.zeros((samples*S,N))
# grid=np.zeros((samples*S,1))
# for i in range(samples*S):
#     ff[i]=f0[i//S,:]
#     grid[i]=gridS[i%S]
# ff=torch.Tensor(ff).to(device)
# grid=torch.Tensor(grid).to(device)
# y=model(ff,grid).view(-1)/100
# y=y.cpu()
# y=y.detach().numpy()
# y_pred=np.zeros((samples,S))
# for i in range(samples):
#     y_pred[i]=y[i*S:(i+1)*S].reshape((-1,))
# y_exact=generate_data_1d.FD_AD_1d(f0,epsilon)/100
# for i in range(samples):
#     if not i:
#         axs[1].plot(gridS, y_exact[i].reshape((-1,)), linestyle='solid', linewidth=2, color='blue', label='reference solution')
#         axs[1].plot(gridS, y_pred[i], linestyle='--', linewidth=2, color='red', label='DeepONet')
#         axs[1].legend()
#     else:
#         axs[1].plot(gridS, y_exact[i].reshape((-1,)), linestyle='solid', linewidth=2, color='blue')  # ,label='reference solution')
#         axs[1].plot(gridS, y_pred[i], linestyle='--', linewidth=2, color='red')  # ,label='DeepONet')
# axs[1].tick_params(axis='both', which='major', labelsize=15)
# axs[1].set_title('predictions')
# for i in range(samples):
#     if not i:
#         axs[2].plot(gridS[-128:], y_exact[i].reshape((-1,))[-128:], linestyle='solid', linewidth=2, color='blue', label='reference solution')
#         axs[2].plot(gridS[-128:], y_pred[i][-128:], linestyle='--', linewidth=2, color='red', label='DeepONet')
#         axs[2].legend()
#     else:
#         axs[2].plot(gridS[-128:], y_exact[i].reshape((-1,))[-128:], linestyle='solid', linewidth=2, color='blue')  # ,label='reference solution')
#         axs[2].plot(gridS[-128:], y_pred[i][-128:], linestyle='--', linewidth=2, color='red')  # ,label='DeepONet')
# axs[2].tick_params(axis='both', which='major', labelsize=15)
# axs[2].set_title('predictions')
# fig.tight_layout()
# for j in range(10):
#     f0=generate_data_1d.generate(samples=1,out_dim=2**8+1,length_scale=2)
#     f=torch.tensor(f0)
#     ff=np.zeros((S,N))
#     for i in range(S):
#         ff[i]=f[:]
#     ff=torch.Tensor(ff).to(device)
#     gridS=torch.Tensor(gridS).to(device)
#     y=model(ff,gridS).view(-1)/100
#     y=y.cpu()
#     gridS = np.hstack(
#         (np.linspace(0, 1 - sigma_S, int((S - 1) / 2) + 1), np.linspace(1 - sigma_S, 1, int((S - 1) / 2) + 1)[1:])).reshape((-1,1))
#
#     y1=generate_data_1d.FD_AD_1d(f0,epsilon)/100
#     if not j:
#         plt.plot(gridS,y1.reshape((-1,)),linestyle='solid', linewidth=2, color='blue',label='reference solution')
#         plt.plot(gridS,y.detach().numpy(),linestyle='--',linewidth=2,color='red',label='DeepONet')
#         plt.legend(fontsize=15, frameon=True)
#     else:
#         plt.plot(gridS, y1.reshape((-1,)), linestyle='solid', linewidth=2, color='blue')  # ,label='reference solution')
#         plt.plot(gridS, y.detach().numpy(), linestyle='--', linewidth=2, color='red')  # ,label='DeepONet')
#
# ax2= plt.gca()
# ax2.tick_params(axis='both', which='major', labelsize=15)
# plt.title('length scale=0.1')
# plt.show()



# fig=plt.figure(figsize=(6.4,2))#figsize=(6.4,4.8)
# plt.plot(gridS[-128:],y1.reshape((-1,))[-128:],linestyle='solid', linewidth=1.5, color='blue',label='reference solution')
# plt.plot(gridS[-128:],y.detach().numpy()[-128:],linestyle='--',linewidth=1.5,color='red',label='DeepONet')
# legend2 = plt.legend(fontsize=10, frameon=True)
# ax2= plt.gca()
# ax2.tick_params(axis='both', which='major', labelsize=10)
# plt.show()

# # ###########对不同数目的f-samples进行测试%%%%%%%%%%%%%%%%%%%%
# EP=1/2**7
# N_max=2**9+1
# NS=2**9+1
# alpha=1
# dN = int((N_max - 1) / (NS - 1))
# sigma_h = min(1 / 2, 2*EP * np.log(N_max) / alpha)
# f_test = np.load('f_test.npy')
# u_test = np.load('u_test.npy')
# weights = np.zeros(N_max)
# grid = np.hstack((np.linspace(0, 1 - sigma_h, int((N_max - 1) / 2) + 1),
#                                 np.linspace(1 - sigma_h, 1, int((N_max - 1) / 2) + 1)[1:]))
# weights[1:] = np.hstack(((1 - sigma_h) / ((N_max - 1) / 2) * np.ones(int((N_max - 1) / 2)),
#                                         sigma_h / ((N_max - 1) / 2) * np.ones(int((N_max - 1) / 2))))
# weights=torch.Tensor(weights).to(device)
# grid=torch.Tensor(grid.reshape((-1,1))).to(device)
# u_test=torch.Tensor(u_test).to(device)
# f_test = torch.Tensor(f_test).to(device)  # f_test.shape=ntrain*NS
# test_mse = 0
# with torch.no_grad():
#     for i in range(4096):
#         x=f_test[i].reshape((1,-1))
#         input = x.repeat(N_max,1)
#         y=u_test[i]
#         out=model(input,grid).reshape((1,-1))
#         mse = generate_data_1d.weighted_mse_loss(out.view(out.numel(), -1) / 100, y.view(y.numel(), -1) / 100,
#                                                  weights.view(weights.numel(), -1))
#         test_mse += mse.item()
#     test_mse/=4096
#     print('test error on high resolution: MSE =', test_mse)

























###########对不同数目的locations进行测试%%%%%%%%%%%%%%%%%%%%
# EP=1/2**6
# N_max=2**12+1
# NS=2**7+1
# alpha=1
# dN = int((N_max - 1) / (NS - 1))
# sigma_h = min(1 / 2, 2*EP * np.log(N_max) / alpha)
# f_train_h = np.load('f_train.npy')
# u_test = np.load('u_train.npy')
# weights = np.zeros(N_max)
# grid = np.hstack((np.linspace(0, 1 - sigma_h, int((N_max - 1) / 2) + 1),
#                                 np.linspace(1 - sigma_h, 1, int((N_max - 1) / 2) + 1)[1:]))
# weights[1:] = np.hstack(((1 - sigma_h) / ((N_max - 1) / 2) * np.ones(int((N_max - 1) / 2)),
#                                         sigma_h / ((N_max - 1) / 2) * np.ones(int((N_max - 1) / 2))))
# weights=torch.Tensor(weights).to(device)
# grid=torch.Tensor(grid.reshape((-1,1))).to(device)
# u_test=torch.Tensor(u_test).to(device)
# f_test = torch.Tensor(f_train_h[:, ::dN]).to(device)  # f_test.shape=ntrain*NS
# dim = f_train_h.shape[-1]
# test_mse = 0
# with torch.no_grad():
#     for i in range(1000):
#         x=f_test[i].reshape((1,-1))
#         input = x.repeat(N_max,1)
#         y=u_test[i]
#         out=model(input,grid).reshape((1,-1))
#         mse = generate_data_1d.weighted_mse_loss(out.view(out.numel(), -1) / 100, y.view(y.numel(), -1) / 100,
#                                                  weights.view(weights.numel(), -1))
#         test_mse += mse.item()
#     test_mse/=1000
#     print('test error on high resolution: MSE =', test_mse)











