
# import numpy as np
# import generate_data_1d
# import importlib

# importlib.reload(generate_data_1d)
# meshtype = "Shishkin"  #"Shishkin" or "Equal"
# n_test = 2**12
# EP = 1/2**7 #, 1/2**7, 1/2**8, 1/2**9, 1/2**10, 1/2**11]
# N_max = 2**9+1

# f_test_h = np.load('f_test.npy')
# # f_test_h = generate_data_1d.generate(samples=n_test,out_dim=N_max) #equally sampled.f_train_h.shape=(ntrain,N_max)
# # np.save('f_test.npy',f_test_h)

# u_test_h = generate_data_1d.FD_AD_1d(f_test_h,EP,meshtype)
# np.save('u_test.npy',u_test_h)
# # u_test = generate_data_1d.FD_AD_1d(f_train_h,EP,"Shishkin")
# # np.save('u_test.npy', u_test)
# print('end')


import generate_data_1d
import matplotlib.pyplot as plt
import importlib
import torch
import deeponet
import numpy as np
importlib.reload(generate_data_1d)
importlib.reload(deeponet)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # torch.save(model.state_dict(), "model_ep6.pt")
state_dict = torch.load("model_1d_shishkin_ep001.pt")
model = deeponet.DeepONet(2**8+1, 1).to(device)
model.load_state_dict(state_dict)


epsilon=0.001
alpha=1
N=2**8+1
S=2**8+1
samples=10
def u_exact(x):
    return x-(np.exp(-(1-x)/epsilon)-np.exp(-1/epsilon))/(1-np.exp(-1/epsilon))
sigma_S = min(1 / 2, 2*epsilon * np.log(S) / alpha)
sigma_N = min(1 / 2, 2*epsilon * np.log(N) / alpha)
gridN = np.hstack(
        (np.linspace(0, 1 - sigma_N, int((N - 1) / 2) + 1), np.linspace(1 - sigma_N, 1, int((N - 1) / 2) + 1)[1:]))
gridS = np.hstack(
        (np.linspace(0, 1 - sigma_S, int((S - 1) / 2) + 1), np.linspace(1 - sigma_S, 1, int((S - 1) / 2) + 1)[1:])).reshape((-1,1))
# f0=np.exp(np.linspace(0,1,N)).reshape((1,-1))
fig,axs=plt.subplots(1,3,figsize=(13,4))
f0=generate_data_1d.generate(samples=samples,out_dim=N,length_scale=0.2)
grid=np.linspace(0,1,f0.shape[-1])
for i in range(10):
    axs[0].plot(grid,f0[i])
axs[0].set_title('$f$ samples($L$=0.2)')
ff=np.zeros((samples*S,N))
grid=np.zeros((samples*S,1))
for i in range(samples*S):
    ff[i]=f0[i//S,:]
    grid[i]=gridS[i%S]
ff=torch.Tensor(ff).to(device)
grid=torch.Tensor(grid).to(device)
y=model(ff,grid).view(-1)/100
y=y.cpu()
y=y.detach().numpy()
y_pred=np.zeros((samples,S))
for i in range(samples):
    y_pred[i]=y[i*S:(i+1)*S].reshape((-1,))
y_exact=generate_data_1d.FD_AD_1d(f0,epsilon)/100
for i in range(samples):
    if not i:
        axs[1].plot(gridS, y_exact[i].reshape((-1,)), linestyle='solid', linewidth=2, color='blue', label='reference solution')
        axs[1].plot(gridS, y_pred[i], linestyle='--', linewidth=2, color='red', label='DeepONet')
        axs[1].legend()
    else:
        axs[1].plot(gridS, y_exact[i].reshape((-1,)), linestyle='solid', linewidth=2, color='blue')  # ,label='reference solution')
        axs[1].plot(gridS, y_pred[i], linestyle='--', linewidth=2, color='red')  # ,label='DeepONet')
axs[1].tick_params(axis='both', which='major', labelsize=15)
axs[1].set_title('predictions')
for i in range(samples):
    if not i:
        axs[2].plot(gridS[-128:], y_exact[i].reshape((-1,))[-128:], linestyle='solid', linewidth=2, color='blue', label='reference solution')
        axs[2].plot(gridS[-128:], y_pred[i][-128:], linestyle='--', linewidth=2, color='red', label='DeepONet')
        axs[2].legend()
    else:
        axs[2].plot(gridS[-128:], y_exact[i].reshape((-1,))[-128:], linestyle='solid', linewidth=2, color='blue')  # ,label='reference solution')
        axs[2].plot(gridS[-128:], y_pred[i][-128:], linestyle='--', linewidth=2, color='red')  # ,label='DeepONet')
axs[2].tick_params(axis='both', which='major', labelsize=15)
axs[2].set_title('predictions')
fig.tight_layout()
for j in range(10):
    f0=generate_data_1d.generate(samples=1,out_dim=2**8+1,length_scale=2)
    f=torch.tensor(f0)
    ff=np.zeros((S,N))
    for i in range(S):
        ff[i]=f[:]
    ff=torch.Tensor(ff).to(device)
    gridS=torch.Tensor(gridS).to(device)
    y=model(ff,gridS).view(-1)/100
    y=y.cpu()
    gridS = np.hstack(
        (np.linspace(0, 1 - sigma_S, int((S - 1) / 2) + 1), np.linspace(1 - sigma_S, 1, int((S - 1) / 2) + 1)[1:])).reshape((-1,1))

    y1=generate_data_1d.FD_AD_1d(f0,epsilon)/100
    if not j:
        plt.plot(gridS,y1.reshape((-1,)),linestyle='solid', linewidth=2, color='blue',label='reference solution')
        plt.plot(gridS,y.detach().numpy(),linestyle='--',linewidth=2,color='red',label='DeepONet')
        plt.legend(fontsize=15, frameon=True)
    else:
        plt.plot(gridS, y1.reshape((-1,)), linestyle='solid', linewidth=2, color='blue')  # ,label='reference solution')
        plt.plot(gridS, y.detach().numpy(), linestyle='--', linewidth=2, color='red')  # ,label='DeepONet')

ax2= plt.gca()
ax2.tick_params(axis='both', which='major', labelsize=15)
plt.title('length scale=0.1')
plt.show()

fig=plt.figure(figsize=(6.4,2))#figsize=(6.4,4.8)
plt.plot(gridS[-128:],y1.reshape((-1,))[-128:],linestyle='solid', linewidth=1.5, color='blue',label='reference solution')
plt.plot(gridS[-128:],y.detach().numpy()[-128:],linestyle='--',linewidth=1.5,color='red',label='DeepONet')
legend2 = plt.legend(fontsize=10, frameon=True)
ax2= plt.gca()
ax2.tick_params(axis='both', which='major', labelsize=10)
plt.show()













