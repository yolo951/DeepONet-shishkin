
import generate_data_1d
import matplotlib.pyplot as plt
import importlib
import torch
import deeponet
import numpy as np
importlib.reload(generate_data_1d)
importlib.reload(deeponet)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load("model_1d_shishkin_ep001.pt")
model = deeponet.DeepONet(2**8+1, 1).to(device)
model.load_state_dict(state_dict)

# ###########对不同数目的f-samples进行测试%%%%%%%%%%%%%%%%%%%%
EP=1/2**7
N_max=2**9+1
NS=2**9+1
alpha=1
dN = int((N_max - 1) / (NS - 1))
sigma_h = min(1 / 2, 2*EP * np.log(N_max) / alpha)
f_test = np.load('f_test.npy')
u_test = np.load('u_test.npy')
weights = np.zeros(N_max)
grid = np.hstack((np.linspace(0, 1 - sigma_h, int((N_max - 1) / 2) + 1),
                                np.linspace(1 - sigma_h, 1, int((N_max - 1) / 2) + 1)[1:]))
weights[1:] = np.hstack(((1 - sigma_h) / ((N_max - 1) / 2) * np.ones(int((N_max - 1) / 2)),
                                        sigma_h / ((N_max - 1) / 2) * np.ones(int((N_max - 1) / 2))))
weights=torch.Tensor(weights).to(device)
grid=torch.Tensor(grid.reshape((-1,1))).to(device)
u_test=torch.Tensor(u_test).to(device)
f_test = torch.Tensor(f_test).to(device)  # f_test.shape=ntrain*NS
test_mse = 0
with torch.no_grad():
    for i in range(4096):
        x=f_test[i].reshape((1,-1))
        input = x.repeat(N_max,1)
        y=u_test[i]
        out=model(input, grid).reshape((1,-1))
        mse = generate_data_1d.weighted_mse_loss(out.view(out.numel(), -1) / 100, y.view(y.numel(), -1) / 100,
                                                 weights.view(weights.numel(), -1))
        test_mse += mse.item()
    test_mse/=4096
    print('test error on high resolution: MSE =', test_mse)

##########对不同数目的locations进行测试%%%%%%%%%%%%%%%%%%%%
EP=1/2**6
N_max=2**12+1
NS=2**7+1
alpha=1
dN = int((N_max - 1) / (NS - 1))
sigma_h = min(1 / 2, 2*EP * np.log(N_max) / alpha)
f_train_h = np.load('f_train.npy')
u_test = np.load('u_train.npy')
weights = np.zeros(N_max)
grid = np.hstack((np.linspace(0, 1 - sigma_h, int((N_max - 1) / 2) + 1),
                                np.linspace(1 - sigma_h, 1, int((N_max - 1) / 2) + 1)[1:]))
weights[1:] = np.hstack(((1 - sigma_h) / ((N_max - 1) / 2) * np.ones(int((N_max - 1) / 2)),
                                        sigma_h / ((N_max - 1) / 2) * np.ones(int((N_max - 1) / 2))))
weights=torch.Tensor(weights).to(device)
grid=torch.Tensor(grid.reshape((-1,1))).to(device)
u_test=torch.Tensor(u_test).to(device)
f_test = torch.Tensor(f_train_h[:, ::dN]).to(device)  # f_test.shape=ntrain*NS
dim = f_train_h.shape[-1]
test_mse = 0
with torch.no_grad():
    for i in range(1000):
        x=f_test[i].reshape((1,-1))
        input = x.repeat(N_max,1)
        y=u_test[i]
        out=model(input,grid).reshape((1,-1))
        mse = generate_data_1d.weighted_mse_loss(out.view(out.numel(), -1) / 100, y.view(y.numel(), -1) / 100,
                                                 weights.view(weights.numel(), -1))
        test_mse += mse.item()
    test_mse/=1000
    print('test error on high resolution: MSE =', test_mse)