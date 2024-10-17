
import importlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from timeit import default_timer
from Adam import Adam
import deeponet
from scipy import interpolate
# from generate_date import *
import generate_data_1d

importlib.reload(generate_data_1d)
importlib.reload(deeponet)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

meshtype = "Shishkin"  # "Shishkin" or "Equal"
ntrain = 1000

learning_rate = 0.001
epochs = 250
step_size =20
gamma =0.5
alpha = 1
EP_list = [0.001]  # 1/2**6]#, 1/2**7, 1/2**8, 1/2**9, 1/2**10, 1/2**11]
N_list = [2**8+1]  # , 2**7+1, 2**8+1, 2**9+1, 2**10+1]
N_max=2**8+1

f_test_h = generate_data_1d.generate(samples=100,out_dim=N_max)  #np.load('f_test.npy')
N_max = f_test_h.shape[-1]
f_train_h=f_test_h[:ntrain,:]
loss_history = dict()


for EP in EP_list:
    # u_train_h = np.load('u_train.npy')
    u_train_h = generate_data_1d.FD_AD_1d(f_train_h, EP, meshtype)
    for NS in N_list:
        mse_history = []
        print("N value : ", NS - 1, ", epsilon value : ", EP)
        sigma = min(1 / 2, 2*EP * np.log(NS) / alpha)
        sigma_h = min(1 / 2, 2*EP * np.log(N_max) / alpha)
        weights = np.zeros(NS)
        weights_h = np.zeros(N_max)
        dN = int((N_max - 1) / (NS - 1))
        f_train = torch.Tensor(f_train_h[:, ::dN])

        if meshtype == "Shishkin":
            gridS = np.hstack((np.linspace(0, 1 - sigma, int((NS - 1) / 2) + 1),
                               np.linspace(1 - sigma, 1, int((NS - 1) / 2) + 1)[1:]))
            grid_h = np.hstack((np.linspace(0, 1 - sigma_h, int((N_max - 1) / 2) + 1),
                                np.linspace(1 - sigma_h, 1, int((N_max - 1) / 2) + 1)[1:]))
            weights[1:] = np.hstack(((1 - sigma) / ((NS - 1) / 2) * np.ones(int((NS - 1) / 2)),
                                       sigma / ((NS - 1) / 2)* np.ones(int((NS - 1) / 2))))#sigma / ((NS - 1) / 2)
            weights_h[1:] = np.hstack(((1 - sigma_h) / ((N_max - 1) / 2) * np.ones(int((N_max - 1) / 2)),
                                        sigma_h / ((N_max - 1) / 2) * np.ones(int((N_max - 1) / 2))))
        else:
            gridS = np.linspace(0, 1, NS)
            grid_h = np.linspace(0, 1, N_max)
            weights[1:] = 1 / (NS - 1) * np.ones(NS - 1)
            weights_h[1:] = 1 / (N_max - 1) * np.ones(N_max - 1)
        u_train = interpolate.interp1d(grid_h, u_train_h)(gridS)

        dim = NS
        batch_size = 2 ** 8 + 1  # dim
        N = f_train.shape[0] * dim
        loc = np.zeros((N, 1))
        res = np.zeros((N, 1))
        f = np.zeros((N, dim))
        weights_train = np.zeros((N, 1))
        for i in range(N):
            f[i] = f_train[i // dim]
            loc[i, 0] = gridS[i % dim]
            res[i, 0] = u_train[i // dim, i % dim]
            weights_train[i, 0] = weights[i % dim]

        f_train = torch.Tensor(f).to(device)
        loc_train = torch.Tensor(loc).to(device)
        res_train = torch.Tensor(res).to(device)
        weights_train = torch.Tensor(weights_train).to(device)
        #mean = f_train.mean(dim=0)
        #std = f_train.std(dim=0)
        #f_train_normalized = (f_train - mean) / std
        #res_train_normalized,res_weight_norm = transform_outputs(res_train, weights_train)
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(f_train,
                                                                                  loc_train, res_train, weights_train),
                                                   batch_size=128, shuffle=True)

        model = deeponet.DeepONet(dim, 1).to(device)
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        start = default_timer()
        myloss = generate_data_1d.LpLoss(size_average=False)
        for ep in range(epochs):
            model.train()
            t1 = default_timer()
            train_mse = 0
            for x, l, y, w in train_loader:
                optimizer.zero_grad()
                out = model(x, l)
                # mse = generate_data_1d.weighted_mse_loss(out.view(out.numel(), -1), y.view(y.numel(), -1), w.view(w.numel(), -1))
                mse = F.mse_loss(out.view(out.numel(), -1), y.view(y.numel(), -1), reduction='mean')
                mse.backward()
                optimizer.step()
                train_mse += mse.item()
            scheduler.step()
            # train_mse /= ntrain
            train_mse /= len(train_loader)
            t2 = default_timer()
            mse_history.append(train_mse)
            print('\repoch {:d}/{:d} , MSE = {:.6f}, using {:.6f}s'.format(ep + 1, epochs, train_mse, t2 - t1), end='',
                  flush=True)

        print('Total training time:', default_timer() - start, 's')
        loss_history["{} {}".format(NS, EP)] = mse_history

        # f_test = torch.Tensor(f_train_h[:, ::dN])  # f_test.shape=ntrain*NS
        # dim = f_train_h.shape[-1]
        # grid = np.linspace(0, 1, dim)
        # N = f_train_h.shape[0] * dim
        # loc = np.zeros((N, 1))
        # res = np.zeros((N, 1))
        # f = np.zeros((N, f_test.shape[-1]))
        # u_train_h0 = interpolate.interp1d(grid_h, u_train_h)(grid)
        # weights_test = np.zeros((N, 1))
        # for i in range(N):
        #     f[i] = f_test[i // dim]
        #     loc[i, 0] = grid[i % dim]
        #     res[i, 0] = u_train_h0[i // dim, i % dim]
        #     weights_test[i, 0] = weights_h[i % dim]
        # f_test = torch.Tensor(f)
        # loc = torch.Tensor(loc)
        # res = torch.Tensor(res)
        # weights_test = torch.tensor(weights_test)
        # #f_test_transform = (f_test - mean) / std
        # #res_test_transform=res/res_weight_norm
        # test_h_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(f_test,
        #                                                                            loc, res, weights_test),
        #                                             batch_size=dim, shuffle=False)

        # pred_h = torch.zeros(u_train_h0.shape)
        # index = 0
        # test_mse = 0
        # test_l2 = 0
        # with torch.no_grad():  # 表示不再计算梯度，以免对之前的梯度产生影响，也节省计算耗费
        #     for x, l, y, w in test_h_loader:
        #         out = model(x, l).view(-1)
        #         pred_h[index] = out
        #         mse = weighted_mse_loss(out.view(out.numel(), -1), y.view(y.numel(), -1), w.view(w.numel(), -1))
        #         #mse = F.mse_loss(out.view(1, -1), y.view(1, -1), reduction='mean')
        #         l2 = myloss(out.view(1, -1), y.view(1, -1))
        #         test_mse += mse.item()
        #         test_l2 += l2.item()
        #         index += 1
        #     test_mse /= ntrain
        #     #test_mse /= len(test_h_loader)
        #     test_l2 /= ntrain
        #     print('test error on high resolution: L2 = ', test_l2, 'MSE =', test_mse)

        # # residual = pred_h-u_train_h
        # # fig = plt.figure()
        # # x_grid = np.linspace(0, 1, N_max)
        # # for i in range(100):
        # #    plt.plot(x_grid,residual[i].detach().numpy())
        # # plt.show()

for NS in N_list:
    plt.figure()
    for EP in EP_list:
        plt.plot(loss_history["{} {}".format(NS, EP)], label="ep = {}".format(EP))
        plt.yscale("log")
    plt.title("N = {}".format(NS))
    plt.legend()
    plt.show()

# for NS in N_list:
#     plt.figure()
#     for EP in EP_list:
#         fig = plt.figure(figsize=(4.8, 4.8))
#         plt.plot(loss_history["{} {}".format(NS, EP)], color="blue")
#         plt.xlabel("epochs", fontsize=12)
#         plt.ylabel("MSE", fontsize=12)
#         plt.yscale("log")
#         plt.legend()
#         plt.show()
