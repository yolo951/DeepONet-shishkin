
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from timeit import default_timer
from Adam import Adam
import deeponet
from scipy import interpolate
import importlib
import generate_data_2d
importlib.reload(generate_data_2d)
importlib.reload(deeponet)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
meshtype = "Equal"  # "Shishkin" or "Equal"
ntrain = 1000
ntest = 200
learning_rate = 0.0001
epochs = 2000#20000
step_size = 400#4000 
batch_size = 5
gamma = 0.2
alpha = 1

EP_list = [0.001]  # , 1/2**7, 1/2**8, 1/2**9, 1/2**10, 1/2**11]
N_list = [2 ** 6 + 1]  # , 2**7+1, 2**8+1, 2**9+1, 2**10+1]
N_max = 2 ** 6 + 1



# equally sampled.f_train_h.shape=(ntrain,N_max,N_max)
#f_train_h = generate_data_2d.generate(samples=ntrain,out_dim=N_max)
f_train_h = np.load('2d_f_train.npy')
f_test_h = np.load('2d_f_test.npy')
loss_history = dict()

for EP in EP_list:
    u_train_h = generate_data_2d.FD_AD_2d(f_train_h, EP, meshtype)
    #u_train_h = np.load('2d_u_train.npy')
    for NS in N_list:
        mse_history = []
        print("N value : ", NS - 1, ", epsilon value : ", EP)
        sigma = min(1 / 2, EP * np.log(NS) / alpha)
        sigma_h = min(1 / 2, EP * np.log(N_max) / alpha)

        dN = int((N_max - 1) / (NS - 1))
        f_train = torch.Tensor(f_train_h[:, ::dN,::dN])

        if meshtype == "Shishkin":
            gridS = np.hstack((np.linspace(0, 1 - sigma, int((NS - 1) / 2) + 1),
                               np.linspace(1 - sigma, 1, int((NS - 1) / 2) + 1)[1:]))
            grid_h = np.hstack((np.linspace(0, 1 - sigma_h, int((N_max - 1) / 2) + 1),
                                np.linspace(1 - sigma_h, 1, int((N_max - 1) / 2) + 1)[1:]))
        else:
            gridS = np.linspace(0, 1, NS)
            grid_h = np.linspace(0, 1, N_max)



        grid_vec=np.zeros((NS**2,2))
        u_train = np.array([generate_data_2d.grid_to_vec(interpolate.interp2d(grid_h, grid_h, y)(gridS, gridS), NS) for y in u_train_h])
        f_train = np.array([generate_data_2d.grid_to_vec(y, NS) for y in f_train])
        for j in range(NS):
            for i in range(NS):
                grid_vec[j*NS+i]=[gridS[i],gridS[j]]



        dim = NS
        N = f_train.shape[0]
        loc = np.zeros((N, dim**2, 2))
        res = np.zeros((N, dim**2, 1))
        f = np.zeros((N, dim**2))

        for i in range(N):
            f[i] = f_train[i].flatten()
            loc[i] = grid_vec
            res[i] = u_train[i]

        f_train = torch.Tensor(f).to(device)
        loc_train = torch.Tensor(loc).to(device)
        res_train = torch.Tensor(res).to(device)
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(f_train, loc_train, res_train), batch_size=batch_size, shuffle=True)

        model = deeponet.DeepONet(dim ** 2, 2).to(device)
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        start = default_timer()
        myloss = generate_data_2d.LpLoss(size_average=False)
        b0 = torch.zeros([batch_size, 4*(dim-1), 1])
        bpoint = torch.zeros([1, 4*(dim-1), 2])
        for i in range(dim-1):
            bpoint[0,i] = torch.Tensor([gridS[i],0])
            bpoint[0,i+dim-1] = torch.Tensor([1,gridS[i]])
            bpoint[0,i+2*(dim-1)] = torch.Tensor([gridS[i],1])
            bpoint[0,i+3*(dim-1)] = torch.Tensor([0,gridS[i]])
        bpoint = bpoint.repeat([batch_size,1,1])
        b0, bpoint = b0.to(device), bpoint.to(device)
        for ep in range(epochs):
            model.train()
            t1 = default_timer()
            train_mse = 0
            for x, l, y in train_loader:
                optimizer.zero_grad()
                out = model(x, l)
                # mse = generate_data_2d.weighted_mse_loss(out.view(out.numel(), -1), y.view(y.numel(), -1),
                #                                          w.view(w.numel(), -1))
                mse1 = F.mse_loss(out.view(out.numel(), -1), y.view(y.numel(), -1), reduction='mean')#y是数据集里真实的输出值，这里由fdm得到
                
                bout = model(x,bpoint)
                mse2 = F.mse_loss(bout.view(bout.numel(), -1), b0.view(b0.numel(), -1), reduction='mean')  #加强边界损失
                mse = mse1 + 0.1*mse2
                
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


###########################test%%%%%%%%%%%%%%%%%%%%%%

        #f_test_h = generate_data_2d.generate(samples=ntest,out_dim=N_max)
        #u_test_h = generate_data_2d.FD_AD_2d(f_test_h, EP, meshtype='Shishkin')
        u_test_h = np.load('2d_u_test.npy')
        
        f_test = torch.Tensor(f_test_h[:, ::dN,::dN])  # f_test.shape=ntrain*NS*NS
        f_test = np.array([generate_data_2d.grid_to_vec(y, NS) for y in f_test])
        dim = N_max
        grid = np.hstack((np.linspace(0, 1 - sigma_h, int((N_max - 1) / 2) + 1),
                            np.linspace(1 - sigma_h, 1, int((N_max - 1) / 2) + 1)[1:]))  #不同于1维，只在Shishkin网格上计算离散均方误差和离散l2范数
        grid_vec = np.zeros((dim ** 2, 2))
        for j in range(dim):
            for i in range(dim):
                grid_vec[j * NS + i] = [grid[i], grid[j]]
        N = f_test_h.shape[0]
        loc = np.zeros((N, dim**2, 2))
        res = np.zeros((N, dim**2, 1))
        f = np.zeros((N, NS**2))
        u_test_h0 = np.array([generate_data_2d.grid_to_vec(interpolate.interp2d(grid_h, grid_h, y)(grid, grid), dim) for y in u_test_h])
        for i in range(N):
            f[i] = f_test[i].flatten()
            loc[i] = grid_vec
            res[i] = u_test_h0[i]
        f_test = torch.Tensor(f).to(device)
        loc = torch.Tensor(loc).to(device)
        res = torch.Tensor(res).to(device)
        test_h_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(f_test, loc, res), batch_size=1,
                                                    shuffle=False)

        #pred_h = torch.zeros(u_train_h0.shape)
        index = 0
        test_mse = 0
        test_l2 = 0
        with torch.no_grad():
            for x, l, y in test_h_loader:
                out = model(x, l).view(-1)
                #[index] = out
                # mse = generate_data_2d.weighted_mse_loss(out.view(out.numel(), -1), y.view(y.numel(), -1), w.view(w.numel(), -1))
                mse = F.mse_loss(out.view(1, -1), y.view(1, -1), reduction='mean')
                l2 = myloss(out.view(1, -1), y.view(1, -1))
                test_mse += mse.item()
                test_l2 += l2.item()
                # index += 1
            # test_mse /= ntrain
            test_mse /= len(test_h_loader)
            test_l2 /= ntest
            print('test error on discrete Shishkin mesh: L2 = ', test_l2, 'MSE =', test_mse)
            

        # residual = pred_h-u_train_h
        # fig = plt.figure()
        # x_grid = np.linspace(0, 1, N_max)
        # for i in range(100):
        #    plt.plot(x_grid,residual[i].detach().numpy())
        # plt.show()

for NS in N_list:
    plt.figure()
    for EP in EP_list:
        plt.plot(loss_history["{} {}".format(NS, EP)], label="ep = {}".format(EP))
        plt.yscale("log")
    plt.title("N = {}".format(NS))
    plt.legend()
    plt.show()
'''
N=2**5+1
EP = 0.001
S=65
sigma_S = min(1 / 2, EP * np.log(S) / alpha)
sigma_N = min(1 / 2, EP * np.log(N) / alpha)
gridN = np.hstack(
        (np.linspace(0, 1 - sigma_N, int((N - 1) / 2) + 1), np.linspace(1 - sigma_N, 1, int((N - 1) / 2) + 1)[1:]))
gridS = np.hstack(
        (np.linspace(0, 1 - sigma_S, int((S - 1) / 2) + 1), np.linspace(1 - sigma_S, 1, int((S - 1) / 2) + 1)[1:])).reshape((-1,1))
f=torch.tensor(np.ones((S**2,N**2)), dtype=torch.float).to(device)
grid_vec = np.zeros((S ** 2, 2))
for j in range(S):
    for i in range(S):
        grid_vec[j * S + i] = [gridS[i], gridS[j]]
grid_vec=torch.Tensor(grid_vec).to(device)
y=model(f,grid_vec).view(-1)/100
y=y.cpu()
y=y.detach().numpy()
y=generate_data_2d.vec_to_grid(y,S)
gridS = np.hstack(
        (np.linspace(0, 1 - sigma_S, int((S - 1) / 2) + 1), np.linspace(1 - sigma_S, 1, int((S - 1) / 2) + 1)[1:])).reshape((-1,1))
f=np.ones((1,S,S))
y1=generate_data_2d.FD_AD_2d(f,EP)/100
xx,yy=np.meshgrid(gridS,gridS)
plt.figure()
ax3 = plt.axes(projection='3d')
ax3.plot_surface(xx,yy,y,cmap='rainbow')
# ax3.plot_surface(xx,yy,y1[0],cmap='rainbow')
plt.show()
'''
