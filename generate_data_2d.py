

import torch
import numpy as np
from scipy import interpolate
from sklearn import gaussian_process as gp

#-eps*(u_xx+u_yy)+p1*u_x+p2*u_y+q*u=f
def FD_AD_2d(f, epsilon, meshtype='Shishkin'):
    def p1(x,y):
        return 1  # p(x)>=alpha=1
    def p2(x,y):
        return 1
    def q(x,y):
        return 1
    alpha = 1 #p1和p2共同的下界


    N = f.shape[-1]
    Nd = f.shape[0]
    sigma = min(1 / 2, epsilon * np.log(N) / alpha)
    grid = np.linspace(0, 1, N)
    gridS = np.hstack(
        (np.linspace(0, 1 - sigma, int((N - 1) / 2) + 1), np.linspace(1 - sigma, 1, int((N - 1) / 2) + 1)[1:]))
    fS=np.zeros((Nd,N,N))
    h1 = (1 - sigma) / ((N - 1) / 2)
    h2 = sigma / ((N - 1) / 2)
    yS = np.zeros((Nd, N, N))
    y = np.zeros((Nd, N, N))

    for k in range(Nd):

        U = np.zeros(((N-2)**2,(N-2)**2))
        def each_row_of_U(i, hx_1, hx_2, hy_1, hy_2, x, y):
            U[i, i] =2*epsilon/hx_1/hx_2+2*epsilon/hy_1/hy_2+p1(x,y)/hx_1+p2(x,y)/hy_1+q(x,y)
            if i%(N-2)>0:
                U[i, i-1] =-2*epsilon/hx_1/(hx_1+hx_2)-p1(x,y)/hx_1
            if i-(N-2)>=0:
                U[i, i-(N-2)] =-2*epsilon/hy_1/(hy_1 + hy_2)-p2(x, y)/hy_1
            if i%(N-2)<N-3:
                U[i, i+1] =-2*epsilon/hx_2/(hx_1 + hx_2)
            if i+N-2<(N-2)**2:
                U[i, i+N-2] =-2*epsilon/hy_2/(hy_1+hy_2)
        Nm = int((N-3)/2)  # here N must be odd
        for j in range(Nm):
            y_j = (j + 1) * h1
            for i in range(Nm):
                n=j*(N-2)+i
                x_i =(i+1)*h1
                each_row_of_U(n,h1,h1,h1,h1,x_i,y_j)
            n=j*(N-2)+Nm
            x_i=(Nm+1)*h1
            each_row_of_U(n,h1,h2,h1,h1,x_i,y_j)
            for i in range(Nm+1,N-2):
                n = j*(N-2)+i
                x_i=(Nm+1)*h1+(i-Nm)*h2
                each_row_of_U(n,h2,h2,h1,h1,x_i,y_j)
        y_j = (Nm + 1) * h1
        for i in range(Nm):
            n = Nm*(N-2)+i
            x_i = (i+1)*h1
            each_row_of_U(n,h1,h1,h1,h2,x_i,y_j)
        n = Nm * (N - 2) + Nm
        x_i = (Nm + 1) * h1
        each_row_of_U(n,h1,h2,h1,h2,x_i,y_j)
        for i in range(Nm+1, N-2):
            n = Nm*(N-2)+i
            x_i =(Nm+1)*h1+(i-Nm)*h2
            each_row_of_U(n,h2,h2,h1,h2,x_i,y_j)
        for j in range(Nm+1,N-2):
            y_j =(Nm+1)*h1+(j-Nm)*h2
            for i in range(Nm):
                n=j*(N-2)+i
                x_i =(i+1)*h1
                each_row_of_U(n,h1,h1,h2,h2,x_i,y_j)
            n=j*(N-2)+Nm
            x_i=(Nm+1)*h1
            each_row_of_U(n,h1,h2,h2,h2,x_i,y_j)
            for i in range(Nm+1,N-2):
                n = j*(N-2)+i
                x_i=(Nm+1)*h1+(i-Nm)*h2
                each_row_of_U(n,h2,h2,h2,h2,x_i,y_j)
        B = np.zeros(((N - 2)**2,1))
        fS[k] = interpolate.interp2d(grid, grid, f[k])(gridS, gridS)  # 默认为线性插值
        B[:] = grid_to_vec(fS[k],N-2)
        X=np.linalg.solve(U, B).flatten()
        yS[k]=vec_to_grid(X,N)

        y[k] = interpolate.interp2d(gridS, gridS,yS[k])(grid,grid)
    if meshtype == 'Shishkin':
        return yS*100
    else:
        return y*100
def vec_to_grid(x,N): #返回size为(N,N)的数组
    res = np.zeros((N, N))
    if N**2==x.shape[0]:
        for i in range(N):
            res[i]=x[i*N:(i+1)*N].T
    elif N**2>x.shape[0]:
        for i in range(1,N-1):
            res[i,1:-1]=x[(i-1)*(N-2):i*(N-2)].T
    else:
        for i in range(N):
            res[i]= x[(i+1)*(N+2)+1:(i+2)*(N+2)-1].T
    return res
def grid_to_vec(X,N): #返回size为(N**2,1)的数组
    n=X.shape[0]
    res = np.zeros((N ** 2, 1))
    if n==N:
        for i in range(N):
            res[i*N:(i+1)*N]=X[i][:,None]
    elif n==N+2:
        for i in range(N):
            res[i*N:(i+1)*N]=X[i+1,1:-1][:,None]
    elif N==n+2:
        for i in range(1,N-1):
            res[i*N+1:(i+1)*N-1]=X[i-1][:,None]
    return res
def generate(samples=10, begin=0, end=1, random_dim=11, out_dim=101, length_scale=1, interp="cubic", A=0):
    space = GRF(begin, end, length_scale=length_scale, N=random_dim, interp=interp)
    features = space.random(samples, A)
    features=np.array([vec_to_grid(y,N=random_dim) for y in features])
    x_grid = np.linspace(begin, end, out_dim)
    x_data = space.eval_u(features, x_grid,x_grid)
    return x_data  # X_data.shape=(samples,out_dim,out_dim)，每一行表示一个GRF在meshgrid上的取值，共有samples个GRF


class GRF(object):
    def __init__(self, begin=0, end=1, length_scale=1, N=1000, interp="cubic"):
        self.N = N
        self.interp = interp
        self.x = np.linspace(begin, end, num=N)
        self.z=np.zeros((self.N**2,2))
        for j in range(self.N):
            for i in range(self.N):
                self.z[j*self.N+i]=[self.x[i],self.x[j]]
        self.K=np.exp(-0.5*self.distance_matrix(self.z,length_scale))
        self.L = np.linalg.cholesky(self.K + 1e-12 * np.eye(self.N**2))
    def distance_matrix(self,x,length_scale):
        n=x.shape[0]
        grid=np.zeros((n,n))
        for i in range(n):
            for j in range(i):
                grid[i][j]=((x[i][0]-x[j][0])**2+(x[i][1]-x[j][1])**2)/length_scale**2
                grid[j][i]=grid[i][j]
        return grid


    def random(self, n, A):
        u = np.random.randn(self.N**2, n)
        return np.dot(self.L, u).T + A


    def eval_u(self, ys, x,y):
        res = np.zeros((ys.shape[0], x.shape[0],x.shape[0]))
        for i in range(ys.shape[0]):
            res[i] = interpolate.interp2d(self.x,self.x, ys[i], kind=self.interp, copy=False)(
                x,y)
        return res

def weighted_mse_loss(y_pred, y_true, weights):
    # 自定义加权MSE损失函数
    squared_errors = torch.square(y_pred - y_true)
    weighted_errors = squared_errors * weights
    return torch.sum(weighted_errors)

# loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        if y.size()[-1] == 1:
            eps = 0.00001
        else:
            eps = 0
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / (y_norms + eps))
            else:
                return torch.sum(diff_norms / (y_norms + eps))

        return diff_norms / (y_norms + eps)

    def __call__(self, x, y):
        return self.rel(x, y)