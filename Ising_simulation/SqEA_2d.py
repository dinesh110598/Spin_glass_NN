import numpy as np
import math
from numba import cuda,jit,prange
import matplotlib.pyplot as plt

@cuda.jit
def update_black (spin, seed, T, J_nn):
    def random_uniform(x, y, z):
        seed[x, y, z] = np.int32((seed[x, y, z]*1664525 + 1013904223) % 2**31)
        return seed[x, y, z] / (2**31)

    def bvc(x):
        if x == spin.shape[0]:
            return 0
        else:
            return x

    def sum_nn (x, y, z):
        sum = 0
        sum += J_nn [x,y,0] * spin[bvc(x+1),y,z]
        sum += J_nn [x,y,1] * spin[x,bvc(y+1),z]
        sum += J_nn [x-1,y,0] * spin[x-1,y,z]
        sum += J_nn [x,y-1,1] * spin[x,y-1,z]
        return sum
    
    def calc(x, y, z):
        probs = random_uniform(x, y, z)
        if (probs < math.exp(-2*spin[x, y, z]*sum_nn(x, y, z)/T[z])):
            spin[x, y, z] *= np.int8(-1)

    x, y, z = cuda.grid(3)
    p, q = x%2, y%2
    if x<spin.shape[0] and y<spin.shape[1] and z<spin.shape[2]:
        if (p==0 and q==0) or (p==1 and q==1):
            calc (x, y, z)

@cuda.jit
def update_white(spin, seed, T, J_nn):
    def random_uniform(x, y, z):
        seed[x, y, z] = np.int32((seed[x, y, z]*1664525 + 1013904223) % 2**31)
        return seed[x, y, z] / (2**31)

    def bvc(x):
        if x == spin.shape[0]:
            return 0
        else:
            return x

    def sum_nn(x, y, z):
        sum = 0
        sum += J_nn[x, y, 0] * spin[bvc(x+1), y, z]
        sum += J_nn[x, y, 1] * spin[x, bvc(y+1), z]
        sum += J_nn[x-1, y, 0] * spin[x-1, y, z]
        sum += J_nn[x, y-1, 1] * spin[x, y-1, z]
        return sum

    def calc(x, y, z):
        probs = random_uniform(x, y, z)
        if (probs < math.exp(-2*spin[x, y, z]*sum_nn(x, y, z)/T[z])):
            spin[x, y, z] *= np.int8(-1)

    x, y, z = cuda.grid(3)
    p, q = x % 2, y % 2
    if x < spin.shape[0] and y < spin.shape[1] and z < spin.shape[2]:
        if (p == 1 and q == 0) or (p == 0 and q == 1):
            calc(x, y, z)

@cuda.jit
def adjust_temp (T, flag):
    z = cuda.grid(1)

    if flag[0] >= 0:
        val = T[np.int32(flag[0])]
        T[np.int32(flag[0])] = T[np.int32(flag[0])+1]
        T[np.int32(flag[0])+1] = val
    
    elif flag[0] == -2:
        T[z] = flag[1]

@jit(nopython=True)
def make_matrix(J_nn, shape, m):
    def bvc(x):
        if x == shape[0]:
            return 0
        else:
            return x

    Jnn_matrix = np.zeros((shape[0]**2, shape[0]**2), np.float32)
    for i in range(shape[0]):
        for j in range(shape[0]):
            for k in range(2):
                l = np.int32(not bool(k))
                Jnn_matrix[i*64 + j][bvc(i+l)*64 + bvc(j+k)] = J_nn[i][j][k]
                Jnn_matrix[bvc(i+l)*64 + bvc(j+k)][i*64 + j] = J_nn[i][j][k]
    return (Jnn_matrix)

class GPUsimulator():
    def __init__(self,lat_len, ens_size):
        self.shape = (lat_len, lat_len)
        self.m = ens_size
        self.tpb = (4, 4, 1)
        self.bpg = (math.ceil(lat_len/self.tpb[0]), math.ceil(lat_len/self.tpb[1]), self.m)
        self.vr_spin = cuda.to_device(np.ones(self.shape + (self.m,), dtype=np.int8))
        seed = np.random.randint(-10000, 10000, size=self.shape+(self.m,), dtype=np.int32)
        self.vr_seed = cuda.to_device(seed)             

    def equilib(self,temp): #yields a thermalised ensemble of EA systems
        J_nn = np.random.uniform(-1.732051, 1.732051, size=self.shape+(2,))
        Jnn_vr = cuda.to_device(J_nn)
        prod = self.shape[0]*self.shape[1]
        
        T = np.linspace(temp, temp+2.5, self.m)
        vr_temp = cuda.to_device(T)
        flag = cuda.to_device(np.array([-1, 0]))
        Jnn_matrix = make_matrix(J_nn, self.shape, self.m)
        for _ in range(250):
            for _ in range(5):
                update_black[self.bpg, self.tpb](self.vr_spin, self.vr_seed, vr_temp, Jnn_vr)
                update_white[self.bpg, self.tpb](self.vr_spin, self.vr_seed, vr_temp, Jnn_vr)
            spin = self.vr_spin.copy_to_host()
            T = vr_temp.copy_to_host()
            #Swap a random temperature and its successor in every step of this loop
            p = np.random.randint(self.m-1)
            e1 = np.reshape(spin[...,p], (1, prod)) @ (Jnn_matrix @
                                                np.reshape(spin[...,p], (prod, 1)))
            e2 = np.reshape(spin[...,p+1], (1, prod)) @ (Jnn_matrix @
                                                np.reshape(spin[...,p+1], (prod, 1)))
            arg = (e1 - e2)*(1/T[p] - 1/T[p+1])
            if np.random.uniform() < np.exp(arg):
                flag = cuda.to_device(np.array([p, 0]))
            else:
                flag = cuda.to_device(np.array([-1, 0]))
            adjust_temp[self.bpg[2],self.tpb[2]](vr_temp,flag)

        flag = cuda.to_device(np.array([-2.0, temp]))
        adjust_temp[self.bpg[2], self.tpb[2]](vr_temp,flag)
        for _ in range(200):
            update_black[self.bpg, self.tpb](
                self.vr_spin, self.vr_seed, vr_temp, Jnn_vr)
            update_white[self.bpg, self.tpb](
                self.vr_spin, self.vr_seed, vr_temp, Jnn_vr)

        spin = self.vr_spin.copy_to_host()
        energy = np.reshape(spin, (self.m, 1, prod)) @ (Jnn_matrix @
                                                np.reshape(spin, (self.m, prod, 1)))
        energy = np.reshape(energy, self.m)
        order = np.argsort(energy)
        output = spin[...,order]
        avgs = np.mean (output, axis=(0,1))
        output =  np.multiply (output, (2*(avgs > 0).astype(np.int8) - 1))
        return output


    def generate_train_data(self, train_len, phase='both', T_c=2.5):
        """Acceptable values of phase arg-'hot','cold' or 'both'"""
        t_lattice = []
        t_label = []
        for _ in range(train_len):
            if phase == 'both':
                dice = np.random.randint(0, 1+1)
                temp = 0.5*T_c if dice == 1 else 2*T_c
            elif phase == 'hot':
                temp = 2*T_c
            else:
                temp = 0.5*T_c
            t_lattice.append(self.equilib(temp).astype(np.float32))
            t_label.append(np.int32(temp < T_c))
        return np.stack(t_lattice), np.stack(t_label)


    def mag_data(self, low, high, step):
        _len = math.ceil((high-low) / step)
        data_x = np.empty(_len, dtype=np.float32)
        mags = np.empty(_len, dtype=np.float32)
        temp = np.float32(low)

        for i in range(_len):
            data_x[i] = temp
            mags[i] = np.mean(self.equilib(temp))
            temp += step
        _, ax = plt.subplots()
        ax.plot(data_x, mags, 'r-', linewidth=2)
        ax.set_ylim(-0.1, 1.0)
        ax.grid()
        plt.show()

    def neural_data(self, low, high, step, brain):
        avg_size = 20
        train_len = math.ceil(((high - low) / step))
        # Denotes size of train_lattice
        t_lattice = []
        temp = np.float32(low)
        for _ in range(train_len):
            for _ in range(avg_size):
                t_lattice.append(np.copy(self.equilib(temp).astype(np.float32)))
            temp += np.float32(step)
        t_lattice = np.stack (t_lattice)

        predictions = brain.predict(t_lattice)

        datax = np.empty(train_len, np.float32)
        datay = np.empty(train_len, np.float32)
        dummy = np.empty(avg_size, np.float32)
        temp = np.float32(low)
        for i in range(train_len):
            datax[i] = temp
            for j in range (avg_size):
                dummy[j] = predictions [i*avg_size + j][1]
            datay[i] = np.mean (dummy)
            temp += step
        _, ax = plt.subplots()
        ax.plot(datax, datay, 'r-', linewidth=2)
        ax.grid()
        plt.show()
