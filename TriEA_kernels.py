import math
import numpy as np
from numba import cuda

@cuda.jit
def update_red_ea(spin, seed, T, J_nn):

    def random_uniform(x, y, z):
        seed[x, y, z] = np.int32((seed[x, y, z]*1664525 + 1013904223) % 2**31)
        return seed[x, y, z] / (2**31)

    def bvc (x):
        if x == spin.shape[0]:
            x = 0
        return x       

    def sum_nn(x, y, z):  # This adds spins of six neighbours instead of 4 subject to
        #many constraints characteristic of triangular lattices
        value = 0.
        if (y % 2 == 0):
            value += J_nn[x,y,0]*spin[x, bvc(y+1), z]
            value += J_nn[x-1, bvc(y+1),2]*spin[x-1, bvc(y+1), z]
            value += J_nn[x,y,2]*spin[x, y-1, z]
            value += J_nn[x-1,y-1,0]*spin[x-1, y-1, z]
        else:
            value += J_nn[x,y,0]*spin[bvc(x+1), bvc(y+1), z]
            value += J_nn[x,bvc(y+1),2]*spin[x, bvc(y+1), z]
            value += J_nn[x,y,2]*spin[bvc(x+1), y-1, z]
            value += J_nn[x,y-1,0]*spin[x, y-1, z]

        value += J_nn[x,y,1]*spin[bvc(x+1), y, z]
        value += J_nn[x-1,y,1]*spin[x-1, y, z]
        return value

    def calc(x, y, z):
        probs = random_uniform(x, y, z)
        if (probs < math.exp(2*spin[x, y, z]*sum_nn(x, y, z)/T[0])):
            spin[x, y, z] *= np.int8(-1)

    x, y, z = cuda.grid(3)
    p, q = x % 3, y % 2

    if x < spin.shape[0] and y < spin.shape[1] and z < spin.shape[2]:
        if (p == 0 and q == 0) or (p == 1 and q == 1):
            calc(x, y, z)

@cuda.jit
def update_blue_ea(spin, seed, T, J_nn):

    def random_uniform(x, y, z):
        seed[x, y, z] = np.int32((seed[x, y, z]*1664525 + 1013904223) % 2**31)
        return seed[x, y, z] / (2**31)

    def bvc (x):
        if x == spin.shape[0]:
            x = 0
        return x       

    def sum_nn(x, y, z):  # This adds spins of six neighbours instead of 4 subject to
        #many constraints characteristic of triangular lattices
        value = 0.
        if (y % 2 == 0):
            value += J_nn[x,y,0]*spin[x, bvc(y+1), z]
            value += J_nn[x-1, bvc(y+1),2]*spin[x-1, bvc(y+1), z]
            value += J_nn[x,y,2]*spin[x, y-1, z]
            value += J_nn[x-1,y-1,0]*spin[x-1, y-1, z]
        else:
            value += J_nn[x,y,0]*spin[bvc(x+1), bvc(y+1), z]
            value += J_nn[x,bvc(y+1),2]*spin[x, bvc(y+1), z]
            value += J_nn[x,y,2]*spin[bvc(x+1), y-1, z]
            value += J_nn[x,y-1,0]*spin[x, y-1, z]

        value += J_nn[x,y,1]*spin[bvc(x+1), y, z]
        value += J_nn[x-1,y,1]*spin[x-1, y, z]
        return value

    def calc(x, y, z):
        probs = random_uniform(x, y, z)
        if (probs < math.exp(2*spin[x, y, z]*sum_nn(x, y, z)/T[0])):
            spin[x, y, z] *= np.int8(-1)

    x, y, z = cuda.grid(3)
    p, q = x % 3, y % 2

    if x < spin.shape[0] and y < spin.shape[1] and z < spin.shape[2]:
        if (p == 1 and q == 0) or (p == 2 and q == 1):
            calc(x, y, z)

#Triangular EA Simulation
@cuda.jit
def update_green_ea(spin, seed, T, J_nn):

    def random_uniform(x, y, z):
        seed[x, y, z] = np.int32((seed[x, y, z]*1664525 + 1013904223) % 2**31)
        return seed[x, y, z] / (2**31)

    def bvc (x):
        if x == spin.shape[0]:
            x = 0
        return x       

    def sum_nn(x, y, z):  # This adds spins of six neighbours instead of 4 subject to
        #many constraints characteristic of triangular lattices
        value = 0.
        if (y % 2 == 0):
            value += J_nn[x,y,0]*spin[x, bvc(y+1), z]
            value += J_nn[x-1,bvc(y+1),2]*spin[x-1, bvc(y+1), z]
            value += J_nn[x,y,2]*spin[x, y-1, z]
            value += J_nn[x-1,y-1,0]*spin[x-1, y-1, z]
        else:
            value += J_nn[x,y,0]*spin[bvc(x+1), bvc(y+1), z]
            value += J_nn[x,bvc(y+1),2]*spin[x, bvc(y+1), z]
            value += J_nn[x,y,2]*spin[bvc(x+1), y-1, z]
            value += J_nn[x,y-1,0]*spin[x, y-1, z]

        value += J_nn[x,y,1]*spin[bvc(x+1), y, z]
        value += J_nn[x-1,y,1]*spin[x-1, y, z]
        return value

    def calc(x, y, z):
        probs = random_uniform(x, y, z)
        if (probs < math.exp(2*spin[x, y, z]*sum_nn(x, y, z)/T[0])):
            spin[x, y, z] *= np.int8(-1)

    x, y, z = cuda.grid(3)
    p, q = x % 3, y % 2

    if x < spin.shape[0] and y < spin.shape[1] and z < spin.shape[2]:
        if (p == 2 and q == 0) or (p == 0 and q == 1):
            calc(x, y, z)

@cuda.jit
def parallel_temper(T, seed, energy):
    z = cuda.grid(1)

    rand_n = 0 if np.float32(seed[0, 0, 0]/2**31) < 0.5 else 1
    ptr = 2*z + rand_n
    if ptr < energy.shape[0]-1:
        val0 = 1./T[ptr]
        val1 = 1./T[ptr+1]
        e0 = energy[ptr]
        e1 = energy[ptr+1]
        rand_unif = np.float32(seed[0, 1, z] / 2**31)
        arg = (e0 - e1)*(val0 - val1)
        if (arg < 0):
            if rand_unif < math.exp(arg):
                T[ptr] = 1/val1
                T[ptr+1] = 1/val0
        else:
            T[ptr] = 1/val1
            T[ptr+1] = 1/val0

@cuda.jit
def calc_energy_Triea (spin, energy, J_nn):
    def bvc (x):
        if x == spin.shape[1]:
            x = 0
        return x
    
    def sum_nn_part(x, y, z):  # This adds spins of six neighbours instead of 4 subject to
        #many constraints characteristic of triangular lattices
        value = 0.
        if (x % 2 == 0):
            value += J_nn[x,y,0]*spin[x, bvc(y+1), z]
            value += J_nn[x,y,2]*spin[x, y-1, z]
        else:
            value += J_nn[x,y,0]*spin[bvc(x+1), bvc(y+1), z]
            value += J_nn[x,y,2]*spin[bvc(x+1), y-1, z]

        value += J_nn[x,y,1]*spin[bvc(x+1), y, z]
        return value
    
    ener = 0
    z = cuda.grid (1)
    for x in range (spin.shape[0]):
        for y in range (spin.shape[1]):
            ener -= spin[x,y,z]*sum_nn_part(x,y,z)
    energy[z] = ener
