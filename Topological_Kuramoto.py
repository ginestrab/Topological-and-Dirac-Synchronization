
from numpy import *
from matplotlib.pyplot import *
import scipy.linalg as linalg
import scipy
import random
import matplotlib as plt
import math
import networkx as nx
from scipy.integrate import odeint
import scipy.integrate as it
from networkx.generators.classic import empty_graph, path_graph, complete_graph
from scipy.sparse import coo_matrix
import json
from scipy.cluster.hierarchy import dendrogram
import itertools
import copy
from functools import reduce
import numpy as np
from scipy.sparse.linalg import eigs
from collections import Counter
from scipy.linalg import null_space
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from collections import defaultdict
from tqdm import tqdm
import xgi




def create_node_edge_incidence_matrix(elist): #ASSUMES FIRST NODE IS INDEXED AS 1 EDGES ARE SORTED (n1,n2) with n1<n2
    num_edges = len(elist)
    data = [-1] * num_edges + [1] * num_edges
    row_ind = [e[0]-1 for e in elist] + [e[1]-1 for e in elist]
    col_ind = [i for i in range(len(elist))] * 2
    B1 = csc_matrix(
        (np.array(data), (np.array(row_ind), np.array(col_ind))), dtype=np.int8)
    return B1.toarray()

def get_B2(elist, tlist): 
    if len(tlist) == 0:
        return csc_matrix([], shape=(len(elist), 0), dtype=np.int8)

    elist_dict = {tuple(sorted(j)): i for i, j in enumerate(elist)}

    data = []
    row_ind = []
    col_ind = []
    for i, t in enumerate(tlist):
        e1 = t[[0, 1]]
        e2 = t[[1, 2]]
        e3 = t[[0, 2]]

        data.append(1)
        row_ind.append(elist_dict[tuple(e1)])
        col_ind.append(i)

        data.append(1)
        row_ind.append(elist_dict[tuple(e2)])
        col_ind.append(i)

        data.append(-1)

        row_ind.append(elist_dict[tuple(e3)])
        col_ind.append(i)

    B2 = csc_matrix((np.array(data), (np.array(row_ind), np.array(
        col_ind))), shape=(len(elist), len(tlist)), dtype=np.int8)
    return B2.toarray()

class Topological_Kuramoto:

    def __init__(self, coupling=1, dt=0.01, T=10, explosive=False, n_simplexes=None, natfreqs=None, init_angles=None):
        if n_simplexes is None and natfreqs is None:
            raise ValueError("n_nodes or natfreqs must be specified")
        self.dt = dt
        self.T = T
        self.coupling = coupling
        self.explosive = explosive
        if natfreqs is not None:
            self.natfreqs = natfreqs
            self.n_simplexes = len(natfreqs)
        else:
            self.n_simplexes = n_simplexes
            self.natfreqs = np.random.normal(size=self.n_simplexes)
        if init_angles is not None:
            self.init_angles = init_angles
        else:
            self.init_angles =2 * np.pi * np.random.random(size=self.n_simplexes)
 

    def derivative(self, angles_vec, t,  B, B_2, coupling, explosive):
        Rplus = abs(sum((np.e ** (1j * B_2.T@angles_vec)))/B_2.shape[1]) if explosive==True else 1
        Rminus = abs(sum((np.e ** (1j * B@angles_vec)))/B.shape[0])  if explosive==True else 1
        a=coupling*Rplus*B.T@np.sin(B@angles_vec)
        b = coupling*Rminus*B_2@np.sin(B_2.T@angles_vec)
        
        return self.natfreqs - a -b

    def integrate(self, angles_vec,  B, B_2):
        '''Updates all states by integrating state of all nodes'''
        # Compute it only once here and pass it to the derivative function
        t = np.linspace(0, self.T, int(self.T/self.dt))
        timeseries = odeint(self.derivative, angles_vec,
                            t, args=( B, B_2, self.coupling, self.explosive))
        return timeseries.T  # transpose for consistency (act_mat:node vs time)

    def run(self,  B, B_2=None,adj_mat=None,angles_vec=None):
 
        if angles_vec is None:
            angles_vec = self.init_angles

        return self.integrate(angles_vec, B, B_2)



class Topological_Dirac:

    def __init__(self, coupling=1, dt=0.01, T=10, n_simplexes=None, natfreqs=None, init_angles=None):
        if n_simplexes is None and natfreqs is None:
            raise ValueError("n_nodes or natfreqs must be specified")
        self.dt = dt
        self.T = T
        self.coupling = coupling

        if natfreqs is not None:
            self.natfreqs = natfreqs
        else:
            self.n_simplexes = n_simplexes
            self.natfreqs = np.random.normal(size=self.n_simplexes)
        if init_angles is not None:
            self.init_angles = init_angles
        else:
            self.init_angles = 2 * np.pi *np.random.random(size=self.n_simplexes)

    def derivative(self, angles_vec, t, D, L_S, gamma, z,coupling):
        gamma = gamma
        a = coupling*D@np.sin((D-z*gamma@L_S)@angles_vec)
        return self.natfreqs - a

    def integrate(self, angles_vec, D, L_S, gamma,z):
        '''Updates all states by integrating state of all nodes'''
        # Compute it only once here and pass it to the derivative function
        t = np.linspace(0, self.T, int(self.T/self.dt))
        timeseries = odeint(self.derivative, angles_vec,
                            t, args=(D, L_S, gamma, z,self.coupling))
        return timeseries.T  # transpose for consistency (act_mat:node vs time)

    def run(self, D, L_S, gamma, z,angles_vec=None):
        if angles_vec is None:
            angles_vec = self.init_angles
        D = D
        L_S = L_S
        gamma = gamma
        return self.integrate(angles_vec, D, L_S, gamma, z)


