import math
import os
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.utils import from_networkx

from utility_functions import (
    MakePlanted,
)
from my_functions import MakePlanted

wd = os.getcwd()
print('Working directory:', wd)

# Parameters
N = [20]
C = np.arange(10, 12, 0.25).tolist()
Q = [5]
num_graphs = 200
name = f'TESTG_list_N{N[0]}.pt'

G_list = []
for n in N:
    for c in C:
        for q in Q:

            m = int(c*n/2)
            max_graphs = math.comb(math.comb(n,2),m)

            if (num_graphs > max_graphs):
                print('Stai generando piu grafi di quanti ne esistano in questa configurazione:')
                print('N: ', n, 'C: ',c, 'Q: ',q)
                print('Numero grafi:',num_graphs, 'MAX:', max_graphs)
                if max_graphs == 0:
                    continue

            graph_list = []
            colors_list = []

            # Create nx graphs list
            for _ in tqdm(range(num_graphs), desc='Processing graphs'):
                graph_edges, graph_colors, _, _, _ = MakePlanted(n,m,q)
                graph_edges = graph_edges.t()
                
                current_graph = nx.Graph()
                current_graph.add_edges_from(graph_edges.numpy())

                graph_list.append(current_graph)
                colors_list.append(graph_colors)
            
            # Transform graph list to pytorch geometric format
            for Gnx, solution in zip(graph_list, colors_list):
                # remove self loops
                loops = list(nx.selfloop_edges(Gnx))
                Gnx.remove_edges_from(loops)

                G = from_networkx(Gnx)

                # Add attributes c and q to the graph
                G.c = c
                G.q = q
                G.sol = solution
                
                G_list.append(G)

        print(f'Done c {c}')

N = N[0]
torch.save(G_list, name)
print('Done!')