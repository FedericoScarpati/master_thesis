# preambolo
import math
import os
import torch
import numpy as np

import networkx as nx
from torch_geometric.utils import from_networkx


from utility_functions import (
    MakeRandom, save_edges, save_colors, cp_solve_graph
)

def torch_geometric_format(graph_list, c):
    data_list = []
    for graph in graph_list:
        Gnx = nx.Graph()
        Gnx.add_edges_from(graph.numpy()) #transpose needed?
        loops = list(nx.selfloop_edges(Gnx))
        Gnx.remove_edges_from(loops)
        G = from_networkx(Gnx)
        G.c = c
        G.q = 5
        data_list.append(G)
    return data_list

wd = os.getcwd()
wd = os.path.join(wd, 'random_graphs')
os.makedirs(wd, exist_ok=True)
print('Salvo i grafi in directory', wd)


# definisco iperparametri

NOCOL_ONLY = False

N = [80]
C = np.arange(5, 15, 0.25).tolist()
#C = [10.0, 10.25, 10.5, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16]
Q = [5]
num_graphs = 50
patience = num_graphs  # kill program and save after patience consecutive iterations without a new col/nocol graph
max_attempts = num_graphs*10 # kill program and save after max_attempts graphs generated
col_list = []
nocol_list = []

if NOCOL_ONLY:
    print('>>> GENERO SOLO GRAFI NON COLORING')

for n in N:
    for c in C:
        for q in Q:
            print('N:', n, 'C:', c, 'Q:', q)
            #print('Max iter:', max_attempts)

            m = int(c * n / 2)
            max_graphs = math.comb(math.comb(n, 2), m)

            if num_graphs > max_graphs:
                print('Stai generando piu grafi di quanti ne esistano in questa configurazione:')
                print('N:', n, 'C:', c, 'Q:', q)
                print('Numero grafi:', num_graphs, 'MAX:', max_graphs)
                if max_graphs == 0:
                    continue

            graph_COL_list = []
            colors_list = []
            graph_NOCOL_list = []

            num_col = 0
            num_nocol = 0
            attempts = 0
            wait_col = 0
            wait_nocol = 0

            # Generate graphs
            while (num_col < num_graphs) and attempts < max_attempts:
                attempts += 1
                if attempts % 100 == 0:
                    print(f'Iter:{attempts} COL:{num_col} NOCOL:{num_nocol}')

                graph_edges, _, _, _ = MakeRandom(n, m)
                graph_edges = graph_edges.t()

                is_colorable, solution = cp_solve_graph(graph_edges, q)

                if is_colorable:
                    wait_col = 0
                    wait_nocol += 1
                else:
                    wait_nocol = 0
                    wait_col += 1

                if wait_col > patience or wait_nocol > patience:
                    print('EARLY STOP\nwait COL:', wait_col, 'NOCOL:', wait_nocol)
                    attempts = max_attempts

                if is_colorable and num_col < num_graphs:
                    graph_COL_list.append(graph_edges)
                    colors_list.append(torch.tensor(solution))
                    num_col += 1
                elif not is_colorable and num_nocol < num_graphs:
                    graph_NOCOL_list.append(graph_edges)
                    num_nocol += 1

                if NOCOL_ONLY and num_nocol == num_graphs:
                    attempts = max_attempts

                if attempts == max_attempts:
                    print('MAX ATTEMPTS config \n''N:', n, 'C:', c, 'Q:', q)
                    print('colorable:', num_col, 'non-colorable:', num_nocol)
            
            print(f'c:{c} col:{num_col}')
            #Transform to pytorch geometric format
            print('Elaborating graphs...')
            if not NOCOL_ONLY:
                graph_COL_list = torch_geometric_format(graph_COL_list, c)
            graph_NOCOL_list = torch_geometric_format(graph_NOCOL_list, c)
            
            col_list.extend(graph_COL_list)
            nocol_list.extend(graph_NOCOL_list)
            
            filename = f"N{n}rand_NOCOL.pt"
            torch.save(nocol_list, filename)
            filename = f"N{n}rand_COL.pt"
            torch.save(col_list, filename)
            
if not NOCOL_ONLY:
    filename = f"N{n}rand_COL.pt"
    torch.save(col_list, filename)
    #path = os.path.join(wd, filename)
    #save_edges(col_list, path)

    #filename = 'solution_' + filename
    #path = os.path.join(wd, filename)
    #save_colors(colors_list, path)


#path = os.path.join(wd, filename)
#save_edges(graph_NOCOL_list, path)

print('Fatto')
