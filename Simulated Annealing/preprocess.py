# Training loop per sola colorazione
# dei grafi, NO predittore

import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_undirected
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
from tabulate import tabulate
import pickle
import torch.optim as optim
import itertools

from my_functions import random_embedding, hard_loss_compute, hard_color_assign
from GNNs import Coloring_Transformer

NUM_HEADS = 3
q = 5

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


from GNNs import Coloring_Transformer

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    #print('I will use CUDA')
else:
    device = torch.device('cpu')
    #print('I will use CPU')

datadir = os.getcwd()
#print('Current directory: ', datadir)

def create_graph_from_edges(edge_list):
    # Converti l'elenco degli edge in tensori PyTorch
    edge_index = torch.tensor(edge_list, dtype=torch.long)
    # Crea il grafo
    graph = Data(edge_index=to_undirected(edge_index.t().contiguous()))
    return graph

G_list = []  # Lista per contenere i grafi

# Apri e leggi il file dei grafi
with open('generated_graphs.txt', 'r') as file:
    for line in file:
        edges = line.strip().split()  # Separa gli edge
        edge_list = []
        for i in range(0, len(edges), 2):
            edge_list.append((int(edges[i]), int(edges[i+1])))
        # Crea un grafo da questa riga del file (elenco di edge) e aggiungilo alla lista
        G_list.append(create_graph_from_edges(edge_list))

for graph in G_list:
    graph.q = q
    graph.num_nodes = graph.edge_index.max().item() + 1
    
# Ora G_list contiene tutti i grafi letti dal file
#print(f"Totale grafi letti: {len(G_list)}")
G_list = random_embedding(G_list)


net_instance = Coloring_Transformer(5, 32, NUM_HEADS)
#load saved model
model_name = 'planted_2_loss_6x3_32hidd_transfALPHA0.95_BETA0.4_lr0.001.pth'
net_instance.load_state_dict(torch.load(model_name))

# 
with open('graph_colors.txt', 'w') as file:
    
    for graph in G_list:
        _ , output = net_instance(graph.x, graph.edge_index)
        hard_color_assign = torch.argmax(output, dim=1)
        
        hard_energy = 0
        for i in range(len(graph.edge_index[0])):
            if hard_color_assign[graph.edge_index[0][i]] == hard_color_assign[graph.edge_index[1][i]]:
                hard_energy += 1
        hard_energy = hard_energy / 2
        #print('Hard energy:', hard_energy)
        # Converti hard_color_assign in una stringa con i valori separati da spazi
        color_string = ' '.join(map(str, hard_color_assign.tolist()))
        file.write(color_string + '\n')
file.close()