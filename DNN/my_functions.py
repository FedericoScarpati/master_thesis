import random
import sys
import io

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Dataset
from ortools.sat.python import cp_model



def get_variables_sizes():
    '''
    Print the size of all variables in the current context
    '''

    local_variables = globals()
    variables_sizes = [(name, sys.getsizeof(var)) for name, var in local_variables.items()]
    variables_sizes.sort(key=lambda x: x[1], reverse=True)

    for name, size in variables_sizes:
        print(f"Variable: {name}, Size: {size} bytes")


class my_PyG_dataset(Dataset):
    '''
    Object to store a list of PyG graphs
    '''
    def __init__(self, data_list):
        super(my_PyG_dataset, self).__init__()
        self.data_list = data_list

    def get(self, idx):
        return self.data_list[idx]

    def len(self):
        return len(self.data_list)

def random_embedding(G_list):
    '''
    Assign a random embedding to each node in the graph
    :param G_list: list of PyG graphs
    :type G_list: list
    :return: list of PyG graphs with random embeddings
    :rtype  list
    '''
    
    for graph in G_list:
        num_nodes = graph.num_nodes
        q = graph.q
        emb_np = np.zeros((num_nodes, q))
        for i in range(num_nodes):
            random_index = np.random.randint(q)
            emb_np[i, random_index] = 1

        # Embedding to torch
        emb = torch.tensor(emb_np, dtype=torch.float)

        graph.x = emb
    return G_list

def hard_color_assign(data):
    '''
    Assign a hard color to each node in the graph
    :param data: torch soft assignments
    :type data: torch.tensor
    :return: torch hard assignment
    :rtype  torch.tensor
    '''

    # Apply maxpooling to get the hard assignment
    maxpool_layer = nn.MaxPool1d(kernel_size=data.shape[1])

    result = maxpool_layer(data.unsqueeze(0).float())
    result = (data == result).float()
    result = result.squeeze(dim=0)

    return result


def hard_loss_compute(hard_color_assignment, edge_index):
    '''
    Compute the hard loss
    :param hard_color_assignment: torch hard assignments
    :type hard_color_assignment: torch.tensor
    :param edge_index: edge index
    :type edge_index: torch.tensor
    :return: hard loss
    :rtype  int
    '''
    loss = 0
    for u, v in edge_index.t():
        loss = loss + torch.mul(hard_color_assignment[u], hard_color_assignment[v]).sum()
        loss = loss.item()
    return loss


def count_parameters(model):
    '''
    Count the number of parameters in a model
    :param model: modello
    :return: numero parametri
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_network_structure(model):
    '''
    Get the structure of a model
    :param model: model
    :return: model structure
    '''
    buffer = io.StringIO()
    print(model, file=buffer)
    structure_str = buffer.getvalue()

    return structure_str

def perturbate_model(model, perturbation_factor=0.01):
    '''
    Perturbate the model parameters
    :param model: model
    :param perturbation_factor: perturbation factor
    '''
    
    for param in model.parameters():
        # Aggiungi una perturbazione casuale ai parametri
        perturbation = torch.randn_like(param) * perturbation_factor
        param.data.add_(perturbation)

def visualize_conflicts(graph):
    '''
    Nice plot of the graph with conflicting edges in red
    :param graph: graph
    '''
    
    # Get information from the graph
    edge_index = graph.edge_index
    node_features = hard_color_assign(graph.x)
    num_nodes = node_features.size(0)

    # Create a networkx graph and add nodes
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    # Add edges and color them based on conflict conditions
    for edge in edge_index.t().tolist():
        node1, node2 = edge
        color = 'black'

        if torch.mul(node_features[node1], node_features[node2]).sum().item() == 1:
            color = 'red'  # Red color for conflicting edges

        G.add_edge(node1, node2, color=color)

    # Extract edge colors for the plot
    edge_colors = [data['color'] for _, _, data in G.edges(data=True)]

    # Extract node colors for the plot
    node_colors = torch.argmax(node_features, dim=1).tolist()

    # Draw the graph
    pos = nx.spring_layout(G)  # Posizionamento dei nodi
    nx.draw(G, pos, with_labels=True, edge_color=edge_colors, font_color='black', node_color=node_colors, cmap=plt.cm.rainbow)

    # Draw red edges with a thicker line
    red_edges = [(node1, node2) for node1, node2, data in G.edges(data=True) if data['color'] == 'red']
    red_edge_widths = 2.0  # Imposta lo spessore degli archi rossi
    nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='red', width=red_edge_widths)

    plt.show(block=False)
    plt.pause(60)
    
##################################
# Functions for graph generation #
##################################

# Function to generate a planted graph
def MakePlanted(N, M, Q):

    NoverQ = N / Q
    degree = torch.zeros(N, dtype=torch.int)
    color  = torch.zeros(N, dtype=torch.int)
    adj = torch.zeros(N, N, dtype=torch.int)

    mapp = list(range(N))
    random.shuffle(mapp)

    neigh = []  # lists of neighboring nodes, one per node
    for k in range(N):
        neigh.append([])
        color[mapp[k]] = int(k/NoverQ)

    graph = torch.zeros(2, M, dtype=torch.int64)

    for i in range(M):
        var1 = random.random()
        var1 *= N
        var1 = int(var1)
        var2 = var1
        while int(var1 / NoverQ) == int(var2 / NoverQ):
            var2 = random.random()
            var2 *= N
            var2 = int(var2)
        var1 = mapp[var1]
        var2 = mapp[var2]
        graph[0][i] = var1
        graph[1][i] = var2


        neigh[var1].append(var2)
        neigh[var2].append(var1)

        degree[var1] += 1
        degree[var2] += 1

    # graph is an edges list
    return graph, color, mapp, neigh, degree


# Function to generate a random graph
def MakeRandom(N, M):
    degree = torch.zeros(N, dtype=torch.int)
    adj = torch.zeros(N, N, dtype=torch.int)

    neigh = [] 
    for _ in range(N):
        neigh.append([])

    graph = torch.zeros(2, M, dtype=torch.int64)

    # Loop over M edges
    for j in range(M):
        var1 = random.random()
        var1 *= N
        var1 = int(var1)
        var2 = var1
        while (var2 == var1):
            var2 = random.random()
            var2 *= N
            var2 = int(var2)
        graph[0][j] = var1
        graph[1][j] = var2

        neigh[var1].append(var2)
        neigh[var2].append(var1)

        adj[var1][var2] = 1
        adj[var2][var1] = 1

        degree[var1] += 1
        degree[var2] += 1

    return graph, neigh, degree, adj

# Function to solve random graphs
def cp_solve_graph(edges_list, q):
    n_nodes = edges_list.max().item() + 1
    nodes = []

    # Create the model
    model = cp_model.CpModel()

    # Include variables and constraints
    for i in range(n_nodes):
        node_var = model.NewIntVar(0, q-1, f"node{i}")
        nodes.append(node_var)

    for i in range(edges_list.shape[0]):
        n1 = edges_list[i,0].item()
        n2 = edges_list[i,1].item()
        model.Add(nodes[n1] != nodes[n2])

    # Solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # Get the solution
    if status == cp_model.OPTIMAL:
        is_colorable = True

        node_colors = [solver.Value(node) for node in nodes]
    else:
        is_colorable = False

        node_colors = None
    return is_colorable, node_colors
