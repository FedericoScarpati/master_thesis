import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, softmax
from torch_geometric.nn import TransformerConv, GCNConv, GraphSAGE


class Coloring_Transformer(nn.Module):
    '''
    Color a graph using Transformer convolution layers
    '''
    def __init__(self, input_dim, hidden_dim, n_heads = 4):
        super(Coloring_Transformer, self).__init__()

        # Lista di moduli TransformerConv con ReLU ogni strato
        self.transformer_layers = nn.ModuleList()
        # Aggiungi strati TransformerConv a gruppi di tre

        self.conv11 = TransformerConv(input_dim, hidden_dim, heads=n_heads, concat=False)
        self.conv12 = TransformerConv(hidden_dim, hidden_dim, heads=n_heads, concat=False)
        self.conv13 = TransformerConv(hidden_dim, hidden_dim, heads=n_heads, concat=False)

        self.conv21 = TransformerConv(hidden_dim, hidden_dim, heads=n_heads, concat=False)
        self.conv22 = TransformerConv(hidden_dim, hidden_dim, heads=n_heads, concat=False)
        self.conv23 = TransformerConv(hidden_dim, input_dim, heads=n_heads, concat=False)

    def forward(self, x, edge_index):
        '''
        Forward pass
        '''
        x = self.conv11(x, edge_index)
        x = relu(x)
        x = self.conv12(x, edge_index)
        x = relu(x)
        x = self.conv13(x, edge_index)
        x = relu(x)

        x_5 = x[:,:5]
        hidden = x[:,5:]
        x_5 = F.softmax(x_5, dim=1)
        x = torch.cat((x_5,hidden), dim=1)

        x = self.conv21(x, edge_index)
        x = relu(x)
        x = self.conv22(x, edge_index)
        x = relu(x)
        x = self.conv23(x, edge_index)
        x = relu(x)

        x = F.softmax(x, dim=1)

        return x_5, x


##############################################################################
class Coloring_GraphSAGE(nn.Module):
    '''
    Color a graph using GraphSAGE convolution layers.
    '''
    def __init__(self, input_dim, hidden_dim, dropout):
        super(Coloring_GraphSAGE, self).__init__()

        # Create a list of GraphSAGE modules, similar to the original Transformer layers
        self.sage_layers = nn.ModuleList([
            GraphSAGE(in_channels=input_dim, hidden_channels=1, num_layers=1, out_channels=hidden_dim),
            GraphSAGE(in_channels=hidden_dim, hidden_channels=1, num_layers=1,out_channels=hidden_dim, dropout=dropout),
            GraphSAGE(in_channels=hidden_dim, hidden_channels=1, num_layers=1,out_channels=hidden_dim, dropout=dropout),
            GraphSAGE(in_channels=hidden_dim, hidden_channels=1, num_layers=1,out_channels=hidden_dim, dropout=dropout),
            GraphSAGE(in_channels=hidden_dim, hidden_channels=1, num_layers=1,out_channels=hidden_dim, dropout=dropout),
            GraphSAGE(in_channels=hidden_dim, hidden_channels=1, num_layers=1, out_channels=input_dim)  # Last layer returns to the input dimension
        ])

    def forward(self, x, edge_index):
        '''
        Forward pass through GraphSAGE layers
        '''
        # First three GraphSAGE layers
        for i in range(3):
            x = self.sage_layers[i](x, edge_index)
            x = relu(x)
        
        # Split the output into two parts for different processing
        x_5 = x[:, :5]
        hidden = x[:, 5:]
        x_5 = softmax(x_5, dim=1)  # Apply softmax to the first part (for coloring probabilities)
        x = torch.cat((x_5, hidden), dim=1)  # Concatenate back together

        # Second set of three GraphSAGE layers
        for i in range(3, 6):
            x = self.sage_layers[i](x, edge_index)
            x = relu(x)

        # Apply softmax to the final output to normalize the last layer's output
        x = softmax(x, dim=1)

        return x_5, x
    


##############################################################
# Honorable mention

# Variable Embedding
class model2(torch.nn.Module):
    '''
    GNN colorazione grafi con q variabile fino a 8
    Embedding nodo ha dimensione 16 = 8 + 8
    Primi 8: random embedding
    Ultimi 8: encoding colore (1,1,1,1,1,0,0,0) = usa 5 colori per questo grafo
    
    Da allenare con termine di loss che penalizza numero di colori eccedente tipo (out[,:4] @ out[,4:]).sum()
    '''
    def __init__(self, num_features):
        super(model2, self).__init__()
        dim1=64 #128
        dim2=32 # 64
        self.conv11 = GCNConv(num_features, dim1)
        self.conv21 = GCNConv(dim1, dim2)
        self.conv31 = GCNConv(dim2, num_features)

        self.conv12 = GCNConv(num_features, dim1)
        self.conv22 = GCNConv(dim1, dim2)
        self.conv32 = GCNConv(dim2, num_features)

        self.conv13 = GCNConv(num_features, dim1)
        self.conv23 = GCNConv(dim1, dim2)
        self.conv33 = GCNConv(dim2, num_features)

        self.conv14 = GCNConv(num_features, dim1)
        self.conv24 = GCNConv(dim1, dim2)
        self.conv34= GCNConv(dim2, num_features)

        self.conv15 = GCNConv(num_features, dim1)
        self.conv25 = GCNConv(dim1, dim2)
        self.conv35 = GCNConv(dim2, num_features)

    def forward(self, data):

        x, edge_index = data.x, data.edge_index
        x = self.conv11(x, edge_index)
        x = relu(x)
        x = self.conv21(x, edge_index)
        x = relu(x)
        x = self.conv31(x, edge_index)
        part1 = x[:, :8]
        part1_softmax = softmax(part1, dim=1)
        x = torch.cat((part1_softmax, x[:, 8:]), dim=1)

        x = self.conv12(x, edge_index)
        x = relu(x)
        x = self.conv22(x, edge_index)
        x = relu(x)
        x = self.conv32(x, edge_index)
        part1 = x[:, :8]
        part1_softmax = softmax(part1, dim=1)
        x = torch.cat((part1_softmax, x[:, 8:]), dim=1)

        x = self.conv13(x, edge_index)
        x = relu(x)
        x = self.conv23(x, edge_index)
        x = relu(x)
        x = self.conv33(x, edge_index)
        part1 = x[:, :8]
        part1_softmax = softmax(part1, dim=1)
        x = torch.cat((part1_softmax, x[:, 8:]), dim=1)


        x = self.conv14(x, edge_index)
        x = relu(x)
        x = self.conv24(x, edge_index)
        x = relu(x)
        x = self.conv34(x, edge_index)
        part1 = x[:, :8]
        part1_softmax = softmax(part1, dim=1)
        x = torch.cat((part1_softmax, x[:, 8:]), dim=1)

        x = self.conv15(x, edge_index)
        x = relu(x)
        x = self.conv25(x, edge_index)
        x = relu(x)
        x = self.conv35(x, edge_index)
        part1 = x[:, :8]
        out = softmax(part1, dim=1)

        return out
