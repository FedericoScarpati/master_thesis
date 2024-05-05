import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, softmax
from torch_geometric.nn import GCNConv, GraphSAGE, global_max_pool, GraphConv, global_mean_pool, TransformerConv, global_max_pool



class Coloring_Transformer(nn.Module):
    '''
    Colora
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

        x = self.conv11(x, edge_index)
        x = relu(x)
        x = self.conv12(x, edge_index)
        x = relu(x)
        x = self.conv13(x, edge_index)
        x = relu(x)

        #x_5, hidden = x.split(5, dim=1)
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

#Fixed embedding
class model2_fixed_emb(torch.nn.Module):
    '''
    GNN Lorenzo
    Solo colorazione
    q fisso
    '''
    def __init__(self, num_features):
        super(model2_fixed_emb, self).__init__()
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
        x = softmax(x, dim = 1)

        x = self.conv12(x, edge_index)
        x = relu(x)
        x = self.conv22(x, edge_index)
        x = relu(x)
        x = self.conv32(x, edge_index)
        x = softmax(x, dim = 1)


        x = self.conv13(x, edge_index)
        x = relu(x)
        x = self.conv23(x, edge_index)
        x = relu(x)
        x = self.conv33(x, edge_index)
        x = softmax(x, dim = 1)


        x = self.conv14(x, edge_index)
        x = relu(x)
        x = self.conv24(x, edge_index)
        x = relu(x)
        x = self.conv34(x, edge_index)
        x = softmax(x, dim = 1)

        x = self.conv15(x, edge_index)
        x = relu(x)
        x = self.conv25(x, edge_index)
        x = relu(x)
        x = self.conv35(x, edge_index)
        x = softmax(x, dim = 1)

        return x
    
# Variable Embedding
class model2(torch.nn.Module):
    '''
    GNN solo colorazione
    Impara la dimensione embedding ammessa, dato q
    In: 8+8
    Out: 8, primi q ammessi
    Funge abbastanza
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
        part1_softmax = softmax(part1, dim=1)

        return part1_softmax
    

class colorableGNN(nn.Module):
    '''
    Gnn con classificazione
    fa schifo
    '''
    def __init__(self, in_channels, hidden_channels, fc_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels=in_channels, out_channels=hidden_channels)
        self.conv2 = GCNConv(in_channels=hidden_channels, out_channels=hidden_channels)
        self.conv3 = GCNConv(in_channels=hidden_channels, out_channels=fc_channels)
        self.fc1 = nn.Linear(in_features=fc_channels, out_features=in_channels)
        self.fc2 = nn.Linear(in_features=in_channels, out_features=2)

    def forward(self, batch):
        x = batch.x
        ei = batch.edge_index
        batchmat = batch.batch

        x = self.conv1(x,ei)
        x = relu(x)
        x = self.conv2(x,ei)
        x = relu(x)
        x = self.conv3(x,ei)
        x = relu(x)

        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)

        out = global_mean_pool(x, batchmat)
        out = F.softmax(out, dim=1)

        return out

# Prototipo GNN
class myGNN(nn.Module):
    '''
    prova GNN con embedding variabile
    solo coloring
    In: embedding 16= 8 + 8 encoding colore
    Out: embedding 8 di cui i primi q ammessi
    '''
    def __init__(self, num_layers, p_dropout):
        super().__init__()

        self.conv1 = GraphSAGE(in_channels=16, hidden_channels=16, num_layers=num_layers, out_channels=16)
        self.conv2 = GraphSAGE(in_channels=16, hidden_channels=16, num_layers=num_layers, out_channels=16)

        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(8, 8)

        self.drop1 = nn.Dropout(p_dropout)
        self.drop2 = nn.Dropout(p_dropout)

    def forward(self, data):
        x = data.x
        ei = data.edge_index

        x = self.conv1(x, ei).relu()
        x = self.drop1(x)
        x = self.fc1(x).relu()

        x = self.conv2(x, ei).relu()

        x = x[:, :8]


        x = self.fc2(x).relu()
        x = self.drop2(x)

        x = F.softmax(x, dim=1)

        return x

# Prototipo GNN
# con embedding dimensione variabile

class myGNN2(nn.Module):
    '''
    Colorazione e Classificazione
    Colorazione ok, classificazione schifo
    '''
    def __init__(self, in_channels, hid_dim, out_channels):
        super().__init__()
        self.conv1 = GraphConv(in_channels, hid_dim)
        self.conv2 = GraphConv(hid_dim, out_channels)
        self.fc1 = nn.Linear(out_channels, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, data):
        x = data.x
        ei = data.edge_index
        batch = data.batch

        x = self.conv1(x, ei).relu()
        x = self.conv2(x, ei).relu()

        mid = x.clone()
        mid = mid.split(8, 1)[0]

        mid = F.softmax(mid, dim=1)

        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = global_max_pool(x, batch)

        x = F.softmax(x, dim=1)
        return mid, x


class Coloring_Transformer_old(nn.Module):
    '''
    Colora
    '''

    def __init__(self, input_dim, hidden_dim, num_layers, n_heads=4):
        super(Coloring_Transformer, self).__init__()

        # Lista di moduli TransformerConv con ReLU ogni strato
        self.transformer_layers = nn.ModuleList()
        # Aggiungi strati TransformerConv a gruppi di tre
        for i in range(num_layers // 3):
            # Primo strato del gruppo
            self.transformer_layers.append(TransformerConv(input_dim, hidden_dim, heads=n_heads, concat=False))
            # Secondo strato del gruppo
            self.transformer_layers.append(TransformerConv(hidden_dim, hidden_dim, heads=n_heads, concat=False))
            # Terzo strato del gruppo
            self.transformer_layers.append(TransformerConv(hidden_dim, input_dim, heads=n_heads, concat=False))

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index

        # Passaggio attraverso i moduli TransformerConv con ReLU ogni strato
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x, edge_index)
            x = F.relu(x)  # Aggiungi ReLU tra ogni strato

            if (i + 1) % 3 == 0:  # Aggiungi softmax ogni tre strati
                x = F.softmax(x, dim=1)

        return x

class Classifier_Transformer(nn.Module):
    '''
    Colora e fa classificazione
    '''
    def __init__(self, input_dim, hidden_dim, num_layers, n_heads=3):
        super(Classifier_Transformer, self).__init__()

        # Lista di moduli TransformerConv con ReLU ogni strato
        self.coloring_layers = nn.ModuleList()
        # Aggiungi strati TransformerConv a gruppi di tre
        for i in range(num_layers // 3):
            # Primo strato del gruppo
            self.coloring_layers.append(TransformerConv(input_dim, hidden_dim, heads=n_heads, concat=False))
            # Secondo strato del gruppo
            self.coloring_layers.append(TransformerConv(hidden_dim, hidden_dim, heads=n_heads, concat=False))
            # Terzo strato del gruppo
            self.coloring_layers.append(TransformerConv(hidden_dim, hidden_dim, heads=n_heads, concat=False))

        self.conv_layer = TransformerConv(hidden_dim, hidden_dim, heads=n_heads, concat=False, dropout=0.2)

        self.MLP1 = nn.Linear(hidden_dim, 2)


    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index

        # Attempt Coloring
        for i, layer in enumerate(self.coloring_layers):
            x = layer(x, edge_index)
            x = F.relu(x)  # Aggiungi ReLU tra ogni strato

        mid = x[:,:5]
        mid = F.softmax(mid, dim=1)

        # Is colorable?
        x = self.conv_layer(x, edge_index)
        x = F.relu(x)

        x = self.MLP1(x)
        x = F.relu(x)

        x = F.softmax(x, dim=1)
        x = global_mean_pool(x, batch.batch)

        return mid, x

class BAD_GraphColoringTransformer(nn.Module):
    '''
    Ibrido trasformer e GCNConv layers
    fa schifo
    '''
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(BAD_GraphColoringTransformer, self).__init__()

        self.conv11 = GCNConv(input_dim, hidden_dim)
        self.conv12 = GCNConv(hidden_dim, hidden_dim)
        self.conv13 = GCNConv(hidden_dim, input_dim)

        self.conv21 = GCNConv(input_dim, hidden_dim)
        self.conv22 = GCNConv(hidden_dim, hidden_dim)
        self.conv23 = GCNConv(hidden_dim, input_dim)

        self.conv31 = GCNConv(input_dim, hidden_dim)
        self.conv32 = GCNConv(hidden_dim, hidden_dim)
        self.conv33 = GCNConv(hidden_dim, input_dim)

        # Lista di moduli TransformerConv con ReLU ogni strato
        self.transformer_layers = nn.ModuleList()
        # Aggiungi strati TransformerConv a gruppi di tre
        for i in range(num_layers // 3):
            # Primo strato del gruppo
            self.transformer_layers.append(TransformerConv(input_dim, hidden_dim, heads=1))

            # Secondo strato del gruppo
            self.transformer_layers.append(TransformerConv(hidden_dim, hidden_dim, heads=1))

            # Terzo strato del gruppo
            self.transformer_layers.append(TransformerConv(hidden_dim, input_dim, heads=1))


    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index

        x = self.conv11(x, edge_index)
        x = relu(x)
        x = self.conv12(x, edge_index)
        x = relu(x)
        x = self.conv13(x, edge_index)
        x = softmax(x, dim = 1)

        # Passaggio attraverso i moduli TransformerConv con ReLU ogni strato
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x, edge_index)
            x = F.relu(x)  # Aggiungi ReLU tra ogni strato

            if (i + 1) % 3 == 0:  # Aggiungi softmax ogni tre strati
                x = torch.softmax(x, dim=1)

        x = self.conv21(x, edge_index)
        x = relu(x)
        x = self.conv22(x, edge_index)
        x = relu(x)
        x = self.conv23(x, edge_index)
        x = softmax(x, dim=1)

        x_5, hidden = x.split(5,dim=1)
        x_5 = F.softmax(x_5,dim=1)

        x=torch.cat((x_5, hidden), dim=1)

        x = self.conv31(x, edge_index)
        x = relu(x)
        x = self.conv32(x, edge_index)
        x = relu(x)
        x = self.conv33(x, edge_index)
        x = softmax(x, dim = 1)

        return x_5, x
