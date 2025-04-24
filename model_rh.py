import torch
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv
import torch_scatter as ts

class Net_mean_correction(torch.nn.Module):
    def __init__(self, NUM_NODE_FEATURES, EMBEDDING_SIZE, GNN_LAYERS, HIDDEN_FEATURES_SIZE):
        super(Net_mean_correction, self).__init__()

        print("GNN_LAYERS =", GNN_LAYERS)
        print("EMBEDDING_SIZE =", EMBEDDING_SIZE)
        print("HIDDEN_FEATURES_SIZE =", HIDDEN_FEATURES_SIZE)
        print(NUM_NODE_FEATURES, EMBEDDING_SIZE, "← input and embedding sizes")

        # Initial linear layer to embed input node features to a latent space ...since we have a lot of features here(226) 
        self.lin0 = torch.nn.Linear(NUM_NODE_FEATURES, EMBEDDING_SIZE, bias=False)

        # GatedGraphConv processes the node embeddings with edge information
        self.conv1 = GatedGraphConv(out_channels=HIDDEN_FEATURES_SIZE, num_layers=GNN_LAYERS)

        # Final linear layer to regress a scalar charge per node
        self.lin1 = torch.nn.Linear(HIDDEN_FEATURES_SIZE, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index  # node features and graph edges
        print("Input shape (x):", x.shape)

        # Step 1: Embed the raw features into latent dimension(reduction from 226 to 20)
        x_embed = torch.sigmoid(self.lin0(x))
        print("Embedded shape:", x_embed.shape)

        # Step 2: Message passing through the graph with GatedGraphConv
        x = self.conv1(x_embed, edge_index)
        x = F.relu(x)

        # Step 3: Predict initial per-node charge (µ_v), before mean correction
        x = self.lin1(x)  # shape = (num_nodes, 1)
        uncorrected_mu = x.clone()  # Save for output/debug

        # Step 4: Compute per-graph mean charge (to enforce neutrality later)

        # data.batch maps each node to its graph index (for mini-batch processing)
        mean_all = ts.scatter_mean(x, data.batch, dim=0)  # shape = (num_graphs, 1)

        # Step 5: Subtract graph-wise mean from each node's charge (mean correction) to enforce charge neutrality 
        for i in range(data.num_graphs):
            x[data.batch == i] = x[data.batch == i] - mean_all[i]

        # Output:
        # x.squeeze(1)          → final corrected per-node charge (shape = [num_nodes])
        # x_embed.squeeze(1)    → node embeddings (optional output, shape = [num_nodes, embed_dim])
        # None                  → reserved for predicted variances (not used in this variant)
        # uncorrected_mu.squeeze(1) → raw uncorrected charges
        return x.squeeze(1), x_embed, None, uncorrected_mu.squeeze(1)
