import torch
from torch_geometric.data import HeteroData
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.loader import NeighborLoader


num_eoa_addresses = 100
num_token_contracts = 50
num_transactions = 200
num_transfers = 150
num_dexes = 20
num_loan_contracts = 10
num_lp = 10
num_features = 16

data = HeteroData()

data['EOA Address'].x = torch.randn(num_eoa_addresses, num_features)
data['Token Contract'].x = torch.randn(num_token_contracts, num_features)
data['Transaction'].x = torch.randn(num_transactions, num_features)
data['Transfer'].x = torch.randn(num_transfers, num_features)
data['DEX'].x = torch.randn(num_dexes, num_features)
data['Loan Contract'].x = torch.randn(num_loan_contracts, num_features)
data['Liquidity Provider'].x = torch.rand(num_lp, num_features)


# CASE 1: A transfers Ethereum to B
num_edges_case_1 = 50  # Number of Ethereum transfers between EOA
senders = torch.randint(0, num_eoa_addresses, (num_edges_case_1,))
receivers = torch.randint(0, num_eoa_addresses, (num_edges_case_1,))

data['EOA Address', 'sends', 'Transaction'].edge_index = torch.stack([senders, torch.arange(num_edges_case_1)])
data['Transaction', 'sent_to', 'EOA Address'].edge_index = torch.stack([torch.arange(num_edges_case_1), receivers])
data['Transaction', 'contains', 'Transfer'].edge_index = torch.stack([torch.arange(num_edges_case_1), torch.arange(num_edges_case_1)])
data['EOA Address', 'sends', 'Transfer'].edge_index = torch.stack([senders, torch.arange(num_edges_case_1)])
data['Transfer', 'sent_to', 'EOA Address'].edge_index = torch.stack([torch.arange(num_edges_case_1), receivers])



# CASE 2: A transfer a token to B
num_edges_case_2 = 75  # Number of token transfers via token contracts
senders = torch.randint(0, num_eoa_addresses, (num_edges_case_2,))
receivers = torch.randint(0, num_eoa_addresses, (num_edges_case_2,))
token_contracts = torch.randint(0, num_token_contracts, (num_edges_case_2,))

data['EOA Address', 'sends', 'Transaction'].edge_index = torch.stack([senders, torch.arange(num_edges_case_2)])
data['Transaction', 'sent_to', 'Token Contract'].edge_index = torch.stack([torch.arange(num_edges_case_2), token_contracts])
data['Transaction', 'contains', 'Transfer'].edge_index = torch.stack([torch.arange(num_edges_case_2), torch.arange(num_edges_case_2)])
data['Token Contract', 'includes', 'Transfer'].edge_index = torch.stack([token_contracts, torch.arange(num_edges_case_2)])
data['Transfer', 'sent_to', 'EOA Address'].edge_index = torch.stack([torch.arange(num_edges_case_2), receivers])



# Case 3: A trades token x for token y on a DEX
num_edges_case_3 = 40  # Number of trades
senders = torch.randint(0, num_eoa_addresses, (num_edges_case_3,))
dexes = torch.randint(0, num_dexes, (num_edges_case_3,))
lp_nodes = torch.randint(0, num_lp, (num_edges_case_3,))
receivers = torch.randint(0, num_eoa_addresses, (num_edges_case_3,))
token_x = torch.randint(0, num_token_contracts, (num_edges_case_3,))
token_y = torch.randint(0, num_token_contracts, (num_edges_case_3,))

data['EOA Address', 'sends', 'Transaction'].edge_index = torch.stack([senders, torch.arange(num_edges_case_3)])
data['Transaction', 'sent_to', 'DEX'].edge_index = torch.stack([torch.arange(num_edges_case_3), dexes])
data['Transaction', 'contains', 'Transfer'].edge_index = torch.stack([torch.arange(num_edges_case_3), torch.arange(num_edges_case_3)])
data['Transfer', 'includes', 'Token Contract'].edge_index = torch.stack([torch.arange(num_edges_case_3), token_x])
data['Transfer', 'sent_to', 'Liquidity Provider'].edge_index = torch.stack([torch.arange(num_edges_case_3), lp_nodes])
data['Liquidity Provider', 'sends', 'Transfer'].edge_index = torch.stack([lp_nodes, torch.arange(num_edges_case_3)])

# Add edges for "Transfer includes Token Y"
data['Transfer', 'includes', 'Token Contract'].edge_index = torch.cat([
    data['Transfer', 'includes', 'Token Contract'].edge_index,  # Keep existing
    torch.stack([torch.arange(num_edges_case_3), token_y])       # Add Token Y inclusion
], dim=1)

# Add edges for "Transfer sent_to B (Token Y to Receiver)"
data['Transfer', 'sent_to', 'EOA Address'].edge_index = torch.cat([
    data['Transfer', 'sent_to', 'EOA Address'].edge_index,  # Keep existing
    torch.stack([torch.arange(num_edges_case_3), receivers]) # Add new transfers to receiver
], dim=1)



# Case 4: A borrows from a loan contract 
num_edges_case_4 = 25  # Number of borrowing transactions
borrowers = torch.randint(0, num_eoa_addresses, (num_edges_case_4,))
loan_contracts = torch.randint(0, num_loan_contracts, (num_edges_case_4,))
lp_nodes = torch.randint(0, num_lp, (num_edges_case_4,))
tokens = torch.randint(0, num_token_contracts, (num_edges_case_4,))

data['EOA Address', 'sends', 'Transaction'].edge_index = torch.stack([borrowers, torch.arange(num_edges_case_4)])
data['Transaction', 'sent_to', 'Loan Contract'].edge_index = torch.stack([torch.arange(num_edges_case_4), loan_contracts])
data['Transaction', 'contains', 'Transfer'].edge_index = torch.stack([torch.arange(num_edges_case_4), torch.arange(num_edges_case_4)])
data['Transfer', 'includes', 'Token Contract'].edge_index = torch.stack([torch.arange(num_edges_case_4), tokens])
data['Transfer', 'sent_to', 'EOA Address'].edge_index = torch.stack([torch.arange(num_edges_case_4), borrowers])
data['Liquidity Provider', 'sends', 'Transfer'].edge_index = torch.stack([lp_nodes, torch.arange(num_edges_case_4)])




# Define the GraphSAGE model
class HeteroGraphSAGE(torch.nn.Module):
    def __init__(self, metadata, in_channels, hidden_channels, out_channels):
        super().__init__()
        # First layer: Aggregates input features to hidden features
        self.conv1 = HeteroConv(
            {
                edge_type: SAGEConv(in_channels, hidden_channels)
                for edge_type in metadata['edge_types']
            },
            aggr='sum',
        )
        # Second layer: Aggregates hidden features to output features
        self.conv2 = HeteroConv(
            {
                edge_type: SAGEConv(hidden_channels, out_channels)
                for edge_type in metadata['edge_types']
            },
            aggr='sum',
        )

    def forward(self, x_dict, edge_index_dict):
        # First layer
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}  
        # Second layer
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict

# Train/validation/test sets
for node_type in data.node_types:
    num_nodes = data[node_type].num_nodes
    data[node_type].train_mask = torch.rand(num_nodes) < 0.8
    data[node_type].val_mask = (torch.rand(num_nodes) >= 0.8) & (torch.rand(num_nodes) < 0.9)
    data[node_type].test_mask = torch.rand(num_nodes) >= 0.9

# Create a NeighborLoader for sampling mini-batches
train_loader = NeighborLoader(
    data,
    num_neighbors={key: [15, 10] for key in data.edge_types},  
    batch_size=64,
    input_nodes=('EOA Address', data['EOA Address'].train_mask), 
)

# Training the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HeteroGraphSAGE(
    metadata=data.metadata(),  
    in_channels=num_features, # Input feature size 
    hidden_channels=32, # Hidden layer size 
    out_channels=2,  # e.g. binary classification
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(10):  
    model.train()
    total_loss = 0

    for batch in train_loader:
        # Move batch to device
        batch = batch.to(device)

        # Forward pass
        out = model(batch.x_dict, batch.edge_index_dict)

        # Compute loss (e.g., cross-entropy for classification)
        loss = F.cross_entropy(
            out['EOA Address'][batch['EOA Address'].train_mask],
            batch['EOA Address'].y[batch['EOA Address'].train_mask],
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {total_loss:.4f}')

# Evaluate the model
model.eval()
correct = 0
total = 0

for batch in train_loader:
    batch = batch.to(device)
    out = model(batch.x_dict, batch.edge_index_dict)

    # Predictions
    pred = out['EOA Address'].argmax(dim=1)
    correct += (pred[batch['EOA Address'].test_mask] == batch['EOA Address'].y[batch['EOA Address'].test_mask]).sum().item()
    total += batch['EOA Address'].test_mask.sum().item()

print(f'Test Accuracy: {correct / total:.4f}')
