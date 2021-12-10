import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

print('torch', torch.__version__)
print('torch_geometric', torch_geometric.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'You are using device: {device}\n')


# class GenresEncoder:
#     def __init__(self, sep='|', device=None):
#         self.sep = sep
#         self.device = device

#     def __call__(self, sr: pd.Series):
#         genres = set(g for col in sr.values for g in col.split(self.sep))
#         mapping = {genre: i for i, genre in enumerate(genres)}

#         x = torch.zeros(len(sr), len(mapping)).to(self.device)
#         for i, col in enumerate(sr.values):
#             for genre in col.split(self.sep):
#                 x[i, mapping[genre]] = 1
#         return x


def load_data(datapath) -> Data:
    movie_path = f'./{datapath}/movies.csv'
    rating_path = f'./{datapath}/ratings.csv'
    # movies = pd.read_csv(movie_path)
    ratings = pd.read_csv(rating_path)

    user_mapping = {
        user_id: index for index, user_id in enumerate(ratings['userId'].unique())
    }
    num_users = len(user_mapping)
    movie_mapping = {
        movie_id: index + num_users
        for index, movie_id in enumerate(ratings['movieId'].unique())
    }
    num_movies = len(movie_mapping)

    num_nodes = num_users + num_movies
    x = torch.arange(num_nodes).reshape(-1, 1)

    src = [user_mapping[index] for index in ratings['userId']]
    dst = [movie_mapping[index] for index in ratings['movieId']]
    edge_index = torch.tensor([src, dst])

    edge_label = torch.tensor(ratings['rating'].values.reshape(-1, 1), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_label=edge_label)
    return data


class Net(torch.nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        
        in_channels = 32
        out_channels = 16
        emb_dim = 16

        self.embed = torch.nn.Embedding(num_embeddings=num_nodes, embedding_dim=emb_dim)
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)
        
        # Treat this as regression, ie: produce 1 value.
        self.fc1 = torch.nn.Linear(out_channels * 2, 1)

    def encode(self, x, edge_index):
        x = self.embed(x)
        x = x.squeeze()
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index): 
        out = torch.cat((z[edge_label_index[0]], z[edge_label_index[1]]), axis=-1)
        return self.fc1(out)


def train(model, optimizer, train_data):
    model.train()

    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    out = model.decode(z, train_data.edge_index)

    loss = F.mse_loss(out, train_data.edge_label.type(torch.float))
    loss.backward()
    optimizer.step()

    return loss


@torch.no_grad()
def test(model, data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


def main():
    datapath = 'ml-latest-small'
    # datapath = 'ml-25m'

    dataset = load_data(datapath)
    print(dataset)
    # dataset = T.NormalizeFeatures()(dataset)
    train_data, val_data, test_data = T.RandomLinkSplit(
        # num_val=0.1,
        # num_test=0.2,
        is_undirected=False,
        add_negative_train_samples=False,
    )(dataset)

    ########### params #############

    n_epoch = 20_000
    learn_rate = 0.0005
    
    ################################

    model = Net(num_nodes=dataset.num_nodes)
    model = model.to(device)
    train_data = train_data.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learn_rate)

    # best_val_auc = final_test_auc = 0
    print(f'idx\tTrain Loss \tValid Loss')
    for epoch_idx in range(1, n_epoch + 1):
        train_loss = train(model, optimizer, train_data)
        val_auc = test(val_data)
        test_auc = test(test_data)
        
        if epoch_idx % 100 == 0:
            print(
                '{}    \t{:.4f}    \t{:.4f}    \t{:.4f}'.format(
                epoch_idx, train_loss, val_auc, test_auc
                )
            )

        # if val_auc > best_val_auc:
        #     best_val = val_auc
        #     final_test_auc = test_auc


if __name__ == '__main__':
    main()
