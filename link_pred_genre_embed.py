import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import GraphSAINTRandomWalkSampler, NeighborSampler
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
    data.rating_index = torch.arange(len(ratings))
    return data


def load_genre_node(datapath) -> Data:
    movie_path = f'./{datapath}/movies.csv'
    rating_path = f'./{datapath}/ratings.csv'
    movies = pd.read_csv(movie_path).set_index('movieId')
    ratings = pd.read_csv(rating_path)

    user_mapping = {
        user_id: index for index, user_id in enumerate(ratings['userId'].unique())
    }
    num_users = len(user_mapping)

    movie_mapping = {
        movie_id: idx + num_users 
        for idx, movie_id in enumerate(movies.index)
    }
    num_movies = len(movie_mapping)

    genres = set(g for gs in movies.genres for g in gs.strip().split('|'))
    num_genres = len(genres)
    genre_mapping = {g: i for i, g in enumerate(genres)}

    num_nodes = num_users + num_movies + num_genres
    x = torch.arange(num_nodes).reshape(-1, 1)

    src, dst = [], []
    for _, u, m in ratings[['userId', 'movieId']].itertuples():
        src.append(user_mapping[u])
        dst.append(movie_mapping[m])

    genre_src, genre_dst = [], []
    for movie, _gs in movies['genres'].items():
        for genre in _gs.strip().split('|'):
            genre_src.append(movie_mapping[movie])
            genre_dst.append(genre_mapping[genre])

    merge_src, merge_dst = src + genre_src, dst + genre_dst
    edge_index = torch.tensor([merge_src, merge_dst])
    edge_label = torch.cat((
        torch.tensor(ratings['rating'].values.reshape(-1, 1), dtype=torch.long),
        torch.ones(len(genre_src), 1, dtype=torch.long),
    ))

    data = Data(x=x, edge_index=edge_index, edge_label=edge_label)
    data.rating_index = torch.arange(len(ratings))
    return data


class Net(nn.Module):
    def __init__(self, num_nodes, hidden_dim, dropout):
        super().__init__()

        emb_dim = hidden_dim
        out_channels = 16

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.embed = nn.Embedding(num_embeddings=num_nodes, embedding_dim=emb_dim)

        self.convs = [
            GCNConv(hidden_dim, hidden_dim).to(device),
            GCNConv(hidden_dim, out_channels).to(device),
        ]

        self.norms = [
            nn.LayerNorm(hidden_dim).to(device),
            # nn.LayerNorm(hidden_dim).to(device),
        ]

        # self.fc1 = nn.Linear(out_channels * 2, 1)
        self.linears = nn.Sequential(
            nn.Linear(out_channels * 2, 16),
            nn.Dropout(0.3),
            # Treat this as regression, ie: produce 1 value.
            nn.Linear(16, 1),
        )

    def encode(self, x, edge_index):
        x = self.embed(x)
        x = x.squeeze()

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            emb = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if i < len(self.convs) - 1:
                x = self.norms[i](x)
        
        return emb

    def decode(self, z, edge_label_index):
        out = torch.cat((z[edge_label_index[0]], z[edge_label_index[1]]), axis=-1)
        return self.linears(out)


def train(model, optimizer, train_data) -> float:
    model.train()

    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    out = model.decode(z, train_data.edge_index)

    loss = F.mse_loss(out, train_data.edge_label.type(torch.float))
    loss.backward()
    optimizer.step()

    return loss


@torch.no_grad()
def test(model, data) -> float:
    model.eval()
    data = data.to(device)
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_index)
    # auc = roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
    loss = F.mse_loss(out, data.edge_label.type(torch.float))
    return loss


@torch.no_grad()
def evaluate(model, train, val, test) -> float:
    model.eval()
    res = []
    for data in (train, val, test):
        data = data.to(device)
        z = model.encode(data.x, data.edge_index)
        out = model.decode(z, data.edge_index)
        loss = F.mse_loss(out, data.edge_label.type(torch.float))
        res.append(loss)
    return res


def edge_stats(data, type_=''):
    labels = data.edge_label.squeeze().numpy()
    label, count = np.unique(labels, return_counts=True)
    print(type_, '\t', len(labels), dict(zip(label, (count/len(labels)).round(3))))


def split(data, test_ratio, val_ratio):
    rating_index = data.rating_index
    data = data.clone()
    del data['rating_index']

    N = len(rating_index)
    remain = torch.arange(N, len(data.edge_index[0]))
    shuffle = rating_index[torch.randperm(N)]
    
    test_split = int(test_ratio * N)
    val_split = int((test_ratio + val_ratio)* N)
    test_idx = torch.cat((shuffle[: test_split], remain))
    val_idx = torch.cat((shuffle[test_split: val_split], remain))
    train_idx = torch.cat((shuffle[val_split: ], remain))

    test_data = data.clone()
    test_data.edge_index = data.edge_index[:, test_idx]
    test_data.edge_label = data.edge_label[test_idx]
    
    val_data = data.clone()
    val_data.edge_index = data.edge_index[:, val_idx]
    val_data.edge_label = data.edge_label[val_idx]
    
    train_data = data.clone()
    train_data.edge_index = data.edge_index[:, train_idx]
    train_data.edge_label = data.edge_label[train_idx]
    
    return train_data, val_data, test_data


def main():
    datapath = 'ml-latest-small'
    # datapath = 'ml-25m'

    dataset = load_data(datapath)
    dataset = load_genre_node(datapath)
    print(dataset, '\n')

    # dataset = T.NormalizeFeatures()(dataset)
    # train_data, val_data, test_data = T.RandomLinkSplit(
    #     # num_val=0.1,
    #     # num_test=0.2,
    #     is_undirected=False,
    #     add_negative_train_samples=False,
    # )(dataset)

    train_data, val_data, test_data = split(
        dataset,
        test_ratio=0.2,
        val_ratio=0.1,
    )

    # edge_stats(dataset, 'dataset')
    # edge_stats(train_data, 'train_data')
    # edge_stats(val_data, 'val_data')
    # edge_stats(test_data, 'test_data')
    # print('\n')

    ########### params #############

    n_epoch = 20_000
    learn_rate = 0.0005
    hidden_dim = 16
    dropout = 0.1
    use_loader = False

    ################################

    model = Net(num_nodes=dataset.num_nodes, hidden_dim=hidden_dim, dropout=dropout)
    model = model.to(device)
    # train_data = train_data.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learn_rate)

    # train_loader = NeighborSampler(
    #     train_data.edge_index,
    #     # node_idx=train_data.nodes,
    #     sizes=[15, 10, 5],
    #     batch_size=1024,
    #     shuffle=True,
    #     num_workers=12,
    # )
    # batch, ids, adj = next(iter(train_loader))

    train_loader = GraphSAINTRandomWalkSampler(
        train_data,
        batch_size=1_000,
        walk_length=10,
        num_steps=5,
        # sample_coverage=100,
        save_dir='.train_load',
        # num_workers=12,
    )

    best = 1_000

    print(f'idx\tTrain_Err\tValid_Err\tTest_Err')
    for epoch_idx in range(1, n_epoch + 1):
        if use_loader:
            data = next(iter(train_loader)).to(device)
        else:
            data = train_data.to(device)

        train_loss = train(model, optimizer, data)
        epoch_eval = evaluate(model, train_data, val_data, test_data)

        if epoch_idx % 500 == 0:
            # print(data)
            print(
                f'{epoch_idx:04d}\t' + '    \t'.join(map('{:.4f}'.format, epoch_eval))
            )

        # Early stopping
        _, val, test = epoch_eval
        if epoch_idx > 500 and test/best > 1.5:
            break
        best = min(best, test)

    print(f'\nBest Test_Err: {best}')


if __name__ == '__main__':
    main()
