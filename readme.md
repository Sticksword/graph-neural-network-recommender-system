[here is a quick recap of traditional recommender systems](https://towardsdatascience.com/intro-to-recommender-system-collaborative-filtering-64a238194a26). basically we have users that give ratings and we want to find users who rated items similarly to our user in question. this is known as user collaborative filtering. we can use pearson correlation or cosine similarity for finding the top K users most similar to our user in question. an advantage of item based collaborative filtering as opposed to user based is that item ratings tend to not drift/fluctuate over time as much as individual user ratings.

the above is known as the nearest neighbor approach to collaborative filtering. one issue is that the user-item matrix might be sparse which makes things inefficient. i don't remember the details of linear algebra but basically matrix factorization is how we can reduce a sparse user-item matrix to a low dimension matrices.

we can optimize a matrix factorization model using alternating least squares (ALS). we can also leverage bayesian personalized ranking loss instead of ALS loss, which directly optimizes for ranking.

for deep recommender systems, we can start by using a deep MLP network with some embedding layers at the beginning for users and items.

one such example can be found in [this notebook](https://github.com/devforfu/pytorch_playground/blob/master/movielens.ipynb).
the idea is to pass in user-movie edges and generate a score for every edge. it's framed as a regression problem and uses MSE to calculate loss.
it uses embedding layers to represent both a movie and a user. the output embeddings is piped into an MLP to ultimately generate the score.
we can probably replace this crude attempt to embed a movie with more advanced techniques like embeddings generated from GCNs.

[this notebook](https://github.com/HarshdeepGupta/recommender_pytorch/blob/master/MLP.py) is basically a dupe of the above. while [associated the article](https://towardsdatascience.com/recommender-systems-using-deep-learning-in-pytorch-from-scratch-f661b8f391d7) and the repo claim it's neural graph collab filtering (NGCF), it is not. it's just normal linear layers. it's neural collaborative filtering. no graph component.

[here's a real NGCF implementation](https://github.com/huangtinglin/NGCF-PyTorch/blob/master/NGCF/NGCF.py). we still use embedding layers (just like the bootleg attempts above) but we also correctly calculate the impact of connected items/users. we use a GNN layer to do the neighborhood aggregation piece. collaborative filtering approaches seem to leverage the concept of "finding the user(s) most similar to you and recommending what they like".

Pinsage and the PgY link pred example are examples that are not collaborative filtering but rather node embedding creation and then leveraging those embeddings to determine if two nodes are similar (link prediction). Actually all of the graph layers are variations of this neighborhood aggregation concept with the ultimate goal of training node embeddings. We can then use those embeddings for the classification problem of whether there is a link of not, given two nodes.

We can also use these same embeddings to search for nodes similar to a given node. ie. given node A, maybe do some sort to find the closest (cosine distance, etc.) node embedding to the embedding of our given node A. we can also use these embeddings as features for ranking candidates for a user in an ml system.

pinsage traings by identifying pairs of nodes (pin/board and implicitly associated pin/board that a user had clicked from initial pin/board) and treating these as positive pairs. they then negatively sample negative pairs (all the pairs that are not positive) and create the dataset. loss is based on difference between inner product of positive query/item pair and inner product of sampled negative query/item pair. notice that users are not treated as nodes unlike the NGCF implementation above. it's instead framed as "given two pins, are they related?"

for the movielens dataset, we can add tag data to the movie attributes if we wanted. these represent the genres.

all the above methods use neighborhood aggregation, it's just the framing of the problem that's different ie. inputs, outputs, regression v classification, etc. a lot of graph approaches seem to train node embeddings for a certain task (eg. for pinsage it's link prediction/recommendation)
