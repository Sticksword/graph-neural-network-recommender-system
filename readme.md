[here is a quick recap of traditional recommender systems](https://towardsdatascience.com/intro-to-recommender-system-collaborative-filtering-64a238194a26). basically we have users that give ratings and we want to find users who rated items similarly to our user in question. this is known as user collaborative filtering. we can use pearson correlation or cosine similarity for finding the top K users most similar to our user in question. an advantage of item based collaborative filtering as opposed to user based is that item ratings tend to not drift/fluctuate over time as much as individual user ratings.

the above is known as the nearest neighbor approach to collaborative filtering. one issue is that the user-item matrix might be sparse which makes things inefficient. i don't remember the details of linear algebra but basically matrix factorization is how we can reduce a sparse user-item matrix to a low dimension matrices.

we can optimize a matrix factorization model using alternating least squares (ALS). we can also leverage bayesian personalized ranking loss instead of ALS loss, which directly optimizes for ranking.

for deep recommender systems, we can start by using a deep MLP network with some embedding layers at the beginning for users and items.

one such example can be found in [this notebook](https://github.com/devforfu/pytorch_playground/blob/master/movielens.ipynb).
the idea is just to pass in user-movie edges and generate a score for every edge.
it uses embedding layers to represent both a movie and a user.
we can probably replace this crude attempt to embed a movie with more advanced techniques like embeddings generated from GCNs.

[this notebook](https://github.com/HarshdeepGupta/recommender_pytorch/blob/master/MLP.py) is basically a dupe of the above. while [associated the article](https://towardsdatascience.com/recommender-systems-using-deep-learning-in-pytorch-from-scratch-f661b8f391d7) and the repo claim it's neural graph collab filtering (NGCF), it is not. it's just normal linear layers.

[here's a real NGCF implementation](https://github.com/huangtinglin/NGCF-PyTorch/blob/master/NGCF/NGCF.py). we still use embedding layers (just like the bootleg attempts above) but we also correctly calculate the impact of connected items/users.


