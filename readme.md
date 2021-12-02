
we have replicated the behavior in [this notebook](https://github.com/devforfu/pytorch_playground/blob/master/movielens.ipynb).
the idea is just to pass in user-movie edges and generate a score for every edge.
it uses embedding layers to represent both a movie and a user.
we can probably replace this crude attempt to embed a movie with more advanced techniques like embeddings generated from GCNs.

[this notebook](https://github.com/HarshdeepGupta/recommender_pytorch/blob/master/MLP.py) is basically a dupe of the above. while [associated the article](https://towardsdatascience.com/recommender-systems-using-deep-learning-in-pytorch-from-scratch-f661b8f391d7) and the repo claim it's neural graph collab filtering (NGCF), it is not. it's just normal linear layers.

[here's a real NGCF implementation](https://github.com/huangtinglin/NGCF-PyTorch/blob/master/NGCF/NGCF.py). we still use embedding layers (just like the bootleg attempts above) but we also correctly calculate the impact of connected items/users.


