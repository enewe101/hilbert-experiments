COOC_STATS_PATH = '5w-dynamic-10k/5w-dynamic-10000-rep0'
K = 15
LEARNING_RATE = 1e-9
EMBEDDINGS_INIT_PATH = 'w2v-testing'
DEVICE = 'cpu'
DTYPE = torch.float32


def check_w2v_sharded():
    """
    This exists to output a basic w2v implementation that doesn't use sharding
    so that it can be compared with one which does, to demonstrate equivalence.
    For simplicity, some of the details omitted (no undersampling common words,
    no unigram smoothing).
    """
    cooc_stats = dp.read_stats(COOC_STATS_PATH)
    embedder = h.embedder.get_w2v_embedder(
        cooc_stats=cooc_stats,
        k=K,
        alpha=None,
        t=None,
        learning_rate=LEARNING_RATE,
        shard_factor=1,
        device=DEVICE
    )
    embeddings = h.embeddings.Embeddings.load(
        EMBEDDINGS_INIT_PATH, device=DEVICE)
    embedder.V = embeddings.V
    embedder.W = embeddings.W
    return embedder


def check_w2v_unsharded():
    """
    This exists to output a basic w2v implementation that doesn't use sharding
    so that it can be compared with one which does, to demonstrate equivalence.
    For simplicity, some of the details omitted (no undersampling common words,
    no unigram smoothing).
    """
    path = dp.path_iteration.get_cooccurrence_path(COOC_STATS_PATH)
    cooc_stats = hn.cooc_stats.read_stats(path)
    Nxx = torch.tensor(
        cooc_stats.Nxx.toarray(), 
        dtype=DTYPE, device=DEVICE
    )
    Nx = torch.tensor(cooc_stats.Nx, dtype=DTYPE, device=DEVICE)
    Nxt = torch.tensor(
        np.asarray(np.sum(cooc_stats.Nxx, axis=0)),
        dtype=DTYPE, device=DEVICE
    )
    N = torch.tensor(cooc_stats.N, dtype=DTYPE, device=DEVICE)
    M = (
        torch.log(Nxx) + torch.log(N) - torch.log(Nx) 
        - torch.log(Nxt) - torch.log(torch.tensor(K, dtype=DTYPE))
    )
    f_delta = hn.f_delta.get_f_w2v((Nxx, Nx, Nxt, N), M, K, device=DEVICE)
    embedder = hn.torch_embedder.TorchHilbertEmbedder(
        M, f_delta, learning_rate=LEARNING_RATE, device=DEVICE)
    embeddings = h.embeddings.Embeddings.load(
        EMBEDDINGS_INIT_PATH, device=DEVICE)
    embedder.V = embeddings.V.t()
    embedder.W = embeddings.W
    return embedder




