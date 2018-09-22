import hilbert as h
import data_preparation as dp
import shared


def get_embedder(cooc_dataset_name):
    path = dp.path_iteration.get_cooccurrence_path(cooc_dataset_name)
    cooc_stats = h.cooc_stats.read_stats(path)
    return h.embedder.get_embedder(cooc_stats)



def mle_embedder(
    cooc_dataset_name,
    device='cuda',
    learning_rate=1e-6,
    solver='sgd',
    momentum_decay=0.9,
    diag=None
):
    path = dp.path_iteration.get_cooccurrence_path(cooc_dataset_name)
    cooc_stats = h.cooc_stats.read_stats(path)
    return h.embedder.get_embedder(
        cooc_stats=cooc_stats,
        f_delta='mle',
        base='pmi',
        solver=solver,
        learning_rate=learning_rate,
        device=device,

        # vvv Defaults
        one_sided=False,
        neg_inf_val=None,
        X_max=None,
        d=300,
        k=None,
        t_undersample=None,
        shift_by=None,
        clip_thresh=None,
        diag=diag,
        k_samples=1,
        k_weight=None,
        alpha=1.0,
        momentum_decay=momentum_decay,
        implementation='torch',
        # ^^^ Defaults
    )



def glove_embedder(
    cooc_dataset_name,
    device='cuda',
    learning_rate=1e-6,
    solver='sgd',
    momentum_decay=0.9
):
    path = dp.path_iteration.get_cooccurrence_path(cooc_dataset_name)
    cooc_stats = h.cooc_stats.read_stats(path)
    return h.embedder.get_embedder(
        cooc_stats=cooc_stats,
        f_delta='glove',
        base='logNxx',
        solver=solver,
        neg_inf_val=0,
        X_max=100,
        learning_rate=learning_rate,
        one_sided=False,
        constrainer=h.constrainer.glove_constrainer,
        device=device,

        # vvv Defaults
        d=300,
        k=None,
        t_undersample=None,
        shift_by=None,
        clip_thresh=None,
        diag=None,
        k_samples=1,
        k_weight=None,
        alpha=1.0,
        momentum_decay=momentum_decay,
        implementation='torch',
        # ^^^ Defaults
    )

