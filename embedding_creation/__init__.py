import hilbert as h
import data_preparation as dp


def get_embedder(cooc_dataset_name):
    path = dp.path_iteration.get_cooccurrence_path(cooc_dataset_name)
    cooc_stats = h.cooc_stats.load_stats(path)
    return h.embedder.get_embedder(cooc_stats)


def get_torch_embedder(cooc_dataset_name):
    path = dp.path_iteration.get_cooccurrence_path(cooc_dataset_name)
    cooc_stats = h.cooc_stats.load_stats(cooc_dataset_name)
    return h.embedder.get_torch_embedder(cooc_stats)
