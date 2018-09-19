import hilbert as h
import data_preparation as dp
import shared


def get_embedder(cooc_dataset_name):
    path = dp.path_iteration.get_cooccurrence_path(cooc_dataset_name)
    cooc_stats = h.cooc_stats.load_stats(path)
    return h.embedder.get_embedder(cooc_stats)
