
import sys
import os
import torch
import hilbert as h
import shared



def translate_embeddings(in_path, out_path):
    embeddings = read_glove_embeddings(in_path)
    embeddings.save(out_path)


def read_glove_embeddings(in_path, device=None):
    device = device or shared.CONSTANTS.MAIN_MEMORY_DEVICE
    with open(in_path) as in_file:
        tokens, vectors = [], []
        for line_num, line in enumerate(in_file):
            fields = line.split(' ')
            try:
                token, vector = fields[0], [float(i) for i in fields[1:]]
            except ValueError:
                print('{line_num}\t{fields}'.format(line_num=line_num, fields=fields))
                raise
            tokens.append(token)
            vectors.append(vector)

    V = torch.Tensor(vectors, device=device)
    dictionary = h.dictionary.Dictionary(tokens)
    return h.embeddings.Embeddings(V, dictionary=dictionary)


if __name__ == '__main__':
    in_dir_name = sys.argv[1]
    in_path = os.path.join(
        shared.CONSTANTS.EMBEDDINGS_DIR, in_dir_name, 'vectors.txt')
    out_path = os.path.join(
        shared.CONSTANTS.EMBEDDINGS_DIR, in_dir_name, 'as-hilbert-embeddings')
    translate_embeddings(in_path, out_path)





