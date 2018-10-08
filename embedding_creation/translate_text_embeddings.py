
import sys
import os
import torch
import hilbert as h
import shared



def translate_embeddings(in_path, out_path, skip=False):
    embeddings = read_text_embeddings(in_path, skip)
    embeddings.save(out_path)


def read_text_embeddings(in_path, skip=False, device=None):
    device = device or shared.CONSTANTS.MAIN_MEMORY_DEVICE
    has_covectors = False
    with open(in_path) as in_file:
        tokens, vectors, covectors = [], [], []
        for line_num, line in enumerate(in_file):

            # Skip the first line, if skip is True
            if skip:
                skip = False
                continue

            if line.startswith('covectors'):
                has_covectors = True
                continue

            fields = line.strip().split(' ')
            try:
                token, vector = fields[0], [float(i) for i in fields[1:]]
            except ValueError:
                print('{line_num}\t{fields}'.format(line_num=line_num, fields=fields))
                raise
            if has_covectors:
                covectors.append(vector)
            else:
                tokens.append(token)
                vectors.append(vector)

    dictionary = h.dictionary.Dictionary(tokens)
    V = torch.Tensor(vectors, device=device)
    if has_covectors:
        W = torch.Tensor(covectors, device=device)
        return h.embeddings.Embeddings(V,W,dictionary=dictionary,device=device)
    return h.embeddings.Embeddings(V, dictionary=dictionary, device=device)


if __name__ == '__main__':
    in_dir_name = sys.argv[1]
    in_fname = sys.argv[2]
    try:
        assert(sys.argv[3] == 'skip')
        skip = True
    except IndexError:
        skip = False

    in_path = os.path.join(
        shared.CONSTANTS.EMBEDDINGS_DIR, in_dir_name, in_fname)
    out_path = os.path.join(
        shared.CONSTANTS.EMBEDDINGS_DIR, in_dir_name, 'as-hilbert-embeddings')
    translate_embeddings(in_path, out_path, skip)

