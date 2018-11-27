
import os
import sys
import argparse

import torch

import hilbert as h
import shared



# TODO: test
def translate_embeddings(
    in_path, out_path, dictionary_path=None, allow_mismatch=False
):
    """
    Translate glove or w2v embeddings into the format used to store 
    hilbert.embeddings.Embeddings.  Eliminate the useless "</s>" token, which
    in w2v is not even trained.  Optionally shuffle the order to match that
    given by a dictionary stored on disc at match_dictionary_path.
    """
    embeddings = read_text_embeddings(in_path)
    if dictionary_path is not None:
        d = h.dictionary.Dictionary.load(dictionary_path)
        embeddings.sort_by_tokens(d.tokens, allow_mismatch=allow_mismatch)
    embeddings.save(out_path)


# TODO: This should be part of hilbert.embeddings
def read_text_embeddings(in_path, device=None):
    device = device or shared.CONSTANTS.MAIN_MEMORY_DEVICE
    has_covectors = False
    with open(in_path) as in_file:

        tokens, vectors, covectors = [], [], []
        for line_num, line in enumerate(in_file):

            # In word2vec embeddings, the first line carries the vocabulary
            # and dimension.  Skip this.
            fields = line.strip().split(' ')
            if line_num == 0 and len(fields) == 2:
                continue

            # A line starting with 'covectors' signals that covectors will
            # follow
            if line.startswith('covectors'):
                has_covectors = True
                continue

            # A line starting with '</s>' has a useless embedding of 
            # sentence-end or document-end.  Skip it.
            if line.startswith('</s>'):
                continue

            token, vector = fields[0], [float(i) for i in fields[1:]]

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

    parser = argparse.ArgumentParser(description=(
        "Convert word2vec or GloVe vector output to format compatible with "
        "hilbert.embeddings.Embeddings."
    ))
    parser.add_argument(
        '--in-path', '-i', required=True,
        help = (
            "Path to embeddings to be translated."
        )
    )
    parser.add_argument(
        '--out-path', '-o', required=True, 
        help="Path at which to write translated embeddings."
    )
    parser.add_argument(
        '--dictionary-path', '-d', default=None,
        help="Dictionary to be used to specify ordering/indexing of embeddings."
    )
    parser.add_argument(
        '--allow-mismatch', '-a', action='store_true',
        help=( 
            "Allow provided dictionary to not match original dictionary.")
    )

    args = vars(parser.parse_args())
    translate_embeddings(**args)

