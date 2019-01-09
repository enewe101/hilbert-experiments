import os
import sys
import argparse
import torch
import hilbert as h
import shared



# TODO: test
def translate_embeddings(
    embedding_type,
    in_path,
    out_path,
    dictionary_path=None,
    allow_mismatch=False,
    has_bias=False,
    has_covectors=False
):
    """
    Translate glove or w2v embeddings into the format used to store 
    hilbert.embeddings.Embeddings.  Eliminate the useless "</s>" token, which
    in w2v is not even trained.  Optionally shuffle the order to match that
    given by a dictionary stored on disc at dictionary_path.
    """
    if embedding_type.lower() == 'word2vec':
        embeddings = read_w2v_embeddings(in_path, has_covectors)
    elif embedding_type.lower() == 'glove':
        embeddings = read_glove_embeddings(in_path, has_covectors, has_bias)
    else:
        raise ValueError(
            "Unrecognized `embedding_type`: {}".format(embedding_type))

    if dictionary_path is not None:
        d = h.dictionary.Dictionary.load(dictionary_path)
        embeddings.sort_by_tokens(d.tokens, allow_mismatch=allow_mismatch)

    embeddings.save(out_path)


def read_glove_embeddings(in_path, has_covectors, has_bias):
    device = shared.CONSTANTS.MAIN_MEMORY_DEVICE

    tokens, vectors, covectors, v_bias, w_bias = [], [], None, None, None
    if has_covectors: 
        covectors = []
    if has_bias: 
        v_bias, w_bias = [], []

    with open(in_path) as in_file:

        for line_num, line in enumerate(in_file):

            fields = line.strip().split(' ')
            tokens.append(fields[0])
            vector_data = [float(v) for v in fields[1:]]

            if has_covectors:
                len_vector = len(vector_data) // 2
                if len_vector * 2 != len(vector_data):
                    raise ValueError(
                        "Something is wrong: the number of components for "
                        "vectors and covectors isn't divisible by 2. "
                        "Is has_bias ({}) correct?  Is has_covectors ({}) "
                        "correct?".format(has_bias, has_covectors)
                    )
                if has_bias:
                    len_vector -= 1
                    vectors.append(vector_data[:len_vector])
                    v_bias.append(vector_data[len_vector])
                    covectors.append(vector_data[len_vector+1:len_vector*2+1])
                    w_bias.append(vector_data[-1])
                else:
                    vectors.append(vector_data[:len_vector])
                    covectors.append(vector_data[len_vector:])

            else:
                if has_bias:
                    v_bias.append(vector_data[-1])
                    vectors.append(vector_data[:-1])
                else:
                    vectors.append(vector_data)

    dictionary = h.dictionary.Dictionary(tokens)
    return h.embeddings.Embeddings(
        vectors, W=covectors, v_bias=v_bias, w_bias=w_bias,
        dictionary=dictionary, device=device
    )


def read_w2v_embeddings(in_path, has_covectors):
    device = shared.CONSTANTS.MAIN_MEMORY_DEVICE
    covectors_reached = False
    with open(in_path) as in_file:

        tokens, vectors, covectors = [], [], None
        for line_num, line in enumerate(in_file):

            # In word2vec embeddings, the first line carries the vocabulary
            # and dimension.  Skip this.
            fields = line.strip().split(' ')
            if line_num == 0 and len(fields) == 2:
                continue

            # A line starting with 'covectors' signals that covectors will
            # follow
            if line.startswith('covectors'):
                covectors_reached = True
                covectors = []
                continue

            # A line starting with '</s>' has a useless embedding of 
            # sentence-end or document-end.  Skip it.
            if line.startswith('</s>'):
                continue

            token, vector = fields[0], [float(i) for i in fields[1:]]

            if covectors_reached:
                covectors.append(vector)
            else:
                tokens.append(token)
                vectors.append(vector)

    if covectors_reached and not has_covectors:
        raise ValueError(
            "The embeddings file has covectors, but none were expected. "
            "Try setting `has_covectors=True`."
        )
    if not covectors_reached and has_covectors:
        raise ValueError(
            "The embeddings file had no covectors, but they were expected. "
            "Try setting `has_covectors=False`."
        )

    dictionary = h.dictionary.Dictionary(tokens)
    return h.embeddings.Embeddings(
        vectors, covectors, dictionary=dictionary,device=device)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=(
        "Convert word2vec or GloVe vector output to format compatible with "
        "hilbert.embeddings.Embeddings."
    ))
    parser.add_argument(
        '--embedding-type', '-t', required=True, choices=('w2v', 'word2vec', 'glove')
    )
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
    parser.add_argument(
        '--has-bias', '-b', action='store_true',
        help=( 
            "Whether biases are included.  Only applies to GloVe vectors.")
    )
    parser.add_argument(
        '--has-covectors', '-c', action='store_true',
        help=("Whether covectors are included.")
    )

    args = vars(parser.parse_args())

    if args['type'] == 'w2v':
        args['embedding_type'] = 'word2vec'

    if args['embedding_type'] == 'word2vec' and args['has_bias']:
        raise ValueError("Word2vec vectors cannot have biases.")

    translate_embeddings(**args)

