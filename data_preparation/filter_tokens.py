import os
import sys
import shared
import hilbert as h



def filter_tokens(in_path, out_path, dictionary):
    with open(in_path) as in_file:
        with open(out_path, 'w') as out_file:
            for lineno, line in enumerate(in_file):
                if lineno % 1000 == 0:
                    print('{lines} lines'.format(lines=lineno))
                out_file.write(
                    ' '.join([t for t in line.split() if t in dictionary])
                    + '\n'
                )


if __name__ == '__main__':            

    vocab_size = int(sys.argv[1])
    in_path = shared.CONSTANTS.TOKENIZED_CAT_FULL_PATH
    out_path = os.path.join(
        shared.CONSTANTS.DATA_DIR, 'gigaword_tokenized_cat',
        'gigaword-tokenized-cat-{vocab}.txt'.format(vocab=vocab_size)
    )

    # Get the set of top-k words (k=vocab_size).
    dictionary = h.dictionary.Dictionary.load(os.path.join(
        shared.CONSTANTS.COOCCURRENCE_DIR, '5w', 'all-5w', 'dictionary'))
    dictionary = set(dictionary.tokens[:vocab_size])

    print((
        'reading from {in_path}, writing to {out_path}, '
        'keeping top {vocab} words.'
    ).format(in_path=in_path, out_path=out_path, vocab=len(dictionary)))

    filter_tokens(in_path, out_path, dictionary)

