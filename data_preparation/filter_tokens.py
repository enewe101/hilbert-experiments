import os
import sys
import shared
import hilbert as h
import argparse


def filter_tokens(in_file, out_file, dictionary):
    with open(in_file) as in_file:
        with open(out_file, 'w') as out_file:
            for lineno, line in enumerate(in_file):
                if lineno % 1000 == 0:
                    print('{lines} lines'.format(lines=lineno))
                out_file.write(' '.join([
                    t for t in line.split() 
                    if t in dictionary and t != '<unk>'
                ]) + '\n')


if __name__ == '__main__':            

    parser = argparse.ArgumentParser(description=(
        "Filters corpus, keeping only words found in provided dictionary"
    ))
    parser.add_argument(
        '--in-file', '-i', required=True, help="File name for input corpus"
    )
    parser.add_argument(
        '--out-file', '-o', required=True, help="File name for output corpus"
    )
    parser.add_argument(
        '--dictionary', '-d', required=True, help=(
            "Directory containing the dictionary, relative to cooccurrences dir"
        )
    )

    # Parse the arguments
    args = vars(parser.parse_args())

    # Corpus path and output directory are relative to standard locations.
    args['in_file'] = os.path.join(
        shared.CONSTANTS.TOKENIZED_CAT_DIR, args['in_file']
    )
    args['out_file'] = os.path.join(
        shared.CONSTANTS.TOKENIZED_CAT_DIR, args['out_file']
    )
    args['dictionary'] = h.dictionary.Dictionary.load(os.path.join(
        shared.CONSTANTS.COOCCURRENCE_DIR, args['dictionary'], 
        'dictionary'
    ))

    print('reading from {}, writing to {}.'.format(
        args['in_file'], args['out_file']))

    filter_tokens(**args)

