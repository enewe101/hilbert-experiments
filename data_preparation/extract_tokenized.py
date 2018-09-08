import os
import re
import sys
import codecs

import corenlpy

import data_preparation as dp




def extract_tokenized_from_file(in_path, out_path):
    print(in_path)
    with open(in_path) as f:
        annotated_text = corenlpy.AnnotatedText(f.read())
    tokenized_sentences = [
        [t['word'].lower() for t in sentence['tokens']]
        for sentence in annotated_text.sentences
    ]
    with codecs.open(out_path, 'w', 'utf8') as f:
        f.write('\n'.join([
            ' '.join([t for t in sentence]) 
            for sentence in tokenized_sentences
        ]))


#def read_tokens(paths):
#    tokens = []
#    for path in paths:
#        with open(path) as f:
#            tokens.extend([token for token in f.read().split()])
#    return tokens



def extract_tokenized_in_place(dirname, in_fname):
    """
    Extracts the given file, placing it in this projects local data folder,
    with a subdir- and file-naming convention that mirrors the 
    gigaword-corenlp dataset's structure.
    """

    in_path = dp.path_iteration.get_gigaword_path(dirname, in_fname)

    # Make the out_path.
    out_fname = in_fname[:-4]
    out_path = dp.path_iteration.get_tokenized_path(dirname, out_fname)

    # Ensure the output directory exists
    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Extract the tokens.
    extract_tokenized_from_file(in_path, out_path)



def extract_all():
    for dirname, fname in dp.path_iteration.iter_gigaword_fnames():
        extract_tokenized_in_place(dirname, fname)
        print('processed ' + fname)
    print('DONE!')



def extract_one_sector(sector_name):

    """
    archive_sector_name (str) -- a three-hexidecimal-digit directory name
        corresponding to a gigaword-corenlp archive sector.

    Here an archive "sector" just refers to a subdirectory in the
    gigaword-corenlp dataset, which is organized into 4096 such "sectors", each
    containing about 2000 individual tared archive files.
    """
    print('\n\n\t--- STARTING EXTRACTION %s ---' % sector_name)
    # Ensure that a destination directory exists for this sector
    out_dir = os.path.join(dp.CONSTANTS.TOKENIZED_DIR, sector_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fname_iterator = dp.path_iteration.iter_gigaword_fnames_in_sector(
        sector_name)
    for dirname, fname in fname_iterator:
        in_path = dp.path_iteration.get_gigaword_path(dirname, fname)
        out_path = dp.path_iteration.get_tokenized_path(dirname, fname[:-4])
        extract_tokenized_from_file(in_path, out_path)



if __name__ == '__main__':

    # You need to provide at least one archive sector to be read and tokenized.
    if len(sys.argv) < 1:
        raise ValueError(
            'This script accepts one or more parameters: the name(s) of the '
            'gigaword-cornlp archive sector(s) to be tokenized.'
        )

    archive_sectors = sys.argv[1:]

    # Extract all the sectors if the special argument "all" is passed
    if len(archive_sectors) == 1 and archive_sectors[0] == 'all':
        extract_all_sectors()
        sys.exit(0)

    dp.path_iteration.raise_if_sectors_not_all_valid(archive_sectors)

    # Extract all the sectors to this project's data dir.
    for sector in archive_sectors:
        extract_one_sector(sector)


