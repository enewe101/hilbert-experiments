import data_preparation as prep
import corenlpy
import os
import cluster_func
import codecs


def read_tokens(paths):
    return [token for path in paths for token in open(path).read().split()]


def extract_tokenized_from_file(in_path, out_path):
    annotated_text = corenlpy.AnnotatedText(open(in_path).read())
    tokenized_sentences = [
        [t['word'].lower() for t in sentence['tokens']]
        for sentence in annotated_text.sentences
    ]
    codecs.open(out_path, 'w', 'utf8').write('\n'.join([
        ' '.join([t for t in sentence]) 
        for sentence in tokenized_sentences
    ]))



def target(dirname, in_fname):

    in_path = prep.path_iteration.get_corenlp_path(dirname, in_fname)

    # Make the out_path.
    out_fname = in_fname[:-4]
    out_path = prep.path_iteration.get_tokenized_path(dirname, out_fname)

    # Ensure the output directory exists
    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Extract the tokens.
    extract_tokenized_from_file(in_path, out_path)



def extract_all():
    for dirname, fname in prep.path_iteration.iter_corenlp_fnames():
        target(dirname, fname)
        print 'processed', fname
    print 'DONE!'



def extract_one():
    in_path = os.path.join(
        prep.CONSTANTS.DATA_DIR, '000', 'CoreNLP', '00000102437441bd.txt.xml')
    out_path = os.path.join(
        prep.CONSTANTS.DATA_DIR, 'tokenized', '00000102437441bd.txt')
    extract_tokenized_from_file(in_path, out_path)




