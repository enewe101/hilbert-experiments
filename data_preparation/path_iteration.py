import os
import data_preparation as prep



def get_corenlp_path(dirname, fname):
    return os.path.join(prep.CONSTANTS.CORENLP_DIR, dirname, 'CoreNLP', fname)


def get_tokenized_path(dirname, fname):
    return os.path.join(prep.CONSTANTS.TOKENIZED_DIR, dirname, fname)


def get_test_path(fname):
    return os.path.join(prep.CONSTANTS.TEST_DOCS_DIR, fname)


def get_cooccurrence_path(fname):
    return os.path.join(prep.CONSTANTS.COOCCURRENCE_DIR, fname)


def get_test_write_path(fname):
    return os.path.join(prep.CONSTANTS.TEST_WRITE_DIR, fname)


def skip_file(fname):
    if fname.startswith('.nfs'):
        return True
    if fname.endswith('.swp') or fname.endswith('.swo'):
        return True
    return False


def iter_test_paths():
    for fname in iter_test_fnames():
        yield get_test_path(fname)

def iter_test_fnames():
    for path in os.listdir(prep.CONSTANTS.TEST_DOCS_DIR):
        if not skip_file(path):
            yield os.path.basename(path)

def get_test_tokens():
    paths = prep.path_iteration.iter_test_paths()
    return prep.extract_tokenized.read_tokens(paths)


def iter_corenlp_paths():
    for fname in iter_test_fnames():
        yield get_corenlp_path(fname)

def iter_corenlp_fnames():
    for dirname in os.listdir(prep.CONSTANTS.CORENLP_DIR):
        corenlp_dir = os.path.join(
            prep.CONSTANTS.CORENLP_DIR, dirname, 'CoreNLP')
        for fname in os.listdir(corenlp_dir):
            if not skip_file(fname):
                yield dirname, fname


def iter_tokenized_paths():
    for fname in iter_test_fnames():
        yield get_tokenized_path(fname)

def iter_tokenized_fnames():
    for dirname in os.listdir(prep.CONSTANTS.TOKENIZED_DIR):
        tokenized_path = os.path.join(
            prep.CONSTANTS.TOKENIZED_DIR, dirname)
        for fname in os.listdir(tokenized_path):
            if not skip_file(fname):
                yield dirname, fname
    


