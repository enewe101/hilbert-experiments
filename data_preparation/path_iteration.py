import os
import data_preparation as dp
import re





def iter_test_fnames():
    for path in os.listdir(dp.CONSTANTS.TEST_DOCS_DIR):
        if not skip_file(path):
            yield os.path.basename(path)

def iter_test_paths():
    for fname in iter_test_fnames():
        yield get_test_path(fname)




def get_gigaword_path(dirname, fname):
    return os.path.join(
        dp.CONSTANTS.LOCAL_GIGAWORD_DIR, dirname, 'CoreNLP', fname)


def get_tokenized_path(dirname, fname):
    return os.path.join(dp.CONSTANTS.TOKENIZED_DIR, dirname, fname)


def get_test_path(fname):
    return os.path.join(dp.CONSTANTS.TEST_DOCS_DIR, fname)


def get_cooccurrence_path(fname):
    return os.path.join(dp.CONSTANTS.COOCCURRENCE_DIR, fname)


def get_test_write_path(fname):
    return os.path.join(dp.CONSTANTS.TEST_WRITE_DIR, fname)


def skip_file(fname):
    if fname.startswith('.'):
        return True
    if fname.endswith('.swp') or fname.endswith('.swo'):
        return True
    return False


def iter_gigaword_fnames():
    """
    Iterates (sector_name, fname) pairs for all of gigaword (all sectors):
    """
    for sector_name in os.listdir(dp.CONSTANTS.LOCAL_GIGAWORD_DIR):
        gigaword_dir = os.path.join(
            dp.CONSTANTS.LOCAL_GIGAWORD_DIR, sector_name, 'CoreNLP')
        for sector_name, fname in iter_gigaword_fnames_in_sector(sector_name):
            yield sector_name, fname


def iter_gigaword_fnames_in_sector(sector_name):
    """
    Iterates (sector_name, fname) pairs for one sector.
    """
    sector_dir = os.path.join(
        dp.CONSTANTS.LOCAL_GIGAWORD_DIR, sector_name, 'CoreNLP')
    for fname in os.listdir(sector_dir):
        if not skip_file(fname):
            yield sector_name, fname


def iter_gigaword_paths_in_sector(sector_name):
    """
    Iterates full file path to all gigaword archive files from one sector.
    """
    for dirname, fname in iter_gigaword_fnames_in_sector(sector_name):
        yield get_gigaword_path(dirname, fname)


def iter_gigaword_paths():
    for dirname, fname in iter_gigaword_fnames():
        yield get_gigaword_path(dirname, fname)



def iter_tokenized_fnames_in_sector(sector):
    dirpath = os.path.join(dp.CONSTANTS.TOKENIZED_DIR, sector)
    for fname in os.listdir(dirpath):
        if not skip_file(fname):
            yield sector, fname

def iter_tokenized_paths_in_sector(sector, limit=None):
    dirpath = os.path.join(dp.CONSTANTS.TOKENIZED_DIR, sector)
    if limit is not None:
        print('Warning: limiting files iterated per sector')
    if not os.path.exists(dirpath):
        print('WARNING: sector does not exist (%s).' % dirpath)
        return
    for i, fname in enumerate(os.listdir(dirpath)):
        if limit is not None and i >= limit:
            break
        if not skip_file(fname):
            yield get_tokenized_path(sector, fname)


def iter_tokenized_paths():
    for dirname, fname in iter_tokenized_fnames():
        yield get_tokenized_path(dirname, fname)

def iter_tokenized_fnames():
    for dirname in os.listdir(dp.CONSTANTS.TOKENIZED_DIR):
        tokenized_path = os.path.join(
            dp.CONSTANTS.TOKENIZED_DIR, dirname)
        for fname in os.listdir(tokenized_path):
            if not skip_file(fname):
                yield dirname, fname
    

VALID_ARCHIVE_MATCHER = re.compile('[0-9a-f]{3,3}')
def raise_if_sectors_not_all_valid(sectors):
    all_valid_sectors = all(
        VALID_ARCHIVE_MATCHER.match(sector) for sector in sectors)
    if not all_valid_sectors:
        raise ValueError(
            'Gigaword-corenlp archive names should comprise three hexidecimal '
            'digits.'
        )

