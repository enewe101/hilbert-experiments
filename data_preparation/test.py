import os
import shutil
from collections import Counter

import numpy as np
from scipy import sparse
from unittest import main, TestCase

import data_preparation as dp
import hilbert as h

import warnings
import logging
logging.captureWarnings(True)





class TestPathIteration(TestCase):

    def test_iter_gigaword_fnames_in_sector(self):
        expected_fname_path = os.path.join(dp.CONSTANTS.TEST_DATA_DIR, '000.ls')
        with open(expected_fname_path) as f:
            expected_fnames = f.read().split()
        expected_fnames = { ('000', fname) for fname in expected_fnames }
        found_fnames = set(
            dp.path_iteration.iter_gigaword_fnames_in_sector('000'))
        self.assertEqual(found_fnames, expected_fnames)


    def test_iter_gigaword_paths_in_sector(self):
        expected_fname_path = os.path.join(dp.CONSTANTS.TEST_DATA_DIR, '000.ls')
        with open(expected_fname_path) as f:
            expected_fnames = f.read().split()
        expected_paths = {
            dp.path_iteration.get_gigaword_path('000', fname)
            for fname in expected_fnames
        }
        found_paths = set(
            dp.path_iteration.iter_gigaword_paths_in_sector('000'))
        self.assertEqual(found_paths, expected_paths)


    def test_get_gigaword_path(self):
        expected_path = os.path.join(
            dp.CONSTANTS.LOCAL_GIGAWORD_DIR, '000', 'CoreNLP', 'xyz.txt.xml')
        found_path =  dp.path_iteration.get_gigaword_path('000', 'xyz.txt.xml')
        self.assertEqual(found_path, expected_path)


    def test_get_tokenized_path(self):
        expected_path = os.path.join(
            dp.CONSTANTS.TOKENIZED_DIR, '000', 'xyz.txt.xml')
        found_path =  dp.path_iteration.get_tokenized_path('000', 'xyz.txt.xml')
        self.assertEqual(found_path, expected_path)


    def test_iter_tokenized_fnames_in_sector(self):
        expected_fname_path = os.path.join(dp.CONSTANTS.TEST_DATA_DIR, '000.ls')
        with open(expected_fname_path) as f:
            expected_fnames = [
                fname[:-4]  # Strip off the ".xml"
                for fname in f.read().split()
            ]
        expected_paths = {('000', fname) for fname in expected_fnames}
        found_paths = set(
            dp.path_iteration.iter_tokenized_fnames_in_sector('000'))
        self.assertEqual(found_paths, expected_paths)


    def test_iter_tokenized_paths_in_sector(self):
        expected_fname_path = os.path.join(dp.CONSTANTS.TEST_DATA_DIR, '000.ls')
        with open(expected_fname_path) as f:
            expected_fnames = [
                fname[:-4]  # Strip off the ".xml"
                for fname in f.read().split()
            ]
        expected_paths = {
            dp.path_iteration.get_tokenized_path('000', fname)
            for fname in expected_fnames
        }
        found_paths = set(
            dp.path_iteration.iter_tokenized_paths_in_sector('000'))
        self.assertEqual(found_paths, expected_paths)



    

class TestCooccurrenceExtraction(TestCase):

    EXPECTED_COUNTS_DOC1 = Counter({
        ('Speak', 'or'): 2, ('Speak', 'hold') : 2, ('Speak', '.'): 1,
        ('Speak', 'peace'): 1, ('or', 'Speak'): 2, ('or', 'hold'): 2,
        ('or', 'your'): 2, ('or', '.'): 1, ('hold', 'Speak'): 2,
        ('hold', 'or'): 2, ('hold', 'your'): 2, ('hold', 'peace'): 2,
        ('your', 'or'): 2, ('your', 'hold'): 2, ('your', 'peace'): 2,
        ('your', '.'): 2, ('peace', 'hold'): 2, ('peace', 'your'): 2,
        ('peace', '.'): 2, ('peace', 'Speak'): 1, ('.', 'your'): 2,
        ('.', 'peace'): 2, ('.', 'Speak'): 1, ('.', 'or'): 1
    })

    EXPECTED_COUNTS_DOC2 = Counter({
        ('Eggs', ','): 2, ('Eggs', 'toast'): 2, ('Eggs', '.'): 1,
        ('Eggs', 'coffee'): 1, (',', 'Eggs'): 2, (',', '.'): 1,
        (',', 'toast'): 4, (',', ','): 4, (',', 'and'): 2,
        (',', 'coffee'): 2, ('toast', 'Eggs'): 2, ('toast', ','): 4,
        ('toast', 'and'): 2, ('and', 'toast'): 2, ('and', ','): 2,
        ('and', 'coffee'): 2, ('and', '.'): 2, ('coffee', ','): 2,
        ('coffee', 'and'): 2, ('coffee', '.'): 2, ('coffee', '.'): 2,
        ('coffee', 'Eggs'): 1, ('.', 'coffee'): 2, ('.', 'and'): 2, 
        ('.', 'Eggs'): 1, ('.', ','): 1, 
    })



    def test_extract_cooccurrence_from_file(self):

        expected_counts = self.EXPECTED_COUNTS_DOC1 + self.EXPECTED_COUNTS_DOC2 
        cooccurrences = h.cooc_stats.CoocStats(verbose=False)

        window = 2
        paths = dp.path_iteration.iter_test_paths()
        for path in paths:
            dp.cooccurrence_extraction.extract_cooccurrence_from_file(
                path, window, cooccurrences)

        for (token1, token2), expected_count in expected_counts.items():
            idx1 = cooccurrences.dictionary.get_id(token1)
            idx2 = cooccurrences.dictionary.get_id(token2)
            self.assertEqual(cooccurrences.counts[idx1, idx2], expected_count)

        self.assertEqual(
            sum(cooccurrences.counts.values()),
            sum(expected_counts.values())
        )
        


    def test_extract_and_write_all(self):

        window = 2
        expected_counts = self.EXPECTED_COUNTS_DOC1 + self.EXPECTED_COUNTS_DOC2 

        paths = list(dp.path_iteration.iter_test_paths())
        write_path = dp.path_iteration.get_test_write_path(
            'test_extract_and_write_all')
        if os.path.exists(write_path):
            shutil.rmtree(write_path)

        cooccurrences = dp.cooccurrence_extraction.extract_and_write_all(
            paths, write_path, window, verbose=False)

        for (token1, token2), count in expected_counts.items():
            idx1 = cooccurrences.dictionary.get_id(token1)
            idx2 = cooccurrences.dictionary.get_id(token2)
            self.assertEqual(cooccurrences.counts[idx1, idx2], count)
            self.assertEqual(cooccurrences.Nxx[idx1, idx2], count)

        # Try loading from what was written, it should be the same as what
        # was returned.
        cooccurrences2 = h.cooc_stats.CoocStats.load(
            write_path, verbose=False)
        self.assertEqual(cooccurrences.counts, cooccurrences2.counts)
        self.assertTrue(np.allclose(cooccurrences.Nxx, cooccurrences2.Nxx))
        self.assertTrue(np.allclose(cooccurrences.Nx, cooccurrences2.Nx))

        shutil.rmtree(write_path)



if __name__ == '__main__':
    main()


