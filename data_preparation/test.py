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


class TestDictionary(TestCase):

    def get_test_dictionary(self):
        tokens = dp.path_iteration.get_test_tokens()
        return tokens, dp.dictionary.Dictionary(tokens)


    def test_dictionary(self):
        tokens, dictionary = self.get_test_dictionary()
        for token in tokens:
            dictionary.add_token(token)

        self.assertEqual(set(tokens), set(dictionary.tokens))
        expected_token_ids = {
            token:idx for idx, token in enumerate(dictionary.tokens)}
        self.assertEqual(expected_token_ids, dictionary.token_ids)


    def test_save_load_dictionary(self):
        write_path = dp.path_iteration.get_test_write_path('test.dictionary')

        # Remove files that could be left from a previous test.
        if os.path.exists(write_path):
            os.remove(write_path)


        tokens, dictionary = self.get_test_dictionary()
        dictionary.save(write_path)
        loaded_dictionary = dp.dictionary.Dictionary.load(
            write_path)

        self.assertEqual(loaded_dictionary.tokens, dictionary.tokens)
        self.assertEqual(loaded_dictionary.token_ids, dictionary.token_ids)

        # Cleanup
        os.remove(write_path)


class TestCooccurrenceStatistics(TestCase):

    def get_test_cooccurrence_stats(self):
        DICTIONARY = dp.dictionary.Dictionary([
            'banana', 'socks', 'car', 'field'])
        COUNTS = {
            (0,1):3, (1,0):3,
            (0,3):1, (3,0):1,
            (2,1):1, (1,2):1,
            (0,2):1, (2,0):1
        }
        DIJ = ([3,1,1,1,3,1,1,1], ([0,0,2,0,1,3,1,2], [1,3,1,2,0,0,2,0]))
        ARRAY = np.array([[0,3,1,1],[3,0,1,0],[1,1,0,0],[1,0,0,0]])
        return DICTIONARY, COUNTS, DIJ, ARRAY


    def test_invalid_arguments(self):
        dictionary, counts, dij, array = self.get_test_cooccurrence_stats()

        # Can make an empty CooccurrenceStatistics instance.
        h.cooc_stats.CooccurrenceStatistics()

        # Can make a non-empty CooccurrenceStatistics instance using counts and
        # a matching dictionary.
        h.cooc_stats.CooccurrenceStatistics(dictionary, counts)

        # Must supply a dictionary to make a  non-empty CooccurrenceStatistics
        # instance when using counts.
        with self.assertRaises(ValueError):
            h.cooc_stats.CooccurrenceStatistics(
                counts=counts)

        # Can make a non-empty CooccurrenceStatistics instance using Nxx and
        # a matching dictionary.
        Nxx = sparse.coo_matrix(dij).tocsr()
        h.cooc_stats.CooccurrenceStatistics(dictionary, counts)

        # Must supply a dictionary to make a  non-empty CooccurrenceStatistics
        # instance when using Nxx.
        with self.assertRaises(ValueError):
            h.cooc_stats.CooccurrenceStatistics(Nxx=Nxx)

        # Cannot provide both an Nxx and counts
        with self.assertRaises(ValueError):
            h.cooc_stats.CooccurrenceStatistics(
                dictionary, counts, Nxx=Nxx)


    def test_add_when_basis_is_counts(self):
        dictionary, counts, dij, array = self.get_test_cooccurrence_stats()
        cooccurrence = h.cooc_stats.CooccurrenceStatistics(
            dictionary, counts, verbose=False)
        cooccurrence.add('banana', 'rice')
        self.assertEqual(cooccurrence.dictionary.get_id('rice'), 4)
        expected_counts = Counter(counts)
        expected_counts[0,4] += 1
        self.assertEqual(cooccurrence.counts, expected_counts)


    def test_add_when_basis_is_Nxx(self):

        dictionary, counts, dij, array = self.get_test_cooccurrence_stats()
        Nxx = sparse.coo_matrix(dij)

        Nx = np.array(np.sum(Nxx, axis=1)).reshape(-1)

        # Create a cooccurrence instance using counts
        cooccurrence = h.cooc_stats.CooccurrenceStatistics(
            dictionary, Nxx=Nxx, verbose=False)

        # Currently the cooccurrence instance has no internal counter for
        # cooccurrences, because it is based on the sparse array.
        self.assertEqual(cooccurrence._counts, None)
        self.assertTrue(np.allclose(cooccurrence._Nxx.toarray(), Nxx.toarray()))
        self.assertTrue(np.allclose(cooccurrence.denseNxx, Nxx.toarray()))
        self.assertTrue(np.allclose(cooccurrence.Nx, Nx))

        # Adding more cooccurrence statistics will force it to decompile into
        # a counter, then add to the counter.  This will cause the stale Nxx
        # arrays to be dropped.
        cooccurrence.add('banana', 'rice')
        cooccurrence.add('rice', 'banana')
        expected_counts = Counter(counts)
        expected_counts[4,0] += 1
        expected_counts[0,4] += 1
        self.assertEqual(cooccurrence._counts, expected_counts)
        self.assertEqual(cooccurrence._Nxx, None)
        self.assertEqual(cooccurrence._denseNxx, None)
        self.assertEqual(cooccurrence._Nx, None)

        # Asking for denseNxx forces it to sync itself.  
        # Ensure it it obtains the correct cooccurrence matrix
        expected_Nxx = np.append(array, [[1],[0],[0],[0]], axis=1)
        expected_Nxx = np.append(expected_Nxx, [[1,0,0,0,0]], axis=0)
        expected_Nx = np.sum(expected_Nxx, axis=1)
        self.assertTrue(np.allclose(cooccurrence.denseNxx, expected_Nxx))
        self.assertTrue(np.allclose(cooccurrence.Nx, expected_Nx))
        self.assertTrue(cooccurrence.synced)


    def test_uncompile(self):
        dictionary, counts, dij, array = self.get_test_cooccurrence_stats()
        Nxx = sparse.coo_matrix(dij)
        Nx = np.array(np.sum(Nxx, axis=1)).reshape(-1)

        # Create a cooccurrence instance using Nxx
        cooccurrence = h.cooc_stats.CooccurrenceStatistics(
            dictionary, Nxx=Nxx, verbose=False)
        self.assertEqual(cooccurrence._counts, None)

        cooccurrence.decompile()
        self.assertEqual(cooccurrence._counts, counts)



    def test_compile(self):

        dictionary, counts, dij, array = self.get_test_cooccurrence_stats()

        # Create a cooccurrence instance using counts
        cooccurrence = h.cooc_stats.CooccurrenceStatistics(
            dictionary, counts, verbose=False)

        # The cooccurrence instance has no Nxx array, but it will be calculated
        # when we try to access it directly.
        self.assertEqual(cooccurrence._Nxx, None)
        self.assertEqual(cooccurrence._denseNxx, None)
        self.assertEqual(cooccurrence._Nx, None)
        self.assertTrue(np.allclose(cooccurrence.Nxx.todense(), array))
        self.assertTrue(np.allclose(cooccurrence.denseNxx, array))
        self.assertTrue(np.allclose(cooccurrence.Nx, np.sum(array, axis=1)))

        # We can still add more counts.  This causes it to drop the stale Nxx.
        cooccurrence.add('banana', 'rice')
        cooccurrence.add('rice', 'banana')
        self.assertEqual(cooccurrence._Nxx, None)
        self.assertEqual(cooccurrence._Nx, None)
        self.assertEqual(cooccurrence._denseNxx, None)

        # Asking for an array forces it to sync itself.  This time start with
        # denseNxx.
        expected_Nxx = np.append(array, [[1],[0],[0],[0]], axis=1)
        expected_Nxx = np.append(expected_Nxx, [[1,0,0,0,0]], axis=0)
        self.assertTrue(np.allclose(cooccurrence.denseNxx, expected_Nxx))
        self.assertTrue(np.allclose(cooccurrence.Nxx.todense(), expected_Nxx))
        self.assertTrue(np.allclose(
            cooccurrence.Nx, np.sum(expected_Nxx, axis=1)))

        # Adding more counts once again causes it to drop the stale Nxx.
        cooccurrence.add('banana', 'field')
        cooccurrence.add('field', 'banana')
        self.assertEqual(cooccurrence._Nxx, None)
        self.assertEqual(cooccurrence._Nx, None)
        self.assertEqual(cooccurrence._denseNxx, None)

        # Asking for an array forces it to sync itself.  This time start with
        # Nx.
        expected_Nxx[0,3] += 1
        expected_Nxx[3,0] += 1
        self.assertTrue(np.allclose(
            cooccurrence.Nx, np.sum(expected_Nxx, axis=1)))
        self.assertTrue(np.allclose(cooccurrence.denseNxx, expected_Nxx))
        self.assertTrue(np.allclose(cooccurrence.Nxx.todense(), expected_Nxx))



    def test_sort(self):
        unsorted_dictionary = dp.dictionary.Dictionary([
            'field', 'car', 'socks', 'banana'
        ])
        unsorted_counts = {
            (0,3): 1, (3,0): 1,
            (1,2): 1, (2,1): 1,
            (1,3): 1, (3,1): 1,
            (2,3): 3, (3,2): 3
        }
        unsorted_Nxx = np.array([
            [0,0,0,1],
            [0,0,1,1],
            [0,1,0,3],
            [1,1,3,0],
        ])
        sorted_dictionary = dp.dictionary.Dictionary([
            'banana', 'socks', 'car', 'field'])
        sorted_counts = {
            (0,1):3, (1,0):3,
            (0,3):1, (3,0):1,
            (2,1):1, (1,2):1,
            (0,2):1, (2,0):1
        }
        sorted_array = np.array([
            [0,3,1,1],
            [3,0,1,0],
            [1,1,0,0],
            [1,0,0,0]
        ])
        cooccurrence = h.cooc_stats.CooccurrenceStatistics(
            unsorted_dictionary, unsorted_counts, verbose=False
        )
        self.assertTrue(np.allclose(cooccurrence.denseNxx, sorted_array))
        self.assertEqual(cooccurrence.counts, sorted_counts)
        self.assertEqual(
            cooccurrence.dictionary.tokens, sorted_dictionary.tokens)


    def test_save_load(self):

        write_path = dp.path_iteration.get_test_write_path(
            'test-save-load-cooccurrences')
        if os.path.exists(write_path):
            shutil.rmtree(write_path)

        dictionary, counts, dij, array = self.get_test_cooccurrence_stats()

        # Create a cooccurrence instance using counts
        cooccurrence = h.cooc_stats.CooccurrenceStatistics(
            dictionary, counts, verbose=False)

        # Save it, then load it
        cooccurrence.save(write_path)
        cooccurrence2 = h.cooc_stats.CooccurrenceStatistics.load(
            write_path, verbose=False)

        self.assertEqual(
            cooccurrence2.dictionary.tokens, 
            cooccurrence.dictionary.tokens
        )
        self.assertEqual(cooccurrence2.counts, cooccurrence.counts)
        self.assertTrue(np.allclose(
            cooccurrence2.denseNxx, 
            cooccurrence.denseNxx
        ))
        self.assertTrue(np.allclose(cooccurrence2.Nx, cooccurrence.Nx))

        shutil.rmtree(write_path)


    def test_density(self):
        dictionary, counts, dij, array = self.get_test_cooccurrence_stats()
        cooccurrence = h.cooc_stats.CooccurrenceStatistics(
            dictionary, counts, verbose=False)
        self.assertEqual(cooccurrence.density(), 0.5)
        self.assertEqual(cooccurrence.density(2), 0.125)


    def test_truncate(self):
        dictionary, counts, dij, array = self.get_test_cooccurrence_stats()
        cooccurrence = h.cooc_stats.CooccurrenceStatistics(
            dictionary, counts, verbose=False)
        cooccurrence.truncate(3)
        truncated_array = np.array([
            [0,3,1],
            [3,0,1],
            [1,1,0],
        ])

        self.assertTrue(np.allclose(
            cooccurrence.Nxx.todense(), truncated_array))


    def test_dict_to_sparse(self):
        dictionary, counts, dij, array = self.get_test_cooccurrence_stats()
        csr_matrix = h.cooc_stats.dict_to_sparse(counts)
        self.assertTrue(isinstance(csr_matrix, sparse.csr_matrix))
        self.assertTrue(np.allclose(csr_matrix.todense(), array))




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
        cooccurrences = h.cooc_stats.CooccurrenceStatistics()

        window = 2
        paths = dp.path_iteration.iter_test_paths()
        for path in paths:
            dp.extract_cooccurrence.extract_cooccurrence_from_file(
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

        cooccurrences = dp.extract_cooccurrence.extract_and_write_all(
            paths, write_path, window, verbose=False)

        for (token1, token2), count in expected_counts.items():
            idx1 = cooccurrences.dictionary.get_id(token1)
            idx2 = cooccurrences.dictionary.get_id(token2)
            self.assertEqual(cooccurrences.counts[idx1, idx2], count)
            self.assertEqual(cooccurrences.Nxx[idx1, idx2], count)

        # Try loading from what was written, it should be the same as what
        # was returned.
        cooccurrences2 = h.cooc_stats.CooccurrenceStatistics.load(
            write_path, verbose=False)
        self.assertEqual(cooccurrences.counts, cooccurrences2.counts)
        self.assertTrue(np.allclose(
            cooccurrences.Nxx.todense(), cooccurrences2.Nxx.todense()))
        self.assertTrue(np.allclose(
            cooccurrences.denseNxx, cooccurrences2.denseNxx))
        self.assertTrue(np.allclose(cooccurrences.Nx, cooccurrences2.Nx))

        shutil.rmtree(write_path)



if __name__ == '__main__':
    main()


