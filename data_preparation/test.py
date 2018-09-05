import os
import shutil
from itertools import izip
from collections import Counter

import numpy as np
from scipy import sparse
from unittest import main, TestCase

import data_preparation as dp


class TestDictionary(TestCase):

    def get_test_dictionary(self):
        tokens = dp.path_iteration.get_test_tokens()
        return tokens, dp.dictionary.Dictionary(tokens)


    def test_dictionary(self):
        tokens, dictionary = self.get_test_dictionary()
        for token in tokens:
            dictionary.add_token(token)

        self.assertItemsEqual(list(set(tokens)), dictionary.tokens)
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
        dp.cooccurrence_statistics.CooccurrenceStatistics()

        # Can make a non-empty CooccurrenceStatistics instance using counts and
        # a matching dictionary.
        dp.cooccurrence_statistics.CooccurrenceStatistics(dictionary, counts)

        # Must supply a dictionary to make a  non-empty CooccurrenceStatistics
        # instance when using counts.
        with self.assertRaises(ValueError):
            dp.cooccurrence_statistics.CooccurrenceStatistics(
                counts=counts)

        # Can make a non-empty CooccurrenceStatistics instance using Nxx and
        # a matching dictionary.
        Nxx = sparse.coo_matrix(dij).tocsr()
        dp.cooccurrence_statistics.CooccurrenceStatistics(dictionary, counts)

        # Must supply a dictionary to make a  non-empty CooccurrenceStatistics
        # instance when using Nxx.
        with self.assertRaises(ValueError):
            dp.cooccurrence_statistics.CooccurrenceStatistics(Nxx=Nxx)

        # Cannot provide both an Nxx and counts
        with self.assertRaises(ValueError):
            dp.cooccurrence_statistics.CooccurrenceStatistics(
                dictionary, counts, Nxx=Nxx)


    def test_add_when_basis_is_counts(self):
        dictionary, counts, dij, array = self.get_test_cooccurrence_stats()
        cooccurrence = dp.cooccurrence_statistics.CooccurrenceStatistics(
            dictionary, counts)
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
        cooccurrence = dp.cooccurrence_statistics.CooccurrenceStatistics(
            dictionary, Nxx=Nxx, verbose=False)

        # Currently the cooccurrence instance has no internal counter for
        # cooccurrences, because it is based on the sparse array.
        self.assertEqual(cooccurrence.counts, Counter())
        self.assertTrue(np.allclose(cooccurrence.denseNxx, Nxx.todense()))
        self.assertTrue(np.allclose(cooccurrence.Nx, Nx))
        self.assertEqual(cooccurrence.basis, 'Nxx')

        # Adding more cooccurrence statistics will force it to decompile into
        # a counter, then add to the counter.  This will cause Nxx to be
        # out of sync
        cooccurrence.add('banana', 'rice')
        cooccurrence.add('rice', 'banana')
        self.assertFalse(cooccurrence.synced)
        self.assertEqual(cooccurrence.basis, 'counts')

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

        # Create a cooccurrence instance using counts
        cooccurrence = dp.cooccurrence_statistics.CooccurrenceStatistics(
            dictionary, Nxx=Nxx, verbose=False)
        self.assertEqual(cooccurrence.counts, Counter())
        self.assertEqual(cooccurrence.basis, 'Nxx')

        cooccurrence.decompile()
        self.assertEqual(cooccurrence.counts, counts)
        self.assertEqual(cooccurrence.basis, 'counts')
        self.assertTrue(cooccurrence.synced)


    def test_compile(self):

        dictionary, counts, dij, array = self.get_test_cooccurrence_stats()

        # Create a cooccurrence instance using counts
        cooccurrence = dp.cooccurrence_statistics.CooccurrenceStatistics(
            dictionary, counts)

        # The cooccurrence instance is not synced.  Attempting to access
        # the Nxx value will cause it to sync.
        self.assertFalse(cooccurrence.synced)
        self.assertTrue(np.allclose(cooccurrence.denseNxx, array))
        self.assertTrue(cooccurrence.synced)

        # We can still add more counts.  This causes it to go back out of sync.
        cooccurrence.add('banana', 'rice')
        cooccurrence.add('rice', 'banana')
        self.assertFalse(cooccurrence.synced)

        # Asking for denseNxx forces it to sync itself.  
        # Ensure it it obtains the correct cooccurrence matrix
        expected_Nxx = np.append(array, [[1],[0],[0],[0]], axis=1)
        expected_Nxx = np.append(expected_Nxx, [[1,0,0,0,0]], axis=0)
        self.assertTrue(np.allclose(cooccurrence.denseNxx, expected_Nxx))
        self.assertTrue(cooccurrence.synced)


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

        print len(unsorted_dictionary)
        cooccurrence = dp.cooccurrence_statistics.CooccurrenceStatistics(
            unsorted_dictionary, unsorted_counts
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
        cooccurrence = dp.cooccurrence_statistics.CooccurrenceStatistics(
            dictionary, counts)

        # Save it, then load it
        cooccurrence.save(write_path)
        cooccurrence2 = dp.cooccurrence_statistics.CooccurrenceStatistics.load(
            write_path)

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
        pass


    def test_truncate(self):
        pass






class TestPathIteration(TestCase):

    def test_iter_gigaword_fnames_in_sector(self):
        expected_fname_path = os.path.join(dp.CONSTANTS.TEST_DATA_DIR, '000.ls')
        expected_fnames = open(expected_fname_path).read().split()
        expected_fnames = { ('000', fname) for fname in expected_fnames }
        found_fnames = set(
            dp.path_iteration.iter_gigaword_fnames_in_sector('000'))
        self.assertEqual(found_fnames, expected_fnames)


    def test_iter_gigaword_paths_in_sector(self):
        expected_fname_path = os.path.join(dp.CONSTANTS.TEST_DATA_DIR, '000.ls')
        expected_fnames = open(expected_fname_path).read().split()
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
        expected_fnames = [
            fname[:-4]  # Strip off the ".xml"
            for fname in open(expected_fname_path).read().split()
        ]
        expected_paths = {('000', fname) for fname in expected_fnames}
        found_paths = set(
            dp.path_iteration.iter_tokenized_fnames_in_sector('000'))
        self.assertEqual(found_paths, expected_paths)


    def test_iter_tokenized_paths_in_sector(self):
        expected_fname_path = os.path.join(dp.CONSTANTS.TEST_DATA_DIR, '000.ls')
        expected_fnames = [
            fname[:-4]  # Strip off the ".xml"
            for fname in open(expected_fname_path).read().split()
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
        dictionary = dp.dictionary.Dictionary()
        counter = Counter()

        window = 2
        paths = dp.path_iteration.iter_test_paths()
        for path in paths:
            dp.extract_cooccurrence.extract_cooccurrence_from_file(
                path, window, dictionary, counter, verbose=False)

        for (idx1, idx2), found_count in counter.items():
            token1 = dictionary.get_token(idx1)
            token2 = dictionary.get_token(idx2)
            self.assertEqual(found_count, expected_counts[token1, token2])

        self.assertEqual(sum(expected_counts.values()), sum(counter.values()))


    def test_dict_to_sparse(self):

        # Extract counts from the test file
        window = 2
        dictionary = dp.dictionary.Dictionary()
        counter = Counter()
        paths = dp.path_iteration.iter_test_paths()
        for path in paths:
            dp.extract_cooccurrence.extract_cooccurrence_from_file(
                path, window, dictionary, counter, verbose=False)

        # Make a sparse matrix
        csr_matrix = dp.cooccurrence_statistics.dict_to_sparse(counter)

        # Verify that the sparse matrix contained all the right data
        expected_counts = self.EXPECTED_COUNTS_DOC1 + self.EXPECTED_COUNTS_DOC2 
        coo_matrix = csr_matrix.tocoo()
        for i, j, v in izip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
            token1 = dictionary.get_token(i)
            token2 = dictionary.get_token(j)
            self.assertEqual(v, expected_counts[token1, token2])

        self.assertEqual(len(coo_matrix.row), len(expected_counts))


    def test_write_read_cooccurrence(self):

        write_path = dp.path_iteration.get_test_write_path('cooccurrence')

        if os.path.exists(write_path):
            shutil.rmtree(write_path)

        ## Remove files that could be left from a previous test.
        #if os.path.exists(os.path.join(write_path, 'Nxx.npz')):
        #    os.remove(os.path.join(write_path, 'Nxx.npz'))
        #if os.path.exists(os.path.join(write_path, 'Nx.npz')):
        #    os.remove(os.path.join(write_path, 'Nx.npz'))
        #if os.path.exists(os.path.join(write_path, 'dictionary')):
        #    os.remove(os.path.join(write_path, 'dictionary'))

        # Extract counts from the test file
        window = 2
        dictionary = dp.dictionary.Dictionary()
        counter = Counter()
        paths = dp.path_iteration.iter_test_paths()
        for path in paths:
            dp.extract_cooccurrence.extract_cooccurrence_from_file(
                path, window, dictionary, counter, verbose=False)

        # Make a sparse matrix
        N_xx = dp.cooccurrence_statistics.dict_to_sparse(counter)
        N_x = np.sum(N_xx, axis=1)

        dp.extract_cooccurrence.save_cooccurrence(
            dp.path_iteration.get_test_write_path('cooccurrence'),
            N_xx, N_x, dictionary
        )

        load_N_xx, load_N_x, load_dictionary = (
            dp.extract_cooccurrence.load_cooccurrence(
                dp.path_iteration.get_test_write_path('cooccurrence'))
        )

        self.assertTrue(np.allclose(
            N_xx.todense(), load_N_xx.todense()))
        self.assertTrue(np.allclose(N_x, load_N_x))

        self.assertEqual(load_dictionary.tokens, dictionary.tokens)
        self.assertEqual(load_dictionary.token_ids, dictionary.token_ids)

        # Cleanup
        shutil.rmtree(write_path)
        #os.remove(os.path.join(write_path, 'Nxx.npz'))
        #os.remove(os.path.join(write_path, 'Nx.npz'))
        #os.remove(os.path.join(write_path, 'dictionary'))



    def test_extract_and_write_all(self):

        window = 2
        expected_counts = self.EXPECTED_COUNTS_DOC1 + self.EXPECTED_COUNTS_DOC2 

        paths = list(dp.path_iteration.iter_test_paths())
        out_path = dp.path_iteration.get_test_write_path(
            'test_extract_and_write_all')
        if os.path.exists(out_path):
            shutil.rmtree(out_path)

        N_xx, N_x, dictionary = dp.extract_cooccurrence.extract_and_write_all(
            paths, out_path, window)

        for (token1, token2), count in expected_counts.iteritems():
            idx1, idx2 = dictionary.get_id(token1), dictionary.get_id(token2)
            self.assertEqual(N_xx[idx1,idx2], count)

        N_x_sorted = np.sort(N_x)[::-1]
        self.assertTrue(np.allclose(N_x_sorted, N_x))
        N_x_summed = np.array(np.sum(N_xx, axis=1)).reshape(-1)
        self.assertTrue(np.allclose(N_x_summed, N_x))

        shutil.rmtree(out_path)


    def test_truncate_cooccurrence(self):
        paths = list(dp.path_iteration.iter_test_paths())
        window = 2
        N_xx, N_x, dictionary = dp.extract_cooccurrence.extract_all(
            paths, window, verbose=False)
        self.assertEqual(N_xx.shape, (11,11))
        self.assertEqual(N_x.shape, (11,))
        self.assertEqual(len(dictionary.tokens), 11)
        k = 6
        N_xx_t, N_x_t, dictionary_t = (
            dp.extract_cooccurrence.truncate_cooccurrence(
                k, N_xx, N_x, dictionary))
        self.assertTrue(
            np.allclose(N_xx_t.todense(), N_xx[:6][:,:6].todense()))
        self.assertTrue(np.allclose(N_x_t, N_x[:6]))
        self.assertEqual(dictionary_t.tokens, dictionary.tokens[:6])
        self.assertEqual(
            dictionary_t.token_ids, 
            {t:i for i,t in enumerate(dictionary.tokens[:6])}
        )




if __name__ == '__main__':
    main()


