from unittest import main, TestCase
import os
import data_preparation as prep
from collections import Counter
from itertools import izip
import numpy as np


class TestDictionary(TestCase):

    def get_test_dictionary(self):
        tokens = prep.path_iteration.get_test_tokens()
        return tokens, prep.extract_cooccurrence.Dictionary(tokens)


    def test_dictionary(self):
        tokens, dictionary = self.get_test_dictionary()
        for token in tokens:
            dictionary.add_token(token)

        self.assertItemsEqual(list(set(tokens)), dictionary.tokens)
        expected_token_ids = {
            token:idx for idx, token in enumerate(dictionary.tokens)}
        self.assertEqual(expected_token_ids, dictionary.token_ids)


    def test_save_load_dictionary(self):
        write_path = prep.path_iteration.get_test_write_path('test.dictionary')

        # Remove files that could be left from a previous test.
        if os.path.exists(write_path):
            os.remove(write_path)


        tokens, dictionary = self.get_test_dictionary()
        dictionary.save(write_path)
        loaded_dictionary = prep.extract_cooccurrence.Dictionary.load(
            write_path)

        self.assertEqual(loaded_dictionary.tokens, dictionary.tokens)
        self.assertEqual(loaded_dictionary.token_ids, dictionary.token_ids)

        # Cleanup
        os.remove(write_path)



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
        dictionary = prep.extract_cooccurrence.Dictionary()
        counter = Counter()

        paths = prep.path_iteration.iter_test_paths()
        for path in paths:
            prep.extract_cooccurrence.extract_cooccurrence_from_file(
                path, 2, dictionary, counter)

        for (idx1, idx2), found_count in counter.items():
            token1 = dictionary.get_token(idx1)
            token2 = dictionary.get_token(idx2)
            self.assertEqual(found_count, expected_counts[token1, token2])

        self.assertEqual(sum(expected_counts.values()), sum(counter.values()))


    def test_dict_to_sparse(self):

        # Extract counts from the test file
        dictionary = prep.extract_cooccurrence.Dictionary()
        counter = Counter()
        paths = prep.path_iteration.iter_test_paths()
        for path in paths:
            prep.extract_cooccurrence.extract_cooccurrence_from_file(
                path, 2, dictionary, counter)

        # Make a sparse matrix
        csr_matrix = prep.extract_cooccurrence.dict_to_sparse(counter)

        # Verify that the sparse matrix contained all the right data
        expected_counts = self.EXPECTED_COUNTS_DOC1 + self.EXPECTED_COUNTS_DOC2 
        coo_matrix = csr_matrix.tocoo()
        for i, j, v in izip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
            token1 = dictionary.get_token(i)
            token2 = dictionary.get_token(j)
            self.assertEqual(v, expected_counts[token1, token2])

        self.assertEqual(len(coo_matrix.row), len(expected_counts))


    def test_write_read_cooccurrence(self):

        write_path = prep.path_iteration.get_test_write_path('cooccurrences')

        # Remove files that could be left from a previous test.
        if os.path.exists(write_path + '.npz'):
            os.remove(write_path + '.npz')
        if os.path.exists(write_path + '.dictionary'):
            os.remove(write_path + '.dictionary')

        # Extract counts from the test file
        dictionary1 = prep.extract_cooccurrence.Dictionary()
        counter = Counter()
        paths = prep.path_iteration.iter_test_paths()
        for path in paths:
            prep.extract_cooccurrence.extract_cooccurrence_from_file(
                path, 2, dictionary1, counter)

        # Make a sparse matrix
        csr_matrix1 = prep.extract_cooccurrence.dict_to_sparse(counter)

        prep.extract_cooccurrence.save_cooccurrences(
            prep.path_iteration.get_test_write_path('cooccurrences'),
            csr_matrix1, dictionary1
        )

        csr_matrix2, dictionary2 = prep.extract_cooccurrence.load_cooccurrences(
            prep.path_iteration.get_test_write_path('cooccurrences')
        )

        self.assertTrue(
            np.allclose(csr_matrix1.todense(), csr_matrix2.todense()))

        self.assertEqual(dictionary2.tokens, dictionary1.tokens)
        self.assertEqual(dictionary2.token_ids, dictionary1.token_ids)

        # Cleanup
        os.remove(write_path + '.npz')
        os.remove(write_path + '.dictionary')





if __name__ == '__main__':
    main()


