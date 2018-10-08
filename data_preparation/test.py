import os
import shutil
import random
from collections import Counter

import numpy as np
from scipy import sparse
from unittest import main, TestCase

import shared
import hilbert as h
import data_preparation as dp

import warnings
import logging
logging.captureWarnings(True)




#
#   Path iteration is not really needed any more.
#
#class TestPathIteration(TestCase):
#
#    def test_iter_gigaword_fnames_in_sector(self):
#        expected_fname_path = os.path.join(
#            dp.CONSTANTS.TEST_DATA_DIR, '000.ls')
#        with open(expected_fname_path) as f:
#            expected_fnames = f.read().split()
#        expected_fnames = { ('000', fname) for fname in expected_fnames }
#        found_fnames = set(
#            dp.path_iteration.iter_gigaword_fnames_in_sector('000'))
#        self.assertEqual(found_fnames, expected_fnames)
#
#
#    def test_iter_gigaword_paths_in_sector(self):
#        expected_fname_path = os.path.join(
#            dp.CONSTANTS.TEST_DATA_DIR, '000.ls')
#        with open(expected_fname_path) as f:
#            expected_fnames = f.read().split()
#        expected_paths = {
#            dp.path_iteration.get_gigaword_path('000', fname)
#            for fname in expected_fnames
#        }
#        found_paths = set(
#            dp.path_iteration.iter_gigaword_paths_in_sector('000'))
#        self.assertEqual(found_paths, expected_paths)
#
#
#    def test_get_gigaword_path(self):
#        expected_path = os.path.join(
#            shared.CONSTANTS.LOCAL_GIGAWORD_DIR,
#            '000', 'CoreNLP', 'xyz.txt.xml'
#        )
#        found_path =  dp.path_iteration.get_gigaword_path('000', 'xyz.txt.xml')
#        self.assertEqual(found_path, expected_path)
#
#
#    def test_get_tokenized_path(self):
#        expected_path = os.path.join(
#            shared.CONSTANTS.TOKENIZED_DIR, '000', 'xyz.txt.xml')
#        found_path =  dp.path_iteration.get_tokenized_path('000', 'xyz.txt.xml')
#        self.assertEqual(found_path, expected_path)
#
#
#    def test_iter_tokenized_fnames_in_sector(self):
#        expected_fname_path = os.path.join(
#            dp.CONSTANTS.TEST_DATA_DIR, '000.ls')
#        with open(expected_fname_path) as f:
#            expected_fnames = [
#                fname[:-4]  # Strip off the ".xml"
#                for fname in f.read().split()
#            ]
#        expected_paths = {('000', fname) for fname in expected_fnames}
#        found_paths = set(
#            dp.path_iteration.iter_tokenized_fnames_in_sector('000'))
#        self.assertEqual(found_paths, expected_paths)
#
#
#    def test_iter_tokenized_paths_in_sector(self):
#        expected_fname_path = os.path.join(
#            dp.CONSTANTS.TEST_DATA_DIR, '000.ls')
#        with open(expected_fname_path) as f:
#            expected_fnames = [
#                fname[:-4]  # Strip off the ".xml"
#                for fname in f.read().split()
#            ]
#        expected_paths = {
#            dp.path_iteration.get_tokenized_path('000', fname)
#            for fname in expected_fnames
#        }
#        found_paths = set(
#            dp.path_iteration.iter_tokenized_paths_in_sector('000'))
#        self.assertEqual(found_paths, expected_paths)


class MockUnigram(h.unigram.Unigram):
    def freq(self, token):
        return {
            "the": 0.1,
            "apple": 0.2,
            "doesn't": 0.3,
            "fall": 0.2,
            "far": 0.1,
            "from": 0.1,
            "tree": 0,
            ".": 0
        }[token]


class TestSampling(TestCase):

    def test_get_count_prob(self):

        token_list_len = 5
        drop_probs = np.array([0.2] * token_list_len)

        # These settings should have no effect on the test.
        window = 2
        thresh = 1

        # Make a sampler (we need empty unigram and bigram instances to do this
        # but they have no direct involvement in the test).
        unigram = h.unigram.Unigram()
        bigram = h.bigram.Bigram(unigram)
        sampler = dp.bigram_sampler.SamplerW2V(bigram, window, thresh)

        found_count_prob = sampler.get_count_prob_right_low_mem(drop_probs)
        expected_count_prob = np.load(os.path.join(
            dp.CONSTANTS.TEST_DATA_DIR, 'count_probabilities.npy'))
        self.assertTrue(np.allclose(found_count_prob, expected_count_prob))


    def test_sample_w2v(self):
        np.random.seed(0)

        tokens = "the apple doesn't fall far from tree .".split()
        unigram = MockUnigram()
        for token in tokens:
            unigram.add(token)

        # First look for a match when threshold is high enough that no
        # tokens will be dropped.  Expected cooccurrences are just given 
        # by the "dynamic sampling" kernel sliding according to token position.
        window = 2
        thresh = 1
        expected_cooccurrences = np.array([
            [0, 1, 1/2, 0, 0, 0, 0, 0],
            [1, 0, 1, 1/2, 0, 0, 0, 0],
            [1/2, 1, 0, 1, 1/2, 0, 0, 0],
            [0, 1/2, 1, 0, 1, 1/2, 0, 0],
            [0, 0, 1/2, 1, 0, 1, 1/2, 0],
            [0, 0, 0, 1/2, 1, 0, 1, 1/2],
            [0, 0, 0, 0, 1/2, 1, 0, 1],
            [0, 0, 0, 0, 0, 1/2, 1, 0],
        ])
        bigram = h.bigram.Bigram(unigram)
        sampler = dp.bigram_sampler.SamplerW2V(bigram, window, thresh)
        #sampler.sample(tokens)
        sampler.sample_low_mem(tokens)
        self.assertTrue(
            np.allclose(bigram.Nxx.toarray(), expected_cooccurrences))


        # Now set the threshold so that some tokens get dropped.  Simulate
        # many replicates of the w2v dynamic sampling, and check that the
        # the average number of counts mathes the direct calculation of
        # counts expectation.
        window = 3
        thresh = 0.05
        mean_weight = np.zeros((8,8), dtype=np.float32)

        bigram = h.bigram.Bigram(unigram)
        sampler = dp.bigram_sampler.SamplerW2V(bigram, window, thresh)

        # Repeatedly bigram counting with common-word undersampling, to
        # estimate the expectation bigram counts.
        drop_probs = sampler.drop_prob(tokens)
        num_replicates = 1000
        for rep in range(num_replicates):
            kept_tokens = [
                token for token, prob in zip(tokens, drop_probs)
                if prob < random.random()
            ]
            for i in range(len(kept_tokens)):
                use_window = random.randint(1,window)
                for j in range(i-use_window, i+use_window+1):
                    if j == i or j < 0 or j >= len(kept_tokens):
                        continue
                    idx1 = unigram.dictionary.get_id(kept_tokens[i])
                    idx2 = unigram.dictionary.get_id(kept_tokens[j])
                    mean_weight[idx1, idx2] += 1

        # Calculate expectation bigram counts. 
        mean_weight = mean_weight / num_replicates

        # Use the sampler to get bigram counts.
        #sampler.sample(tokens)
        sampler.sample_low_mem(tokens)

        self.assertTrue(np.allclose(
            bigram.Nxx.toarray(), mean_weight, atol=0.05))
    

class TestUnigramExtraction(TestCase):

    def test_extract_unigram(self):

        corpus_path = dp.CONSTANTS.TOKENIZED_CAT_TEST_PATH
        with open(corpus_path) as test_corpus:
            tokens = test_corpus.read().strip().split()

        expected_counts = Counter(tokens)

        unigram = h.unigram.Unigram()
        dp.bigram_extraction.extract_unigram(
            corpus_path, unigram, verbose=False)
        self.assertEqual(len(unigram), len(expected_counts))
        for token in expected_counts:
            self.assertTrue(unigram.count(token), expected_counts[token])




class CoocStatsMock:
    def __init__(self):
        self.counts = Counter()
    def add(self, token1, token2, count=1):
        self.counts[token1, token2] += count


class TestBigramExtraction(TestCase):

    def test_extract_bigram(self):

        corpus_path = dp.CONSTANTS.TOKENIZED_CAT_TEST_PATH
        window = 3
        with open(corpus_path) as test_file:
            documents = [doc.split() for doc in test_file.read().split('\n')]
        unigram = h.unigram.Unigram()
        dp.bigram_extraction.extract_unigram(
            corpus_path, unigram, verbose=False)

        # Test extracting using flat weighting
        expected_counts = Counter()
        for doc in documents:
            for i in range(len(doc)):
                for j in range(i-window, i+window+1):
                    if j==i or j<0 or j>=len(doc):
                        continue
                    expected_counts[doc[i],doc[j]] += 1

        bigram = h.bigram.Bigram(unigram)
        sampler = dp.bigram_sampler.SamplerFlat(bigram, window)
        dp.bigram_extraction.extract_bigram(
            corpus_path, sampler, verbose=False)

        for token1 in unigram.dictionary.tokens:
            for token2 in unigram.dictionary.tokens:
                self.assertEqual(
                    bigram.count(token1, token2),
                    expected_counts[token1, token2]
                )

        # Test extracting using harmonic weighting
        expected_counts = Counter()
        for doc in documents:
            for i in range(len(doc)):
                for j in range(i-window, i+window+1):
                    if j==i or j<0 or j>=len(doc):
                        continue
                    expected_counts[doc[i],doc[j]] += 1.0/abs(i-j)

        bigram = h.bigram.Bigram(unigram)
        sampler = dp.bigram_sampler.SamplerHarmonic(bigram, window)
        dp.bigram_extraction.extract_bigram(
            corpus_path, sampler, verbose=False)

        for token1 in unigram.dictionary.tokens:
            for token2 in unigram.dictionary.tokens:
                self.assertEqual(
                    bigram.count(token1, token2),
                    expected_counts[token1, token2]
                )


        # Test dynamic weight
        expected_counts = Counter()
        random.seed(0)
        for doc in documents:
            for i in range(len(doc)):
                for j in range(i-window, i+window+1):
                    if j==i or j<0 or j>=len(doc):
                        continue
                    d = abs(i-j)
                    weight = (window - d + 1) / window
                    expected_counts[doc[i],doc[j]] += weight


        thresh = 1 # Disables common-word undersampling
        bigram = h.bigram.Bigram(unigram)
        sampler = dp.bigram_sampler.SamplerW2V(bigram, window, thresh)
        dp.bigram_extraction.extract_bigram(
            corpus_path, sampler, verbose=False)

        for token1 in unigram.dictionary.tokens:
            for token2 in unigram.dictionary.tokens:
                self.assertEqual(
                    bigram.count(token1, token2),
                    expected_counts[token1, token2]
                )


    #def test_cooccurrence_samplers(self):

    #    window_size = 5
    #    tokens = [str(i) for i in range(10)]

    #    # Test flat weight
    #    expected_counts = Counter()
    #    for i in range(len(tokens)):
    #        for j in range(i-window_size, i+window_size+1):
    #            if j==i or j<0 or j>=len(tokens):
    #                continue
    #            expected_counts[tokens[i],tokens[j]] += 1
    #    flat_sampler = dp.bigram_extraction.get_sampler('flat')
    #    cooc_stats = CoocStatsMock()
    #    flat_sampler(tokens, cooc_stats, window_size)
    #    self.assertEqual(cooc_stats.counts, expected_counts)

    #    # Test harmonic weight
    #    expected_counts = Counter()
    #    for i in range(len(tokens)):
    #        for j in range(i-window_size, i+window_size+1):
    #            if j==i or j<0 or j>=len(tokens):
    #                continue
    #            expected_counts[tokens[i],tokens[j]] += 1.0/abs(i-j)
    #    harmonic_sampler = dp.bigram_extraction.get_sampler('harmonic')
    #    cooc_stats = CoocStatsMock()
    #    harmonic_sampler(tokens, cooc_stats, window_size)
    #    self.assertEqual(cooc_stats.counts, expected_counts)

    #    # Test dynamic weight
    #    expected_counts = Counter()
    #    random.seed(0)
    #    for i in range(len(tokens)):
    #        use_window = random.randint(1,window_size)
    #        for j in range(i-use_window, i+use_window+1):
    #            if j==i or j<0 or j>=len(tokens):
    #                continue
    #            expected_counts[tokens[i],tokens[j]] += 1
    #    dynamic_sampler = dp.bigram_extraction.get_sampler('dynamic')
    #    cooc_stats = CoocStatsMock()
    #    random.seed(0)
    #    dynamic_sampler(tokens, cooc_stats, window_size)
    #    self.assertEqual(cooc_stats.counts, expected_counts)



    #def test_extract_and_write_all(self):

    #    window = 2
    #    expected_counts = self.EXPECTED_COUNTS_DOC1 + self.EXPECTED_COUNTS_DOC2 

    #    paths = list(dp.path_iteration.iter_test_paths())
    #    write_path = dp.path_iteration.get_test_write_path(
    #        'test_extract_and_write_all')
    #    if os.path.exists(write_path):
    #        shutil.rmtree(write_path)

    #    cooccurrences = dp.bigram_extraction.extract_and_write_all(
    #        paths, write_path, window, verbose=False)

    #    for (token1, token2), count in expected_counts.items():
    #        idx1 = cooccurrences.dictionary.get_id(token1)
    #        idx2 = cooccurrences.dictionary.get_id(token2)
    #        self.assertEqual(cooccurrences.counts[idx1, idx2], count)
    #        self.assertEqual(cooccurrences.Nxx[idx1, idx2], count)

    #    # Try loading from what was written, it should be the same as what
    #    # was returned.
    #    cooccurrences2 = h.cooc_stats.CoocStats.load(
    #        write_path, verbose=False)
    #    self.assertEqual(cooccurrences.counts, cooccurrences2.counts)
    #    self.assertTrue(np.allclose(
    #        cooccurrences.Nxx.toarray(), cooccurrences2.Nxx.toarray()))
    #    self.assertTrue(np.allclose(cooccurrences.Nx, cooccurrences2.Nx))

    #    shutil.rmtree(write_path)



if __name__ == '__main__':
    main()


