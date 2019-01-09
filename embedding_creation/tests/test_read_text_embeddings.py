import os
import shutil
from unittest import TestCase, main
import torch
import hilbert as h
import embedding_creation as ec

class TestTranslateEmbeddings(TestCase):

    def test_translate_w2v_embeddings(self):

        subtests = [
            ('word2vec', True, False),
            ('word2vec', False, False),
            ('glove', True, False),
            #('glove', True, True),
            ('glove', False, False),
        ]

        for embedding_type, has_covectors, has_bias in subtests:

            print(embedding_type, has_covectors, has_bias)
            path_fragment = 'w2v-' if embedding_type == 'word2vec' else 'glv-'
            path_fragment += 'covecs' if has_covectors else 'nocovecs'
            path_fragment += '-bias' if has_bias else ''

            in_path = os.path.join(
                ec.CONSTANTS.TEST_DATA_DIR, 'vecs-{}.txt'.format(path_fragment))
            out_path = os.path.join(
                ec.CONSTANTS.TEST_DATA_DIR, 'test-embeddings')
            expected_embeddings_path = os.path.join(
                ec.CONSTANTS.TEST_DATA_DIR, 'embs-{}'.format(path_fragment))
            dictionary_path = os.path.join(
                expected_embeddings_path, 'dictionary')

            ec.translate_text_embeddings.translate_embeddings(
                embedding_type=embedding_type,
                in_path=in_path,
                out_path=out_path,
                dictionary_path=dictionary_path,
                allow_mismatch=False,
                has_bias=has_bias,
                has_covectors=has_covectors
            )

            expected_embeddings = h.embeddings.Embeddings.load(
                expected_embeddings_path)
            found_embeddings = h.embeddings.Embeddings.load(out_path)

            self.assertTrue(torch.allclose(
                found_embeddings.V, expected_embeddings.V))
            self.assertEqual(
                found_embeddings.dictionary.tokens,
                expected_embeddings.dictionary.tokens
            )
            if has_covectors:
                self.assertTrue(torch.allclose(
                    found_embeddings.W, expected_embeddings.W))
            if has_bias:
                self.assertTrue(torch.allclose(
                    found_embeddings.v_bias, expected_embeddings.v_bias))
            if has_bias and has_covectors:
                import pdb; pdb.set_trace()
                self.assertTrue(torch.allclose(
                    found_embeddings.w_bias, expected_embeddings.w_bias))

            #unsorted_dictionary_path = os.path.join(
            #    ec.CONSTANTS.TEST_DATA, 'unsorted-dictionary')

            # Clean up.
            shutil.rmtree(out_path)


if __name__ == '__main__':
    main()
