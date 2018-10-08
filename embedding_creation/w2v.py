import os
import sys
import shared
import subprocess
import embedding_creation as ec

def make_w2v_embeddings(
    corpus_fname,
    out_dir_name,
    d=300,
    window=5,
    k=15,
    t_undersample=1,
    threads=1,
    num_iterations=1000,
    min_count=1
):
    corpus_path = os.path.join(shared.CONSTANTS.TOKENIZED_CAT_DIR, corpus_fname)
    # Prepare the path to vectors, and make sure output dir exists.
    out_dir = os.path.join(shared.CONSTANTS.EMBEDDINGS_DIR, out_dir_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    vectors_path = os.path.join(out_dir, 'vectors.txt')

    # Set options that are considered as constants for our purposes.
    mode = 0
    hierarchical_sample = 0
    binary = 0
    cbow = 0 # 0 => skipgram; 1 => cbow

    # Run the word2vec program, using the established options settings.
    command = [
        #'time',
        ec.CONSTANTS.W2V_EXECUTABLE_PATH,
        '-train', corpus_path,
        '-output', vectors_path,
        '-cbow', str(cbow),
        '-size', str(d),
        '-window', str(window),
        '-negative', str(k),
        '-hs', str(hierarchical_sample),
        '-sample', str(t_undersample),
        '-threads', str(threads),
        '-binary', str(binary), 
        '-iter', str(num_iterations),
        '-min-count', str(min_count)
    ]
    print(' '.join(command))
    return subprocess.run(command)


if __name__ == '__main__':

    corpus_fname = sys.argv[1]
    out_dir_name = sys.argv[2]
    threads = int(sys.argv[3])

    print("Reading corpus {}.\nWriting embeddings to {}.\nThreads {}.\n".format(
        corpus_fname,
        out_dir_name,
        threads
    ))

    make_w2v_embeddings(
        corpus_fname,
        out_dir_name,
        threads=threads
    )

