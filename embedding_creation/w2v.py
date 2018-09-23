import os
import sys
import shared
import subprocess

def make_w2v_embeddings(
    corpus,
    out_dir_name='w2v',
    d=300,
    window_size=5,
    k=15,
    t_undersample=1e-5,
    threads=8,
    num_iterations=15
):

    command = [
        shared.CONSTANTS.LOCAL_W2V_EXECUTABLE_PATH,
        corpus,
        os.path.join(shared.CONSTANTS.EMBEDDINGS_DIR, out_dir_name),
        str(d),
        str(window_size),
        str(k),
        str(t_undersample),
        str(threads),
        str(num_iterations),
    ]
    print(' '.join(command))
    return subprocess.run(command)

if __name__ == '__main__':

    replicate = int(sys.argv[1])
    vocab_size = int(sys.argv[2])
    threads = int(sys.argv[3])

    if vocab_size != 10000 and vocab_size != 100000:
        print('vocab size should be 10000 or 100000.')
        sys.exit(1)
    if vocab_size == 10000:
        corpus = os.path.join(
            shared.CONSTANTS.TOKENIZED_CAT_DIR,
            'gigaword-tokenized-cat-10001.txt'
        )
        out_dir_name = 'std-w2v-10k-rep{rep}'.format(rep=replicate)

    elif vocab_size == 100000:
        corpus = os.path.join(
            shared.CONSTANTS.TOKENIZED_CAT_DIR,
            'gigaword-tokenized-cat-100255.txt'
        )
        out_dir_name = 'std-w2v-100k-rep{rep}'.format(rep=replicate)

    print(
        "Using vocab size {vocab_size}. "
        "Reading corpus {corpus}. "
        "Writing embeddings to {out}. "
        "Replicant {rep}.  "
        "Threads {threads}.".format(
            vocab_size=vocab_size,
            corpus=corpus,
            out=out_dir_name,
            rep=str(replicate),
            threads=str(threads)
        )
    )
    make_w2v_embeddings(
        corpus,
        out_dir_name,
        threads=threads
    )

