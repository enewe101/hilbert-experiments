import os
import sys
import shared
import subprocess

def make_glove_embeddings(out_dir_name='glove', min_count=5, d=300, iterations=15, window_size=5, threads=8, X_max=100):
    corpus = shared.CONSTANTS.TOKENIZED_CAT_FULL_PATH
    out_dir = os.path.join(shared.CONSTANTS.EMBEDDINGS_DIR, out_dir_name)
    command = [
        shared.CONSTANTS.LOCAL_GLOVE_EXECUTABLE_PATH,
        corpus,
        out_dir, 
        shared.CONSTANTS.GLOVE_EXECUTABLE_PATH,
        str(min_count),
        str(d),
        str(iterations),
        str(window_size),
        str(threads),
        str(X_max)
    ]
    print(' '.join(command))
    return subprocess.run(command)


if __name__ == '__main__':

    replicate = int(sys.argv[1])
    out_dir_name='std-glv-10k-rep{replicate}'.format(replicate=replicate)
    min_count = 17921
    threads = 16
    print(
        "making embeddings at {out}.  Min count {min_count}. "
        "Replicant {rep}.  Threads {threads}.".format(
            out=out_dir_name,
            min_count=str(min_count),
            rep=str(replicate),
            threads=str(threads)
        )
    )
    make_glove_embeddings(
        out_dir_name=out_dir_name,
        min_count=min_count,
        threads=threads
    )

