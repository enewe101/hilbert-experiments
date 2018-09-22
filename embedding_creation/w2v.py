import os
import sys
import shared
import subprocess

def make_w2v_embeddings(
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
        shared.CONSTANTS.TOKENIZED_CAT_DIR,
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

#if __name__ == '__main__':
#
#    replicate = int(sys.argv[1])
#    out_dir_name='std-w2v-10k-rep{replicate}'.format(replicate=replicate)
#    min_count = 17921
#    threads = 16
#    print(
#        "making embeddings at {out}.  Min count {min_count}. "
#        "Replicant {rep}.  Threads {threads}.".format(
#            out=out_dir_name,
#            min_count=str(min_count),
#            rep=str(replicate),
#            threads=str(threads)
#        )
#    )
#    make_glove_embeddings(
#        out_dir_name=out_dir_name,
#        min_count=min_count,
#        threads=threads
#    )
#
