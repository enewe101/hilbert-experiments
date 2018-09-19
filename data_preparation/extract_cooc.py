#!/usr/bin/env python
import os
import sys
import itertools

import data_preparation as dp


if __name__ == '__main__':

    usage = (
        'Usage\n'
        'extract_cooc.py <window-size-integer> <out-path> <sector> '
        '[<sector> ...]\n'
        '\tor\n'
        'extract_cooc.py <window-size-integer> <out-path> all'
    )
    try:
        window_size = int(sys.argv[1])
        out_path = sys.argv[2]
        sectors = sys.argv[3:]
    except (IndexError, ValueError):
        print(usage)
        sys.exit(1)

    if len(sectors) == 1 and sectors[0] == 'all':
        sectors = [
            i+j+k 
            for i,j,k in itertools.product('0123456789abcdef', repeat=3)
        ]

    else:
        dp.path_iteration.raise_if_sectors_not_all_valid(sectors)

    # Name file to contain cooccurrence statistics, and get its full path.
    if len(sectors) > 1:
        out_dir_name = '%s-%s-%dw' % (sectors[0], sectors[-1], window_size)
    else:
        out_dir_name = '%s-%dw' % (sectors[0], window_size)
    out_path = os.path.join(out_path, out_dir_name)

    dp.cooccurrence_extraction.extract_sectors(
        sectors, out_path, window_size)


