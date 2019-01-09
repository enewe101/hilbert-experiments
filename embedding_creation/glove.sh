#!/bin/bash
set -e

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

#make
#if [ ! -e text8 ]; then
#  if hash wget 2>/dev/null; then
#    wget http://mattmahoney.net/dc/text8.zip
#  else
#    curl -O http://mattmahoney.net/dc/text8.zip
#  fi
#  unzip text8.zip
#  rm text8.zip
#fi

CORPUS=$1
OUT_DIR=$2
BUILDDIR=$3
SAVE_FILE=$OUT_DIR/vectors
VERBOSE=2
MEMORY=4.0
VOCAB_MIN_COUNT=$4
VECTOR_SIZE=$5
MAX_ITER=$6
WINDOW_SIZE=$7
NUM_THREADS=$8
X_MAX=$9
SAVE_MODE=${10}
STATS_PATH=${11}

BINARY=0

#mkdir -p $OUT_DIR
echo

# If STATS_PATH isn't provided, extract stats from scratch and place them
# in the output directory.
if [ -z "$STATS_PATH" ]; then

    VOCAB_FILE=$OUT_DIR/vocab.txt
    COOCCURRENCE_FILE=$OUT_DIR/cooccurrence.bin
    COOCCURRENCE_SHUF_FILE=$OUT_DIR/cooccurrence.shuf.bin

    # Generate vocabulary file
    echo "$ $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT \
        -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
    $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT \
        -verbose $VERBOSE < $CORPUS > $VOCAB_FILE

    # Generate cooccurrence counts file
    echo "$ $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose \
        $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE"
    $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose \
        $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE

    # Shuffle cooccurrence counts file
    echo "$ $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE \
        < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
    $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE \
        < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE

# If STATS_PATH is provided, then look for stats there, don't extract them.
else
    VOCAB_FILE=$STATS_PATH/vocab.txt
    COOCCURRENCE_SHUF_FILE=$STATS_PATH/cooccurrence.shuf.bin

fi

# Train the model.
echo "$ $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -model $SAVE_MODE -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE"
$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -model $SAVE_MODE -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE

#if [ "$CORPUS" = 'text8' ]; then
#   if [ "$1" = 'matlab' ]; then
#       matlab -nodisplay -nodesktop -nojvm -nosplash < ./eval/matlab/read_and_evaluate.m 1>&2 
#   elif [ "$1" = 'octave' ]; then
#       octave < ./eval/octave/read_and_evaluate_octave.m 1>&2
#   else
#       echo "$ python eval/python/evaluate.py"
#       python eval/python/evaluate.py
#   fi
#fi
