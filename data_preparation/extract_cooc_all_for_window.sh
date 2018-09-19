window=$1
path=$2
usage="./extract_cooc_all_for_window.sh <window-size> <out-path>"
echo window $window
echo path $path
if [ -z "$window" ];
then
    echo $usage
    exit 1
fi
if ! [ "$window" -eq "$window" ] 2>/dev/null
then
    echo $usage
    exit 1;
fi
if [ -z "$path" ]
then 
    echo $usage
    exit 1
fi
echo ok
for i in {0..9} {a..f}; do for j in {0..9} {a..f}; do for k in {0..9} {a..f}; do echo $i$j$k; done; done; done | xargs -n256 -P16 ./extract_cooc.py $window $path
