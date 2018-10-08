### Assumes you have MXNET and GLUONNLP installed.
# alternatively, just download the actual datasets from my website with curl (in bash script)
import re
import gluonnlp as nlp
import gluonnlp.data as d
import multiprocessing as mp

datasets = [
    ('WordSim353_relatedness', lambda: d.WordSim353(segment='relatedness')), 
    ('WordSim353_similarity', lambda: d.WordSim353(segment='similarity')), 
    #('MEN_train', lambda: d.MEN(segment='train')), # nonexistent
    ('MEN_dev', lambda: d.MEN(segment='dev')), 
    ('MEN_test', lambda: d.MEN(segment='test')), 
    ('RadinskyMTurk', d.RadinskyMTurk), 
    ('RareWords', d.RareWords), 
    ('SimLex999', d.SimLex999), 
    ('SimVerb3500', d.SimVerb3500), 
    ('SemEval17Task2_trial', lambda: d.SemEval17Task2(segment='trial')), 
    ('SemEval17Task2_test', lambda: d.SemEval17Task2(segment='test')), 
    ('BakerVerb143', d.BakerVerb143), 
    ('YangPowersVerb130', d.YangPowersVerb130), 
    ('GoogleAnalogyTestSet', d.GoogleAnalogyTestSet), 
    ('BiggerAnalogyTestSet', d.BiggerAnalogyTestSet), 
]

for name, ds_const in datasets:
    ds = ds_const() # declares it and downloads it to ~/.mxnet
    print('Downloaded {}'.format(name))
    print('\t example sample: {}'.format(ds[10]))
    
    # header for the CSV file
    header = ['word1', 'word2']
    header += ['word3', 'word4'] if 'Analogy' in name else ['score']
    for sample in ds:
        assert len(sample) == len(header)
        for val in sample:
            if type(val) == str:
                assert ',' not in val
                assert '\n' not in val

    # serialize to our directory
    with open('unsup_datasets/{}.csv'.format(name.lower()), 'w') as f:
        f.write(','.join(header))
        f.write('\n')
        for sample in ds:
            f.write(','.join(map(str, sample)))
            f.write('\n')



####### will also be doing the same with the IMDB data
datasets = [
    ('imdb_train', d.IMDB(segment='train')),
    ('imdb_test', d.IMDB(segment='test'))
]

# tokenize and set max length of 500
tokenizer = d.SpacyTokenizer('en')
length_clip = d.ClipSequence(500)

# keep some of the punct, periods and exclamation points. remove all else
REPLACE_NO_SPACE = re.compile("(\*)|(\;)|(\:)||(\')||(\,)|(\")|(\()|(\))|(\[)|(\])|(\t)|(\n)")

# remove the stupid html tags that snuck into the dataset
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess(sample):
    st, label = sample
    st = REPLACE_NO_SPACE.sub("", st.lower())
    st = REPLACE_WITH_SPACE.sub(" ", st)
    st = re.sub(' +', ' ', st) # replace all extra spaces with just 1
    label = 'pos' if label > 5 else 'neg'
    return length_clip(tokenizer(st)), label


# iterate and multiprocess for speeeed
for name, data in datasets:
    with mp.Pool() as pool:
        dataset = list(pool.map(preprocess, data))
    print('{} has {} samples'.format(name, len(dataset)))
    print('\tExample: {}'.format(' '.join(dataset[150][0])))
    print('\tExample: {}'.format(' '.join(dataset[500][0])))
    print()

    with open('sup_datasets/{}.csv'.format(name), 'w') as f:
        f.write('tokenized_text,sentiment\n')
        for sample, label in dataset:
            s = '{},{}\n'.format(' '.join(sample), label)
            assert s.count(',') == 1
            f.write(s)

    



