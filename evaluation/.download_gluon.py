### Assumes you have MXNET and GLUONNLP installed.
# alternatively, just download the actual datasets from my website with curl (in bash script)

import gluonnlp as nlp
import gluonnlp.data as d

HILBERT_DATASET_DIR = 'datasets/'

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
    with open('{}{}.txt'.format(HILBERT_DATASET_DIR, name), 'w') as f:
        f.write(','.join(header))
        f.write('\n')
        for sample in ds:
            f.write(','.join(map(str, sample)))
            f.write('\n')

