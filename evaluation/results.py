
class ResultsObtainedError(Exception): pass


class ResultsHolder(object):
    """
    Helpful, simple class for holding final results and then
    serializing them in different ways.
    """

    def __init__(self, exp_name):
        self.name = exp_name
        self.results_by_dataset = {}
        self.result_keys = set()


    @classmethod
    def _str_format(cls, key_str):
        return key_str.replace('-', ' ').replace('_', ' ').capitalize()


    def add_ds_results(self, ds_name, results_dict):
        if ds_name in self.results_by_dataset:
            raise ResultsObtainedError('We have already obtained the'
                                       'results for {}!'.format(ds_name))

        # assume results_dict is a dictionary
        self.results_by_dataset[ds_name] = results_dict
        self.result_keys.update(results_dict.keys())


    def pretty_print(self):
        print('Results on {}...'.format(self.name))
        spacing = '   '
        max_res_len = max(map(len, self.result_keys)) + 1
        for ds, res in self.results_by_dataset.items():
            print('\n{}Dataset {}:'.format(spacing, self._str_format(ds)))
            for key, value in res.items():
                if type(value) == list: continue
                print('{}{:{fill}} - {:3.4f}'.format(spacing*2,
                                                     self._str_format(key),
                                                     value,
                                                     fill=max_res_len))
