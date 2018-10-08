import numpy as np
import hilbert as h


def get_sampler(name, **kwargs):
    if name == 'flat':
        return SamplerFlat(**kwargs)
    elif name == 'harmonic':
        return SamplerHarmonic(**kwargs)
    elif name == 'w2v':
        return SamplerW2V(**kwargs)
    else:
        raise ValueError(
            "Unexpected sampler type {}.  Expected 'flat', 'harmonic', "
            "or 'w2v'.".format(repr(name))
        )


class SamplerFlat:

    def __init__(self, bigram, window):
        self.bigram = bigram
        self.window = window

    def sample(self, tokens):
        for i in range(len(tokens)):
            for j in range(i-self.window, i+self.window+1):
                if j == i or j < 0 or j >= len(tokens):
                    continue
                self.bigram.add(tokens[i], tokens[j])


class SamplerHarmonic:

    def __init__(self, bigram, window):
        self.bigram = bigram
        self.window = window

    def sample(self, tokens):
        for i in range(len(tokens)):
            for j in range(i-self.window, i+self.window+1):
                if j == i or j < 0 or j >= len(tokens):
                    continue
                self.bigram.add(tokens[i], tokens[j], count=1.0/abs(i-j))


class SamplerW2V:

    def __init__(self, bigram, window, thresh, min_count=0):
        self.bigram = bigram
        self.window = window
        self.thresh = thresh
        self.min_count = min_count
        self.max_dist = max(2*window, window+5)


    def sample(self, tokens):
        # Filter out rare words
        tokens = [
            t for t in tokens 
            if self.bigram.unigram.count(t) >= self.min_count
        ]
        drop_probs = self.drop_prob(tokens)
        expected_weights = self.get_count_prob(drop_probs)
        expected_weight_right, expected_weight_left = expected_weights

        for i in range(len(tokens)):
            for c in range(self.max_dist):
                j = i - c
                if j < 0: continue
                self.bigram.add(
                    tokens[i], tokens[j], expected_weight_right[i,c])

        for i in range(len(tokens)):
            for c in range(self.max_dist):
                j = i + c
                if j >= len(tokens): continue
                self.bigram.add(
                    tokens[i], tokens[j], expected_weight_left[i,c])


    #def sample(self, tokens):
    #    drop_probs = self.drop_prob(tokens)
    #    expected_weight = self.get_count_prob(drop_probs)
    #    for i in range(len(tokens)):
    #        for j in range(len(tokens)):
    #            self.bigram.add(tokens[i], tokens[j], expected_weight[i,j])


    #def get_count_prob(self, drop_probs):
    #    weight = np.array(
    #        [0] + [(self.window-d)/self.window for d in range(self.window)] 
    #        + [0] * (len(drop_probs) - self.window - 1)
    #    ).reshape(1,-1,1)
    #    count_prob_right = self.get_count_prob_right(drop_probs)
    #    count_prob_left = self.get_count_prob_left(drop_probs)
    #    expected_weight = (count_prob_right + count_prob_left) * weight
    #    return expected_weight.sum(axis=1)


    def get_count_prob(self, drop_probs):
        weight = np.array(
            [0] + [(self.window-d)/self.window for d in range(self.window)] 
            + [0] * (self.max_dist - self.window - 1)
        ).reshape(1,-1,1)
        count_prob_right = self.get_count_prob_right(drop_probs)
        expected_weight_right = (count_prob_right * weight).sum(axis=1)
        count_prob_left = self.get_count_prob_left(drop_probs)
        expected_weight_left = (count_prob_left * weight).sum(axis=1)
        return expected_weight_right, expected_weight_left


    def get_count_prob_left(self, drop_prob):
        reverse_drop_prob = drop_prob[::-1]
        count_prob = self.get_count_prob_right(reverse_drop_prob)
        return count_prob[::-1]


    #def get_count_prob_left(self, drop_prob):
    #    reverse_drop_prob = drop_prob[::-1]
    #    count_prob = self.get_count_prob_right(reverse_drop_prob)
    #    return count_prob[::-1,:,::-1]


    def get_count_prob_right(self, drop_prob):

        l = len(drop_prob)

        # Axis 1 indexes context words, axis 2 indexes relative distances, 
        # axis 3 indexes target words
        pos_prob = np.zeros((l,self.max_dist,self.max_dist))
        count_prob = np.zeros((l,self.max_dist,self.max_dist))
        pos_prob[range(l),0,0] = (1-drop_prob)

        for d in range(1,l):
            pos_prob[d,:,1:] += pos_prob[d-1,:,:-1] * drop_prob[d]
            pos_prob[d, 1:, 1:] += pos_prob[d-1, :-1, :-1] * (1-drop_prob[d])
            count_prob[d, 1:, 1:] += pos_prob[d-1, :-1, :-1] * (1-drop_prob[d])

        return count_prob


    #def get_count_prob_right(self, drop_prob):

    #    l = len(drop_prob)

    #    # Axis 1 indexes context words, axis 2 indexes relative distances, 
    #    # axis 3 indexes target words
    #    pos_prob = np.zeros((l,l,l))
    #    count_prob = np.zeros((l,l,l))
    #    pos_prob[range(l),0,range(l)] = (1-drop_prob)

    #    for d in range(1,l):
    #        pos_prob[d] += pos_prob[d-1] * drop_prob[d]
    #        pos_prob[d, 1:] += pos_prob[d-1, :-1] * (1-drop_prob[d])
    #        count_prob[d, 1:] += pos_prob[d-1, :-1] * (1-drop_prob[d])

    #    return count_prob


    def drop_prob(self, tokens):
        drop_probabilities = np.zeros(len(tokens))
        for i, token in enumerate(tokens):
            freq = self.bigram.unigram.freq(token)
            if freq == 0:
                prob = 0
            else:
                prob = (freq - self.thresh) / freq - (self.thresh / freq)**.5
            drop_probabilities[i] = h.utils.clip(0,1,prob)
        return drop_probabilities



