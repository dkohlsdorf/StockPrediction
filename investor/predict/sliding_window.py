import numpy as np
import investor.predict

from investor.predict.numerics import roi_window, roi


class SlidingWindow:

    def __init__(self, win, target, lag):
        '''
        Sliding window prediction.
        For an n-dimensional time series we extract windows
        of a certain size at each time and take the target dimension
        at time + lag as the prediction.

        :param win: window size
        :param target: target dimension
        :param lag: n steps into the future
        '''

        self.win = win
        self.target = target
        self.lag = lag

    def slide(self, sequence):
        '''
        Slide extracts sliding windows from a sequence,
        and the return for each window.

        :param sequence: input sequences
        :return: the windows, the target predictions and the investing value
        '''
        n = sequence.shape[0]
        invest_at   = []
        windows     = []
        predictions = []
        for t in range(self.win + 1, n - self.lag):
            sample       = sequence[t-self.win:t]
            roi_sample   = roi_window(sample, sequence[t-self.win-1])
            roi_lag      = roi(sample[-1][self.target], sequence[t + self.lag][self.target])
            windows     += [roi_sample.flatten()]
            predictions += [roi_lag]
            invest_at   += [sample[-1][self.target]]
        invest_at = np.array(invest_at)
        windows = np.array(windows)
        predictions = np.array(predictions)
        return windows, predictions, invest_at
