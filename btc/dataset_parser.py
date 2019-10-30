import os.path
import numpy as np
import sys
from abc import ABCMeta, abstractmethod
from ta import rsi
import pandas

# File for parsing historical data and passing them to environment in batches.
# All data should be located in 'datasets' in home folder

MINUTES_PER_MONTH = 60 * 24 * 30
MINUTES_PER_WEEK = 60 * 24 * 7


class BTCBitstampNMin:

    def __init__(self, candle_min, batch_size):
        self._src_file = 'btc/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv'
        assert os.path.isfile(self._src_file), 'Download dataset from link above and place it into ~/datasets'
        self._data = np.genfromtxt(self._src_file, delimiter=',')
        self._data_length = np.shape(self._data)[0]
        # skip first four years due to low volume and last two months used for evaluation
        self._start = int((MINUTES_PER_MONTH * 12 * 3) / candle_min)
        self._end = int((self._data_length - MINUTES_PER_MONTH*2)/candle_min)
        self._candle_length = candle_min
        self._candles_per_batch = batch_size
        self._repair_volume()

        mod = self._data.shape[0] % candle_min

        # get timeseries
        self._price_ts = self._data[:-mod, 7]
        self._volume_ts = self._data[:-mod, 6]

        # min candles to N-min candles
        self._price_ts = self._price_ts[::candle_min]
        self._volume_ts = self._get_sum_ts(self._volume_ts)
        self._rsi_ts = rsi(pandas.Series(self._price_ts)).values

        # get incremental timeseries and normalize
        self._price_incr_ts = self._get_incremental_ts(self._price_ts)
        self._volume_incr_ts = self._get_incremental_ts(self._volume_ts)
        self._rsi_ts = (self._rsi_ts - 50)/50   # [-1, 1]

        # align data
        self._price_ts = self._price_ts[1:]
        self._volume_ts = self._volume_ts[1:]
        self._rsi_ts = self._rsi_ts[1:]

        # delete unused data
        self._data = None

    def _repair_volume(self):
        """
        Bitstamp dataset has corrupted volume. When no trade occur, there are two same records in row.
        """
        err = 1
        for i in range(1, self._data_length):
            if self._data[i - err, 6] == self._data[i, 6] or self._data[i, 6] == self._data[i, 7]:
                self._data[i, 6] = 1
                err += 1
            else:
                err = 1

    def _get_incremental_ts(self, ts):
        ts_from = ts[:-1]
        ts_to = ts[1:]
        return (ts_to/ts_from - 1)*100

    def _get_sum_ts(self, ts):
        new_ts = np.reshape(ts, (int(ts.size/self._candle_length), self._candle_length))
        new_ts = np.sum(new_ts, axis=1)
        return new_ts

    def get_batch(self):
        batch_start = np.floor(np.random.rand(1) * (self._end - self._start)).astype(int)[0] + self._start
        ret = np.empty((self._candles_per_batch, 3))
        ret[:, 0] = self._price_incr_ts[batch_start:(batch_start + self._candles_per_batch)]
        ret[:, 1] = self._volume_incr_ts[batch_start:(batch_start + self._candles_per_batch)]
        ret[:, 2] = self._rsi_ts[batch_start:(batch_start + self._candles_per_batch)]
        return ret, self._price_ts[batch_start:(batch_start + self._candles_per_batch)]

    def get_random_validation(self):
        end = self._price_ts.size - self._candles_per_batch
        batch_start = np.floor(np.random.rand(1) * (end - self._end)).astype(int)[0] + self._end
        ret = np.empty((self._candles_per_batch, 3))
        ret[:, 0] = self._price_incr_ts[batch_start:(batch_start + self._candles_per_batch)]
        ret[:, 1] = self._volume_incr_ts[batch_start:(batch_start + self._candles_per_batch)]
        ret[:, 2] = self._rsi_ts[batch_start:(batch_start + self._candles_per_batch)]
        return ret, self._price_ts[batch_start:(batch_start + self._candles_per_batch)]

    def get_whole_validation(self):
        size = self._price_ts.size - self._end
        ret = np.empty((size, 3))
        ret[:, 0] = self._price_incr_ts[self._end:]
        ret[:, 1] = self._volume_incr_ts[self._end:]
        ret[:, 2] = self._rsi_ts[self._end:]
        return ret, self._price_ts[self._end:]


if __name__ == "__main__":
    pass
