import os.path
import numpy as np
import sys
from abc import ABCMeta, abstractmethod
from ta import rsi, macd, bollinger_hband
import pandas

# File for parsing historical data and passing them to environment in batches.
# All data should be located in 'datasets' in home folder

MINUTES_PER_MONTH = 60 * 24 * 30
MINUTES_PER_WEEK = 60 * 24 * 7


class BTCBitstampNMin:

    def __init__(self, candle_min, batch_size):
        print("------ Parsing dataset ------")
        self._src_file = 'btc/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv'
        assert os.path.isfile(self._src_file), 'Download dataset from link above and place it into ./btc'
        self._data = pandas.read_csv(self._src_file)
        self._volume_ts = self._data["Volume_(Currency)"].values
        self._volume_ts[self._volume_ts != self._volume_ts] = 1  # fix NaNs
        self._data = self._data.fillna(method='pad')
        self._price_ts = self._data["Weighted_Price"].values

        self._data_length = self._price_ts.size
        # skip first three years due to low volume and last two months used for evaluation
        self._start = int((MINUTES_PER_MONTH * 12 * 3) / candle_min)
        self._end = int((self._data_length - MINUTES_PER_MONTH*1)/candle_min)
        self._candle_length = candle_min
        self._candles_per_batch = batch_size

        mod = self._data_length % candle_min

        # get timeseries
        self._price_ts = self._price_ts[:-mod]
        self._volume_ts = self._volume_ts[:-mod]

        # min candles to N-min candles
        self._price_ts = self._price_ts[::candle_min]
        self._volume_ts = self._get_sum_ts(self._volume_ts)
        self._rsi_ts = rsi(pandas.Series(self._price_ts)).values
        self._macd_ts = macd(pandas.Series(self._price_ts)).values
        self._bb_ts = bollinger_hband(pandas.Series(self._price_ts)).values

        # get incremental timeseries and normalize
        self._price_incr_ts = self._get_incremental_ts(self._price_ts)
        self._volume_incr_ts = self._get_incremental_ts(self._volume_ts)
        self._rsi_ts = (self._rsi_ts - 50)/50  # [-1, 1]
        self._macd_ts = self._macd_ts/self._price_ts  # +- percentage
        self._bb_ts = self._bb_ts/self._price_ts  # TODO: check whether data here are correct

        # align data
        self._price_ts = self._price_ts[1:]
        self._volume_ts = self._volume_ts[1:]
        self._rsi_ts = self._rsi_ts[1:]
        self._macd_ts = self._macd_ts[1:]
        self._bb_ts = self._bb_ts[1:]

        # delete unused data
        self._data = None
        print("------ Data parsed ------")

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
        ret = np.empty((self._candles_per_batch, 5))
        ret[:, 0] = self._price_incr_ts[batch_start:(batch_start + self._candles_per_batch)]
        ret[:, 1] = self._volume_incr_ts[batch_start:(batch_start + self._candles_per_batch)]
        ret[:, 2] = self._rsi_ts[batch_start:(batch_start + self._candles_per_batch)]
        ret[:, 3] = self._macd_ts[batch_start:(batch_start + self._candles_per_batch)]
        ret[:, 4] = self._bb_ts[batch_start:(batch_start + self._candles_per_batch)]
        return ret, self._price_ts[batch_start:(batch_start + self._candles_per_batch)]

    def get_random_validation(self):
        end = self._price_ts.size - self._candles_per_batch
        batch_start = np.floor(np.random.rand(1) * (end - self._end)).astype(int)[0] + self._end
        ret = np.empty((self._candles_per_batch, 5))
        ret[:, 0] = self._price_incr_ts[batch_start:(batch_start + self._candles_per_batch)]
        ret[:, 1] = self._volume_incr_ts[batch_start:(batch_start + self._candles_per_batch)]
        ret[:, 2] = self._rsi_ts[batch_start:(batch_start + self._candles_per_batch)]
        ret[:, 3] = self._macd_ts[batch_start:(batch_start + self._candles_per_batch)]
        ret[:, 4] = self._bb_ts[batch_start:(batch_start + self._candles_per_batch)]
        return ret, self._price_ts[batch_start:(batch_start + self._candles_per_batch)]

    def get_whole_validation(self):
        size = self._price_ts.size - self._end
        ret = np.empty((size, 5))
        ret[:, 0] = self._price_incr_ts[self._end:]
        ret[:, 1] = self._volume_incr_ts[self._end:]
        ret[:, 2] = self._rsi_ts[self._end:]
        ret[:, 3] = self._macd_ts[self._end:]
        ret[:, 4] = self._bb_ts[self._end:]
        return ret, self._price_ts[self._end:]


if __name__ == "__main__":
    pass
