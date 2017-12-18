from fact.io import read_h5py
import pandas as pd


def read_timestamp(path):
    try:
        timestamp = read_h5py(path, key='events', columns=['timestamp'])
        timestamp = pd.to_datetime(timestamp['timestamp'])
    except KeyError:
        try:
            col = 'unix_time_utc'
            unix_time_utc = read_h5py(path, key='events', columns=[col])
            timestamp = pd.to_datetime(
                unix_time_utc[col + '_0'] * 1e6 + unix_time_utc[col + '_1'],
                unit='us',
            )
        except KeyError:
            raise KeyError('File contains neither "timestamp" nor "unix_time_utc"')
