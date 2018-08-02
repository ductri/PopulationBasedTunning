
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
import os
import pandas as pd
import numpy as np
from random import shuffle
import pickle
from text2vector import Text2Vector
import re


class Dataset:
    def __init__(self, dataset=None, batch_size=1, repeat=1, shuffle_buffer_size=1):
        if dataset is None:
            self.data = [] # generator
        else:
            self.data = dataset.data
        self.__batch_size = batch_size
        self.__repeat = repeat
        self.__shuffle_buffer_size = shuffle_buffer_size
        self.__iterator = None
        
    def map(self, foo):
        new_dataset = Dataset(None, self.__batch_size, self.__repeat)
        new_dataset.data = [foo(data_point) for data_point in self.data]
        return new_dataset
    
    def batch(self, batch_size):
        return Dataset(self, batch_size, self.__repeat)
    
    def repeat(self, count):
        return Dataset(self, self.__batch_size, count)
    
    def shuffle(self, buffer_size):
        return Dataset(self, self.__batch_size, self.__repeat, buffer_size)
    
    def padded_batch(self, batch_size, list_lengths, padded_value):
        if isinstance(list_lengths, int):
            length = list_lengths
            new_data = [list(item[:length]) + [padded_value]*(length - len(item)) for item in self.data]
        elif isinstance(list_lengths, tuple):
            new_data = []
            for datapoint in self.data:
                new_datapoint = []
                for idx, length in enumerate(list_lengths):
                    if length is not None:
                        new_datapoint.append(list(datapoint[idx][:length]) + [padded_value]*(length - len(datapoint[idx])))
                    else:
                        new_datapoint.append(datapoint[idx])
                new_data.append(tuple(new_datapoint))
        else:
            raise NotImplementedError()
            
        new_dataset = Dataset(None, batch_size, self.__repeat)
        new_dataset.data = new_data
        return new_dataset
    
    def get_iterator(self):
        data_length = len(self.data)
        for i in range(self.__repeat):
            for j in range(0, data_length-self.__batch_size+1, self.__batch_size):
                sample = self.data[j : j + self.__shuffle_buffer_size]
                shuffle(sample)
                self.data[j : j + self.__shuffle_buffer_size] = sample
                start = j
                end = j+self.__batch_size
                yield self.data[start: end]

    @staticmethod    
    def from_csv(filename, columns=None):
        df = pd.read_csv(filename)
        if isinstance(columns, list):
            datas = [list(df[col]) for col in columns]
        else:
            datas = [list(df[col]) for col in df.columns]
        new_dataset = Dataset()
        new_dataset.data = list(zip(*datas))
        return new_dataset

    @staticmethod
    def from_tensor_slices(tensors):
        assert isinstance(tensors, tuple)
        new_dataset = Dataset()
        new_dataset.data = list(zip(*tensors))
        return new_dataset
    
    @staticmethod
    def from_pickle_file(filename):
        return pickle.load(open(filename, 'rb'))
    
    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'))


