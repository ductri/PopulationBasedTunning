
# coding: utf-8

# In[6]:


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


# In[83]:


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
    def __init__(self, data, batch_size, repeat, shuffle_buffer_size):
        
        # data: list of tuple, because of the requirement of shuffling
        self.data = data
        self.__batch_size = batch_size
        self.__repeat = repeat
        self.__shuffle_buffer_size = shuffle_buffer_size
        self.__iterator = None
        
    def map(self, foo):
        new_data = [foo(data_point) for data_point in self.data]
        new_dataset = Dataset(new_data, self.__batch_size, self.__repeat, self.__shuffle_buffer_size)
        return new_dataset
    
    def batch(self, batch_size):
        return Dataset(self.data, batch_size, self.__repeat, self.__shuffle_buffer_size)
    
    def repeat(self, count):
        return Dataset(self.data, self.__batch_size, count, self.__shuffle_buffer_size)
    
    def shuffle(self, buffer_size):
        return Dataset(self.data, self.__batch_size, self.__repeat, buffer_size)
    
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
            
        new_dataset = Dataset(new_data, batch_size, self.__repeat, self.__shuffle_buffer_size)
        return new_dataset
    
    def get_iterator(self):
        data_length = len(self.data)
        self.__do_estimate_number_steps()
        for i in range(self.__repeat):
            for j in range(0, data_length-self.__batch_size+1, self.__batch_size):
                sample = self.data[j : j + self.__shuffle_buffer_size]
                shuffle(sample)
                self.data[j : j + self.__shuffle_buffer_size] = sample
                start = j
                end = j+self.__batch_size
                yield tuple(zip(*self.data[start: end]))
    
    def get_data_length(self):
        return len(self.data)
    
    @staticmethod    
    def from_csv(filename, columns=None):
        df = pd.read_csv(filename)
        if isinstance(columns, list):
            datas = [list(df[col]) for col in columns]
        else:
            datas = [list(df[col]) for col in df.columns]
        datas = tuple(datas)
        Dataset.__assert_valid_data(datas)
        
        new_dataset = Dataset(list(zip(*datas)), 1, 1, 1)
        return new_dataset

    @staticmethod
    def from_tensor_slices(tensors):
        Dataset.__assert_valid_data(tensors)
        new_tensors = []
        for i in range(len(tensors)):
            new_tensors.append(list(tensors[i]))
        new_dataset = Dataset(list(zip(*new_tensors)), 1, 1, 1)
        return new_dataset
    
    @staticmethod
    def load(filename):
        data, batch_size, repeat, shuffle_buffer_size = pickle.load(open(filename, 'rb'))
        return Dataset(data, batch_size, repeat, shuffle_buffer_size)
    
    def save(self, filename):
        pickle.dump((self.data, self.__batch_size, self.__repeat, self.__shuffle_buffer_size), open(filename, 'wb'))
    
    @staticmethod
    def __assert_valid_data(data):
        if not isinstance(data, tuple):
            raise DatasetException('data must be a tuple')
        for i in range(len(data) - 1):
            if len(data[i]) != len(data[i+1]):
                raise DatasetException('All element must have the same length. Length of element {}-th is {},                                        length of element {}-th is {}'.format(i, len(data[i]), i+1, len(data[i+1])))
    def __do_estimate_number_steps(self):
        num_steps_per_epoch = int(self.get_data_length()/self.__batch_size)
        logging.info('There will be {} step/epoch'.format(num_steps_per_epoch))
        logging.info('There will be total {} steps'.format(num_steps_per_epoch*self.__repeat))
        
    class DatasetException(Exception):
        def __init__(self, message):
            Exception.__init__(self, message)

