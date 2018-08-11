
# coding: utf-8

# In[1]:


import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import itertools
import nltk
import collections
import pickle
import re


# In[6]:


class Text2Vector:
    OUT_OF_VOCAB = 'OUT_OF_VOCAB'
    PADDING = 'PADDING'
    VOCAB_SIZE = 10000
    
    def __init__(self):
        self.counts = None
        self.int_to_vocab = None
        self.vocab_to_int = None

    def __tokenize(self, text):
        """

        :param text:
        :return: list
        """
        return word_tokenize(text)

    def doc_to_vec(self, list_documents):
        logging.debug('-- From doc_to_vec')
        assert isinstance(list_documents, collections.Sequence)
        len_list = len(list_documents)
        tokenized_documents = []
        
        for i, doc in enumerate(list_documents):
            if i % 100 == 0:
                logging.debug('--- Tokenizing: {}\{}, len={}'.format(i, len_list, len(doc)))
            tokenized_documents.append(self.__tokenize(doc))

        return [self.__transform(doc) for doc in tokenized_documents]

    def vec_to_doc(self, list_vecs):
        assert isinstance(list_vecs, collections.Sequence)
        
        return [self.__invert_transform(vec) for vec in list_vecs]

    def fit(self, list_texts):
        logging.debug('-- From fit')
        if self.counts or self.vocab_to_int or self.int_to_vocab:
            logging.info('"fit" is a one-time function')
            return
        list_tokenized_texts = [self.__tokenize(text) for text in list_texts]
        all_tokens = itertools.chain(*list_tokenized_texts)
        self.counts = collections.Counter(all_tokens)

        self.int_to_vocab = self.__get_vocab(vocab_size=Text2Vector.VOCAB_SIZE-2) # 1 for PADDING, 1 for OUT_OF_VOCAB
        self.int_to_vocab = self.int_to_vocab + [Text2Vector.OUT_OF_VOCAB, Text2Vector.PADDING]
        self.vocab_to_int = {word: index for index, word in enumerate(self.int_to_vocab)}

    def __transform(self, list_tokens):
        if not self.vocab_to_int:
            raise Exception('vocab_to_int is None')

        return [self.vocab_to_int[token] if token in self.vocab_to_int else self.vocab_to_int[Text2Vector.OUT_OF_VOCAB] for token in list_tokens]

    def __invert_transform(self, list_ints):
        """

        :param list_ints:
        :return: A document str
        """
        if not self.int_to_vocab:
            raise Exception('vocab_to_int is None')

        return ' '.join([self.int_to_vocab[int_item] for int_item in list_ints])

    def __get_vocab(self, vocab_size=1):
        if not self.counts:
            raise Exception('counts is None')
        return [item[0] for item in self.counts.most_common(n=vocab_size)]

    def get_most_common(self, n=10):
        if not self.counts:
            raise Exception('counts is None')
        return self.counts.most_common(n)

    def export_vocab(self, output_file):
        pd.DataFrame({'word': self.int_to_vocab}).to_csv(output_file, index=False, header=False)
        logging.debug('Exported %s words in vocab into file %s', len(self.int_to_vocab), output_file)
        
    def save(self, filename):
        pickle.dump((self.counts, self.int_to_vocab, self.vocab_to_int), open(filename, 'wb' ))
    
    @staticmethod
    def load(filename):
        counts, int_to_vocab, vocab_to_int = pickle.load(open(filename, 'rb'))
        text2vector = Text2Vector()
        text2vector.counts = counts
        text2vector.int_to_vocab = int_to_vocab
        text2vector.vocab_to_int = vocab_to_int
        return text2vector


# In[8]:




