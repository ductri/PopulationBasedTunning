
# coding: utf-8

# In[5]:


import numpy as np
import tensorflow as tf
import logging


# In[6]:



def count_trainable_variables():
    params_count = 0
    for v in tf.trainable_variables():
        v_size = np.prod(v.get_shape().as_list())
        logging.debug('-- -- Variable %s contributes %s parameters', v, v_size)
        params_count += v_size
    return params_count

