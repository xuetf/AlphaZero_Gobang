import numpy as np
import pickle
from Config import *


def load_config(file_name, only_load_param=True):
    '''
    :param
      file_name: the loading file name
      only_load_param:if only load the parameters of network ，be True； Then Need Manual setup some parameters
    :return:
    '''
    # 如果只保存了参数
    if only_load_param:
        config = Config()
        # manual setup parameters
        n = 4
        width, height = 6, 6
        network = ResNet

        policy_param = pickle.load(open(file_name, 'rb'))

        config.policy_param = policy_param
        config.n_in_row = n
        config.board_width = width
        config.board_height = height
        config.network = network
    # else load the whole config object
    else:
        config = pickle.load(open(file_name, 'rb'))

    return config


def softmax(x):
    '''防止数值溢出, 减去一个值，不改变最终的大小'''
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs