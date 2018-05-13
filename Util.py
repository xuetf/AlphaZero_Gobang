import numpy as np
import pickle
from Config import *
from AlphaZeroPlayer import AlphaZeroPlayer


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


def load_player_from_file(file_name, add_noise=True):
    config = load_config(file_name, only_load_param=False)
    best_policy = PolicyValueNet(config.board_width, config.board_height,
                                 Network=config.network,
                                 net_params=config.policy_param)  # setup which Network to use based on the net_params

    best_player = AlphaZeroPlayer(best_policy.predict, c_puct=config.c_puct,
                                  nplays=1200, add_noise=add_noise)  #increase nplays=1200, add_noise=True, add_noise_to_best_player, avoid the same play every game
    return best_player
