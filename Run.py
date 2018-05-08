# -*- coding: utf-8 -*-
import numpy as np
from Board import Board
from HumanPlayer import HumanPlayer
from AlphaZeroPlayer import AlphaZeroPlayer
from RolloutPlayer import RolloutPlayer
from Game import Game
from PolicyValueNet import *
import pickle
from Config import *

root_data_file = "data/"
tmp_data_file = "tmp/"
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


'''
Play Game between Human and AlphaZero
'''
def run():

    config = load_config(file_name=root_data_file+'resnet_6_6_4.model', only_load_param=True)

    try:
        board = Board(width=config.board_width, height=config.board_height, n_in_row=config.n_in_row)
        game = Game(board)

        #--------------- human VS AI ----------------
        best_policy = PolicyValueNet(config.board_width, config.board_height,
                                     Network=config.network, net_params=config.policy_param) # setup which Network to use based on the net_params

        mcts_player = AlphaZeroPlayer(best_policy.predict, c_puct=config.c_puct,
                                 nplays=1000)  #set larger nplays for better performance

        # uncomment the following line to play with pure MCTS
        # mcts_player2 = RolloutPlayer(nplays=1000, c_puct=config.c_puct)

        # human player, input your move in the format: 2,3
        human = HumanPlayer()

        # set start_player=0 for human first
        game.start_game(human, mcts_player, who_first=1, is_shown=1)

    except KeyboardInterrupt:
        print('\n\rquit')



if __name__ == '__main__':
    run()


