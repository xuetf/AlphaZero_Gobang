# -*- coding: utf-8 -*-
import numpy as np
from Board import Board
from HumanPlayer import HumanPlayer
from AlphaZeroPlayer import AlphaZeroPlayer
from RolloutPlayer import RolloutPlayer
from Game import Game
from PolicyValueNet import *
from Config import *
from Util import load_config


'''
Play Game between Human and AlphaZero
'''
def run(config=None):
    if config == None:  config = load_config(file_name=root_data_file+'resnet_6_6_4.model', only_load_param=True)
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

        # set who_first=0 for human first
        game.start_game(human, mcts_player, who_first=1, is_shown=1)

    except KeyboardInterrupt:
        print('\n\rquit')



if __name__ == '__main__':
    config = load_config(file_name=tmp_data_file + 'epochs-1080-6_6_4_best_resnet.pkl', only_load_param=False)
    #run(config)
    setattr(config,"episode_records", [])
    print (config.episode_records)


