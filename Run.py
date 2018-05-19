# -*- coding: utf-8 -*-
import numpy as np
from Board import Board
from HumanPlayer import HumanPlayer
from AlphaZeroPlayer import AlphaZeroPlayer
from RolloutPlayer import RolloutPlayer
from Game import Game
from PolicyValueNet import *
from Config import *
from Util import load_config, load_player_from_file
import collections

'''
Play Game between Human and AlphaZero
'''
def run(config=None):
    if config == None:  config = load_config(file_name=root_data_file+'resnet_6_6_4.model', only_load_param=True)
    try:
        board = Board(width=config.board_width, height=config.board_height, n_in_row=config.n_in_row)

        #--------------------1.set player:alphazero VS human---------------------#
        best_policy = PolicyValueNet(config.board_width, config.board_height,
                                     Network=config.network, net_params=config.policy_param) # setup which Network to use based on the net_params

        player1 = AlphaZeroPlayer(best_policy.predict, c_puct=config.c_puct,
                                 nplays=1000)  #set larger nplays for better performance

        # uncomment the following line to play with pure MCTS
        #player2 = RolloutPlayer(nplays=1000, c_puct=config.c_puct)
        player2 = HumanPlayer()
        # --------------------2.set order---------------------#
        who_first = 0 # 0 means player1 first, otherwise player2 first

        # --------------------3.start game--------------------#
        game = Game(board,is_visualize=True)
        t = threading.Thread(target=game.start_game, args=(player1, player2, who_first))
        t.start()
        game.show()

    except:
        print('\n\rquit')



if __name__ == '__main__':
    config = load_config(file_name=root_data_file + 'epochs-1100-resnet2.pkl', only_load_param=False)
    run(config)







