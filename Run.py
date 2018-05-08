# -*- coding: utf-8 -*-
import numpy as np
from Board import Board
from HumanPlayer import HumanPlayer
from AlphaZeroPlayer import AlphaZeroPlayer
from RolloutPlayer import RolloutPlayer
from Game import Game
from PolicyValueNet import *
import pickle

root_data_file = "data/"

'''
Play Game between Human and AlphaZero
'''
def run():
    n = 4
    width, height = 6, 6
    model_file = root_data_file + 'resnet_6_6_4.model'
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        ################ human VS AI ###################
        try:
            policy_param = pickle.load(open(model_file, 'rb'))
        except:
            policy_param = pickle.load(open(model_file, 'rb'), encoding='bytes')  # To support python3
        best_policy = PolicyValueNet(width, height, Network=ResNet, net_params=policy_param) # setup which Network to use based on the net_params
        mcts_player = AlphaZeroPlayer(best_policy.predict, c_puct=5,
                                 nplays=1000)  # set larger n_playout for better performance

        # uncomment the following line to play with pure MCTS
        # mcts_player2 = RolloutPlayer(nplays=1000, c_puct=5)

        # human player, input your move in the format: 2,3
        human = HumanPlayer()

        # set start_player=0 for human first
        game.start_game(human, mcts_player, who_first=1, is_shown=1)

    except KeyboardInterrupt:
        print('\n\rquit')



if __name__ == '__main__':
    run()


