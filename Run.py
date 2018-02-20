# -*- coding: utf-8 -*-
import numpy as np
from Board import Board
from HumanPlayer import HumanPlayer
from AlphaZeroPlayer import AlphaZeroPlayer
from RolloutPlayer import RolloutPlayer
from Game import Game
from PolicyValueNet import PolicyValueNet
from PolicyValueNetNumpy import PolicyValueNetNumpy
import pickle

root_data_file = "data/"

def run():
    n = 4
    width, height = 6, 6
    model_file = root_data_file + 'current_policy_simple_net_epochs_1200_6_6_4.model'
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        ################ human VS AI ###################
        try:
            policy_param = pickle.load(open(model_file, 'rb'))
        except:
            policy_param = pickle.load(open(model_file, 'rb'), encoding='bytes')  # To support python3
        best_policy = PolicyValueNet(width, height, policy_param)
        mcts_player = AlphaZeroPlayer(best_policy.predict, c_puct=5,
                                 nplays=1000)  # set larger n_playout for better performance

        # uncomment the following line to play with pure MCTS (its much weaker even with a larger n_playout)
        # mcts_player2 = RolloutPlayer(nplays=1000, c_puct=5)

        # human player, input your move in the format: 2,3
        human = HumanPlayer()
        # set start_player=0 for human first
        game.start_game(human, mcts_player, who_first=1, is_shown=1)

    except KeyboardInterrupt:
        print('\n\rquit')


def run_between_alphazero():
    n = 4
    width, height = 6, 6
    cur_model_file = root_data_file + 'current_policy_simple_net_6_6_4.model'
    best_model_file = root_data_file + 'current_policy_simple_net_epochs_1200_6_6_4.model'

    cur_policy_param = pickle.load(open(cur_model_file, 'rb'))
    best_policy_param = pickle.load(open(best_model_file, 'rb'))
    best_policy = PolicyValueNet(width, height, best_policy_param)
    cur_policy = PolicyValueNet(width, height, cur_policy_param)
    epochs = 0
    best_win_count = 0
    cur_win_count = 0
    tie = 0
    try:
        while(epochs < 50):
            board = Board(width=width, height=height, n_in_row=n)
            game = Game(board)

            ################ AI VS AI ###################
            best_mcts_player = AlphaZeroPlayer(best_policy.predict, c_puct=5,
                                     nplays=1000, player_name="Current 1200")  # set larger n_playout for better performance
            cur_mcts_player = AlphaZeroPlayer(cur_policy.predict, c_puct=5,
                                     nplays=1000, player_name="Current")

            # set start_player=0 for human first
            winner = game.start_game(best_mcts_player, cur_mcts_player, who_first=epochs%2, is_shown=1)
            print ("winner no:{}".format(winner))
            if winner == 1: best_win_count += 1
            if winner == 2: cur_win_count += 1
            if winner == -1: tie += 1
            epochs += 1
            if epochs % 10 == 0:
                print ("Best win {}, Current win {}, Tie {}".format(best_win_count, cur_win_count, tie))

    except KeyboardInterrupt:
        print('\n\rquit')

if __name__ == '__main__':
    run()


