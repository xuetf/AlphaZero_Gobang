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

def tour(n_games=20):
    name = '../drive/workspace/work_deep_learning/tmp_5_in_rows_resnet2/epochs-{}-opponent-Pure-win-1.00.pkl'
    win_ratio = collections.defaultdict(float)
    player2 = load_player_from_file(name.format(1500), add_noise=True, nplays=100)  # 最终模型
    for i in range(50, 1501, 50):
        win_cnt = collections.defaultdict(int)
        print ('AlphaGoZero{}'.format(i) + ' VS ' 'AlphaGoZero1500')
        try:
            player1 = load_player_from_file(name.format(i), add_noise=True, nplays=100)
        except:
            print ('not exist ' + name.format(i))
            continue
        for num in range(n_games):
            board = Board(width=8, height=8, n_in_row=5)
            game = Game(board)
            winner = game.start_game(player1, player2, who_first=num % 2, is_shown=0)
            win_cnt[winner] += 1
            print ('Game {}, Winner No is {}'.format(num+1, winner))
        print("win: {}, lose: {}, tie:{}".format(win_cnt[1], win_cnt[2], win_cnt[-1]))
        win_ratio[i] = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
    pickle.dump(dict(win_ratio), open('win_ratio.pkl','wb'))
    return win_ratio

def tour_1100_vs_1500(n_games=100):
    name = '../drive/workspace/work_deep_learning/tmp_5_in_rows_resnet2/epochs-{}-opponent-Pure-win-1.00.pkl'
    player1 = load_player_from_file(name.format(1100), add_noise=True, nplays=500)  # 最终模型
    player2 = load_player_from_file(name.format(1500), add_noise=True, nplays=500)  # 最终模型
    win_cnt = collections.defaultdict(int)
    for num in range(n_games):
        board = Board(width=8, height=8, n_in_row=5)
        game = Game(board)
        winner = game.start_game(player1, player2, who_first=num % 2, is_shown= (1 if num<5 else 0))
        win_cnt[winner] += 1
        print('Game {}, Winner No is {}'.format(num + 1, winner))
    print("win: {}, lose: {}, tie:{}".format(win_cnt[1], win_cnt[2], win_cnt[-1]))
    pickle.dump(dict(win_cnt), open('../drive/workspace/work_deep_learning/tmp_5_in_rows_resnet2/win_ratio.pkl', 'wb'))



if __name__ == '__main__':
    #config = load_config(file_name=tmp_data_file + 'epochs-1100-resnet2.pkl', only_load_param=False)
    #run(config)
    tour_1100_vs_1500()






