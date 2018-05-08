# -*- coding: utf-8 -*-
import numpy as np
from Player import Player
from AlphaZeroMCTS import AlphaZeroMCTS



class AlphaZeroPlayer(Player):
    def __init__(self, policy_value_fn, nplays=1000, c_puct=5, player_no=0, is_selfplay=False, player_name=""):
        Player.__init__(self, player_no, player_name)
        self.mcts = AlphaZeroMCTS(policy_value_fn, nplays, c_puct, is_selfplay=is_selfplay, epsilon=0)
        self._is_selfplay = is_selfplay


    def reset_player(self):
        '''reset, reconstructing the MCTS Tree for next simulation'''
        self.mcts.reuse(-1)

    def play(self, board, temp=1e-3, explore_step=30, return_prob=False):
        sensible_moves = board.availables
        move_probs = np.zeros(board.width * board.height)  # the pi vector returned by MCTS as in the alphaGo Zero paper
        if len(sensible_moves) > 0:
            temp = 1.0 if (self._is_selfplay and len(board.states) < explore_step) else 1e-3
            acts, probs = self.mcts.simulate(board, temp)
            move_probs[list(acts)] = probs

            if self._is_selfplay:
                # add Dirichlet Noise for exploration (only needed for self-play training)
                move = np.random.choice(acts, p=(1-0.2) * probs + 0.2 * np.random.dirichlet(0.3 * np.ones(len(probs))))
                #move = np.random.choice(acts, p=probs)
                self.mcts.reuse(move)  # update the root node and reuse the search tree
            else:
                # with the default temp=1e-3, this is almost equivalent to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.reuse(-1)
                # location = board.move2loc(move)
                # print("AI move: %d,%d\n" % (location[0], location[1]))

            if return_prob:
                return move, move_probs #for train
            else:
                return move #for run
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "AlphaZeroPlayer{} {}".format(self.get_player_no(), self.get_player_name())