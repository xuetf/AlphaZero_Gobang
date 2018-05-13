# -*- coding: utf-8 -*-
import numpy as np
from Player import Player
from AlphaZeroMCTS import AlphaZeroMCTS



class AlphaZeroPlayer(Player):
    def __init__(self, policy_value_fn, nplays=1000, c_puct=5, player_no=0, is_selfplay=False, add_noise=None, player_name=""):
        Player.__init__(self, player_no, player_name)
        self.mcts = AlphaZeroMCTS(policy_value_fn, nplays, c_puct, is_selfplay=is_selfplay, epsilon=0)

        # True then reuse mcts, Else reset mcts at every move
        self._is_selfplay = is_selfplay

        # if add_noise = None, then selfplay=True add noise; selfplay=False, don't add noise
        # if add_noise = True, then whatever selfplay=True or False, both add noise
        self._add_noise = is_selfplay if add_noise is None else add_noise

    def reset_player(self):
        '''reset, reconstructing the MCTS Tree for next simulation'''
        self.mcts.reuse(-1)

    def play(self, board, temp=1e-3, explore_step=30, epsilon=0.2, alpha=0.3, return_prob=False):
        sensible_moves = board.availables
        move_probs = np.zeros(board.width * board.height)  # the pi vector returned by MCTS as in the alphaGo Zero paper
        if len(sensible_moves) > 0:
            # with the default temp=1e-3, this is almost equivalent to choosing the move with the highest prob
            temp = 1.0 if (self._is_selfplay and len(board.states) < explore_step) else 1e-3
            acts, probs = self.mcts.simulate(board, temp)
            move_probs[list(acts)] = probs

            if self._add_noise:
                # different from paper, in the paper, noise is added to the root of MCTS Tree
                # Here, noise is just added to the result
                move = np.random.choice(acts,
                                        p=(1 - epsilon) * probs + epsilon * np.random.dirichlet(alpha * np.ones(len(probs))))
            else:
                move = np.random.choice(acts, p=probs)

            if self._is_selfplay:
                # add Dirichlet Noise for exploration (only needed for self-play training)
                self.mcts.reuse(move)  # update the root node and reuse the search tree
            else:
                # reset the root node
                self.mcts.reuse(-1)

            if return_prob:
                return move, move_probs #for train
            else:
                return move #for run
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "AlphaZeroPlayer{} {}".format(self.get_player_no(), self.get_player_name())