# -*- coding: utf-8 -*-
from MCTS import MCTS
import numpy as np
from operator import itemgetter
import copy

'''
Random Rollout MCTS,
Every step of simulation, the action is randomly chosen
'''
class RolloutMCTS(MCTS):
    def __init__(self, nplays=1000, c_puct=5.0, epsilon=0, alpha=0.3, limit=1000):
        MCTS.__init__(self, nplays, c_puct, epsilon, alpha)
        self._limit = limit

    def _evaluate(self, state):
        """Use the rollout policy to play until the end of the game, returning +1 if the current
                player wins, -1 if the opponent wins, and 0 if it is a tie.
                """
        action_probs = self._policy(state)
        is_end, _ = state.game_end() # from the perspective of beginning of the rollout

        # begin rollout
        for i in range(self._limit):
            rollout_end, rollout_winner = state.game_end()
            if rollout_end: break
            rollout_action = self._rollout(state)
            state.do_move(rollout_action)
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")

        # 设置leaf_value
        if rollout_winner == -1:  # 平局 或者 达到limit步还没决出胜负
            leaf_value = 0
        else:
            leaf_value = 1.0 if rollout_winner == state.get_current_player() else -1.0

        return is_end, action_probs, leaf_value

    def _play(self, temp=1e-3):
        return max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]



    def _rollout(self, board):
        """rollout_policy_fn -- a coarse, fast version of policy_fn used in the rollout phase."""
        # rollout randomly
        action_probs = np.random.rand(len(board.availables))
        tmp_action_probs = zip(board.availables, action_probs)
        return max(tmp_action_probs, key=itemgetter(1))[0]


    def _policy(self, board):
        """a function that takes in a state and outputs a list of (action, probability)
        tuples"""
        # return uniform probabilities and 0 score for pure MCTS
        action_probs = np.ones(len(board.availables))/len(board.availables)
        return zip(board.availables, action_probs)



    def __str__(self):
        return "RolloutMCTS"

