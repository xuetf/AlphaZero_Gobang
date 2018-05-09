# -*- coding: utf-8 -*-
from MCTS import MCTS
import numpy as np
from Util import softmax


class AlphaZeroMCTS(MCTS):
    def __init__(self, policy_value_fn=None, nplays=1000, cpuct=5, epsilon=0, alpha=0.3, is_selfplay=False):
        MCTS.__init__(self, nplays, cpuct, epsilon, alpha, is_selfplay=is_selfplay)
        self._policy_value_fn = policy_value_fn


    def _evaluate(self, state):
        action_probs, leaf_value = self._policy_value_fn(state)

        # Check for end of game, Adjust the leaf_value
        # 若对局结束，直接根据胜负调整评估。policy evaluation
        is_end, winner = state.game_end()
        if is_end:
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == state.get_current_player() else -1.0

        return is_end, action_probs, leaf_value

    def _play(self, temp=1e-3):
        '''
        calc the move probabilities based on the visit counts at the root node
        temp: 温度参数,见论文
        '''
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)

        pi = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        # pi = np.power(visits, 1/temp)
        # pi = pi / np.sum(pi * 1.0)

        return acts, pi


    def __str__(self):
        return "AlphaZeroMCTS"