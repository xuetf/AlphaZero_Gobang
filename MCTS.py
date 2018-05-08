# -*- coding: utf-8 -*-
from TreeNode import TreeNode
import copy

class MCTS(object):
    def __init__(self, nplays=1000, c_puct=5.0, epsilon=0, alpha=0.3, is_selfplay=False):
        self._root = TreeNode(None, 1.0)
        self._nplays = nplays # number of plays of one simulation
        self._c_puct = c_puct # a number controlling the relative impact of values, Q, and P
        self._epsilon = epsilon # the ratio of dirichlet noise
        self._alpha = alpha # dirichlet noise parameter
        self._is_selfplay = is_selfplay # whether used to selfplay

    def _search(self, state):
        """Run a single search from the root to the leaf, getting a value at the leaf and
        propagating it back through its parents. State is modified in-place, so a copy must be
        provided.
        Arguments:
        state -- a copy of the state.
        """
        node = self._root
        while(True):
            if node.is_leaf():
                break
            # Greedily select next move.
            if self._is_selfplay: # add noise when training
                epsilon = self._epsilon if node == self._root else 0 # only root add dirichlet
            else:
                epsilon = 0
            action, node = node.select(self._c_puct, epsilon, self._alpha) # MCTS of SELECT step
            state.do_move(action)

        # Evaluate the leaf using a network which outputs a list of (action, probability)
        # tuples p and also a score v in [-1, 1] for the current player.
        is_end, pi, value = self._evaluate(state) # MCTS Of the EVALUATE step

        if not is_end: # if not end then expand
            node.expand(pi) # MCTS of the [EXPAND] step

        # Update value and visit count of nodes in this traversal.
        node.backup(-value) # MCTS of the [BACKUP] step



    def _evaluate(self, state):
        '''
        Template Method, Override for different child class
        MCTS of the [EVALUATE] Step
        Return the move probabilities of each available action and the evaluation value of winning
        '''
        raise NotImplementedError



    def _play(self, temp=1e-3):
        '''
        Template Method, Override for different child class
        MCTS of the [PLAY] Step
        Return the final action
        '''
        raise NotImplementedError


    def reuse(self, last_move):
        """
        Step forward in the tree, keeping everything we already know about the subtree.
        if self-play then update the root node and reuse the search tree, speeding next simulation
        else reset the root
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)


    def simulate(self, state, temp=1e-3):
        """Runs all simulations sequentially and returns the available actions and their corresponding probabilities
                Arguments:
                state -- the current state, including both game state and the current player.
                temp -- temperature parameter in (0, 1] that controls the level of exploration
                Returns:
                the available actions and the corresponding probabilities
                """
        for n in range(self._nplays):
            state_copy = copy.deepcopy(state) # key!!!
            self._search(state_copy) # the reference will be changed

        return self._play(temp) # override for different child class


    def __str__(self):
        return "MCTS"
