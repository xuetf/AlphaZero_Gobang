# -*- coding: utf-8 -*-
import numpy as np


class TreeNode(object):
    def __init__(self, parent, prior_p):
        self._parent = parent # 父节点
        self._children = {}  # 子节点，a map from action to TreeNode
        self._n_visits = 0 # 访问次数
        self._Q = 0 # Q值
        self._u = 0 # bonus，根据先验概率和访问次数计算
        self._P = prior_p # 先验概率，由价值网络得到

    def expand(self, action_priors):
        """Expand tree by creating new children.
                action_priors -- output from policy function - a list of tuples of actions
                    and their prior probability according to the policy function.
                """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct=5.0, epsilon=0.0, alpha=0.3):
        """Select action among children that gives maximum action value, Q plus bonus u(P).
        (1-e)pa+e*dirichlet(eta) # add Dirichlet Noise for exploration
                Returns:
                A tuple of (action, next_node)
                """
        return max(self._children.items(), key=lambda act_node: act_node[1]._get_value(c_puct, epsilon, alpha))


    def _get_value(self, c_puct, epsilon=0, alpha=0.3):
        """Calculate and return the value for this node: a combination of leaf evaluations, Q, and
        this node's prior adjusted for its visit count, u
        c_puct -- a number in (0, inf) controlling the relative impact of values, Q, and
            prior probability, P, on this node's score.
        epsilon -- the fraction of the prior probability, and 1-epsilon is the corresponding dirichlet noise fraction
        alpha -- the parameter of dirichlet noise
        """
        noise = np.random.dirichlet([alpha])[0] # 添加噪声，目前噪声比例epsilon=0,即，不使用噪声
        self._u = c_puct * ((1-epsilon) * self._P + epsilon * noise) * \
                  np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        return self._Q + self._u

    def backup(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
                """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.backup(-leaf_value)

        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        # 这一步合并了W，Q, (参见AlphaGoZero论文方法论部分)推导如下：
        # W = W_old + leaf_value; Q_old = W_old / (n-1) => W_old = (n-1)*Q_old; Q = W/n
        # Q = W/n=(W_old + leaf_value)/n = ((n-1)*Q_old+leaf_value)/n
        #   = (n*Q_old-Q_old+leaf_value)/n = Q_old + (leaf_value-Q_old)/n
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None