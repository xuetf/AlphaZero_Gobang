# -*- coding: utf-8 -*-
from Player import Player
from RolloutMCTS import RolloutMCTS

class RolloutPlayer(Player):
    def __init__(self, nplays=1000, c_puct=5, player_no=0, player_name=""):
        Player.__init__(self, player_no, player_name)
        self.mcts = RolloutMCTS(nplays, c_puct)

    def reset_player(self):
        self.mcts.reuse(-1)

    def play(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.mcts.simulate(board)
            self.mcts.reuse(-1)
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "RolloutPlayer {} {}".format(self.get_player_no(), self.get_player_name())