# -*- coding: utf-8 -*-
from Player import Player


class HumanPlayer(Player):
    def __init__(self, player_no=0, player_name=""):
        Player.__init__(self, player_no, player_name)

    def play(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):
                location = [int(n, 10) for n in location.split(",")]  # for python3
            move = board.loc2move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.play(board)
        return move

    def __str__(self):
        return "HumanPlayer {} {}".format(self.get_player_no(), self.get_player_name())