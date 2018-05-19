# -*- coding: utf-8 -*-
from Player import Player


class HumanPlayer(Player):
    def __init__(self, player_no=0, player_name=""):
        Player.__init__(self, player_no, player_name)
        # self.tool = tool
        self.can_click = True # can click the board


    # def play(self, board):
    #     '''play based on human input'''
    #     try:
    #         location = input("Your move: ")
    #         if isinstance(location, str):
    #             location = [int(n, 10) for n in location.split(",")]  # for python3
    #         move = board.loc2move(location)
    #     except Exception as e:
    #         move = -1
    #     if move == -1 or move not in board.availables:
    #         print("invalid move")
    #         move = self.play(board)
    #     return move


    def play(self, board, **kwargs):
        tool = kwargs['tool']
        while(not tool.flag): # block
            pass
        location = tool.getmove() # [x,y]
        move = board.loc2move(location)
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.play(board)
        return move


    def __str__(self):
        return "HumanPlayer{}".format(self.get_player_name())

