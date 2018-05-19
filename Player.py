# -*- coding: utf-8 -*-

'''
Base Player Class
extract the Play Abstract Method for different player to override
'''
class Player(object):
    def __init__(self, player_no=0, player_name=""):
        self.player_no = player_no
        self.player_name = player_name
        self.can_click = False

    def set_player_no(self, player_no):
        self.player_no = player_no

    def get_player_no(self):
        return self.player_no

    def get_player_name(self):
        return self.player_name

    # abstract
    def play(self, board, **kwargs):
        raise NotImplementedError


    def __str__(self):
        return "player"