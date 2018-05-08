# -*- coding: utf-8 -*-
import numpy as np


class Board(object):
    def __init__(self, width=8, height=8, n_in_row=5, who_first=0):
        self.width = width
        self.height = height
        self.n_in_row = n_in_row
        self.players = [1, 2]
        self.init_board(who_first)

    def init_board(self, who_first=0):
        if who_first not in (0, 1):
            raise Exception('start_player should be 0 (player1 first) or 1 (player2 first)')
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not less than %d' % self.n_in_row)
        self.current_player = self.players[who_first]  # start player
        self.availables = list(range(self.width * self.height)) # available moves
        self.states = {} # board states, key:move as location on the board, value:player as pieces type，保存走棋和对应的玩家；用于渲染棋盘展示，构建特征等
        self.last_move = -1

    def move2loc(self, move):
        """
        3*3 board's moves like this:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def loc2move(self, loc):
        '''transfer the loc tuple to id of the moving'''
        if (len(loc) != 2):
            return -1
        move = loc[0] * self.width + loc[1]
        if (move not in range(self.width * self.height)):
            return -1
        return move

    def current_state(self):
        """
        According to the feature engineer
        return the board state from the perspective of the current player
                shape: 4*width*height
        """

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][self.move2loc(move_curr)] = 1.0 # first feature map: self moves
            square_state[1][self.move2loc(move_oppo)] = 1.0 # second feature map: oppo moves
            square_state[2][self.last_move // self.width, self.last_move % self.height] = 1.0  # third feature map, last move indication
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # fourth feature map, current player is player 1?  1 for player 1；0 for player 2;
        return square_state[:, ::-1, :]  # the second dimension for row reversed


    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = self.players[0] if self.current_player == self.players[1] else self.players[1]
        self.last_move = move


    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if(len(moved) < self.n_in_row + 2):
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n))) == 1): # horizon
                return True, player

            if (h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1): # vertical
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1): # right diagonal
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1): # left diagonal
                return True, player

        return False, -1


    def game_end(self):
        """Check whether the game is ended or not, win or tie"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner # has a winner
        elif not len(self.availables):  # tie，no available location。
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player

    def __str__(self):
        return "Board"