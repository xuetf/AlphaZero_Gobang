# coding:utf-8

###########################################################################################
# guide:
# temp = visualTool(board_size, line_distance)
# board_size: board_size:m*n
# line_distance: 75 recommended
# eg: temp = visualTool([10,10], 75)

# after initial, then call temp.draw() show the window

# interface 1   getmove()
# return [x,y] represents the last move of human
# note: if flag is True, that means the human has click a valid location, then getmove() is ok

# interface 2   graphic(x, y)
# parameters: x, y location of x,y

# interface 3   wininfo(winner)
# parameters: winner info
###########################################################################################

from tkinter import *
import math
import threading

class VisualTool:

	def __init__(self, board_size= (8, 8), line_distance=75):

		self.flag = False # the click flag for human, used for blocking
		self.location = [0,0] # the last click loction of human
		self.isblack = True # current player is black or not
		self.line_distance = line_distance # line distance
		self.board_size = board_size # board_size
		self.master = Tk() # container
		self.canvas = Canvas(self.master, width=(self.board_size[0]+1)*self.line_distance, height=(self.board_size[1]+1)*self.line_distance+70)
		# init 1 for blacks, 2 for white
		self.chessdata = []
		self.stone_num = 0
		self.player_name = ["Computer", "Human"]
		for i in range(board_size[0]):
			self.chessdata.append([])
			for j in range(board_size[1]):
				self.chessdata[i].append(0)

	def set_player(self, player1, player2, who_first):
		if who_first == 1: player1, player2 = player2, player1 # change
		player1.set_player_no(1)
		player2.set_player_no(2)
		can_click = [False, False]
		can_click[0] = player1.can_click
		can_click[1] = player2.can_click
		player_name = (str(player1), str(player2))
		self.can_click = can_click
		self.player_name = player_name


	# get last move of human
	def getmove(self):
		self.flag = False # important!!! block again
		return (self.location[0], self.location[1])


	# graphic a stone
	def graphic(self, x, y):
		self.stone_num += 1
		y_location = (self.board_size[0] - x) * self.line_distance
		x_location = (y + 1) * self.line_distance
		if self.isblack:
			self.canvas.create_oval(x_location-(0.5*self.line_distance)+0.1*self.line_distance,
									y_location-(0.5*self.line_distance)+0.1*self.line_distance,
									x_location+(0.5*self.line_distance)-0.1*self.line_distance,
								y_location+(0.5*self.line_distance)-0.1*self.line_distance, fill="black")
			self.chessdata[x][y] = 1
		else:
			self.canvas.create_oval(x_location-(0.5*self.line_distance)+0.1*self.line_distance,
									y_location-(0.5*self.line_distance)+0.1*self.line_distance,
									x_location+(0.5*self.line_distance)-0.1*self.line_distance,
									y_location+(0.5*self.line_distance)-0.1*self.line_distance,
									fill="lightgray", outline="lightgray")
			self.chessdata[x][y] = 2
		# change current player
		self.isblack = not self.isblack

	# show winner info
	def wininfo(self, winner):
		'''output the winner info to the window'''
		self.canvas.create_text(0.5 * (self.board_size[0] + 2) * self.line_distance,
										30, font="Times 20 italic bold", fill="red", text=winner)

		self.canvas.delete('chess_board')
		self.canvas.delete('board_line')

	# click event handler
	def onclick(self, event):
		if (self.isblack and self.can_click[0]) or (not self.isblack and self.can_click[1]):
				# decide whether the click is valid
			if event.x > self.line_distance/2 and event.x < (self.board_size[1]+0.5)*self.line_distance and event.y > self.line_distance/2 and event.y < (self.board_size[0]+0.5)*self.line_distance:
				y = math.floor((event.x - self.line_distance / 2) / self.line_distance)
				x = self.board_size[0] - 1 - (math.floor((event.y - self.line_distance / 2) / self.line_distance))
				# decided whether a position that has been used or not
				if self.chessdata[x][y] == 0:
					self.location = [x, y]
					self.flag = True
				else:
					print("re-click the same position")

	# draw canvas
	def draw(self):
		# create canvas
		self.canvas.pack()
		# create chess board
		self.canvas.create_rectangle(0, 0, (self.board_size[1]+1)*self.line_distance, (self.board_size[0]+1)*self.line_distance, fill='white', outline='white', tags=('chess_board'))
		# create players info
		self.canvas.create_text(0.25*(self.board_size[0]+2)*self.line_distance,(self.board_size[0]+1)*self.line_distance+20,fill="darkblue",font="Times 20 italic bold",text="Black: "+self.player_name[0])
		self.canvas.create_text(0.65*(self.board_size[0]+2)*self.line_distance,(self.board_size[0]+1)*self.line_distance+20,fill="darkblue",font="Times 20 italic bold",text="White: "+self.player_name[1])
		# create board
		for x in range(0, (self.board_size[1]+1)*self.line_distance, self.line_distance):
			self.canvas.create_line(x,self.line_distance,x,self.board_size[0]*self.line_distance,fill="#476042")
		for y in range(0, (self.board_size[0]+1)*self.line_distance, self.line_distance):
			self.canvas.create_line(self.line_distance,y,self.board_size[1]*self.line_distance,y,fill="#476042")

		for x in range(0, (self.board_size[1]+1)*self.line_distance, self.line_distance):
			self.canvas.create_line(x,self.line_distance,x,self.board_size[0]*self.line_distance,fill="#476042", tags=('board_line'))
		for y in range(0, (self.board_size[0]+1)*self.line_distance, self.line_distance):
			self.canvas.create_line(self.line_distance,y,self.board_size[1]*self.line_distance,y,fill="#476042", tags=('board_line'))

		# bind the click event
		self.canvas.tag_bind('board_line', '<Button-1>', self.onclick)
		self.canvas.tag_bind('chess_board', '<Button-1>', self.onclick)

		# show the window until being close
		self.master.mainloop()



