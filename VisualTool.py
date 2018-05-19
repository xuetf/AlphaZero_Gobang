# coding:utf-8

###########################################################################################
# 初始化传入参数
# temp = visualTool(board_size, line_distance)
# board_size: 棋盘大小 m*n
# line_distance: 绘图网格间距(建议75)
# player_name: 玩家名称
# eg: temp = visualTool([10,10], 75, ["computer","human"])

# 初始化后马上调用 temp.draw() 函数后开始运行

# 接口1   getmove()
# 传入参数: 无
# 返回参数: [x,y] 表示上次用户的下棋位置
# 注: flag 变量值为真时, getmove() 获取的值为有效值

# 接口2   graphic(x, y)
# 传入参数: x, y 表示落子位置
# 返回参数: 无

# 接口3   wininfo(winner)
# 传入参数: winner 限定为0/1 表示玩家0/玩家1胜利
# 返回参数: 无
###########################################################################################

from tkinter import *
import math
import threading

class VisualTool:

	def __init__(self, board_size= (8, 8), line_distance=75):

		self.flag = False # 是否点击的标志
		self.location = [0,0] #用户上次下棋的位置
		self.isblack = True # 当前下棋人是否为黑棋
		self.line_distance = line_distance # 绘制网格间距
		self.board_size = board_size # 棋盘大小 n*n
		self.master = Tk() # 画图容器
		self.canvas = Canvas(self.master, width=(self.board_size[1]+1)*self.line_distance, height=(self.board_size[0]+1)*self.line_distance+70)
		#棋盘容器 初始化为全零 1黑 2白
		self.chessdata = []
		self.stone_num = 0
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


	# 获取用户上次动作
	def getmove(self):
		self.flag = False
		return (self.location[0], self.location[1])


	# 绘制一个棋子
	def graphic(self, x, y):
		# 用户上步棋子位置失效
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
		# 变换当前下棋人颜色
		self.isblack = not self.isblack

	# 显示胜利信息
	def wininfo(self, winner):
		'''直接将winner字段输出到界面即可'''
		self.canvas.create_text(0.5 * (self.board_size[0] + 2) * self.line_distance,
										30, font="Times 20 italic bold", fill="red", text=winner)

		self.canvas.delete('chess_board')

	# 点击事件
	def onclick(self, event):
		if (self.isblack and self.can_click[0]) or (not self.isblack and self.can_click[1]):
				# 判断点击范围是否合理
			if event.x > self.line_distance/2 and event.x < (self.board_size[1]+0.5)*self.line_distance and event.y > self.line_distance/2 and event.y < (self.board_size[0]+0.5)*self.line_distance:
				y = math.floor((event.x - self.line_distance / 2) / self.line_distance)
				x = self.board_size[0] - 1 - (math.floor((event.y - self.line_distance / 2) / self.line_distance))
				# 判断是否重复点击
				if self.chessdata[x][y] == 0:
					# 绘制一个棋子
					# self.graphic(x, y)
					self.location = [x, y]
					self.flag = True
				else:
					print("重复点击")

	# 绘制函数
	def draw(self):
		# 产生画布
		self.canvas.pack()
		# 绘制点击读取板
		self.canvas.create_rectangle(0, 0, (self.board_size[1]+1)*self.line_distance, (self.board_size[0]+1)*self.line_distance, fill='white', outline='white', tags=('chess_board'))
		# 绘制玩家信息
		self.canvas.create_text(0.25*(self.board_size[0]+2)*self.line_distance,(self.board_size[0]+1)*self.line_distance+20,fill="darkblue",font="Times 20 italic bold",text="Black: "+self.player_name[0])
		self.canvas.create_text(0.65*(self.board_size[0]+2)*self.line_distance,(self.board_size[0]+1)*self.line_distance+20,fill="darkblue",font="Times 20 italic bold",text="White: "+self.player_name[1])
		# 绘制棋盘
		for x in range(0, (self.board_size[1]+1)*self.line_distance, self.line_distance):
			self.canvas.create_line(x,self.line_distance,x,self.board_size[0]*self.line_distance,fill="#476042")
		for y in range(0, (self.board_size[0]+1)*self.line_distance, self.line_distance):
			self.canvas.create_line(self.line_distance,y,self.board_size[1]*self.line_distance,y,fill="#476042")
		# 绑定点击事件
		self.canvas.tag_bind('chess_board', '<Button-1>', self.onclick)
		# 保留绘制图像
		self.master.mainloop()



