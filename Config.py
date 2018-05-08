from Board import *
from  Game import *
from PolicyValueNet import *
import pickle
from collections import deque
class Config:
    def __init__(self):
        self.board_width = 6
        self.board_height = 6
        self.n_in_row = 4
        self.board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.lr_decay_per_iterations = 100  # learning rate decay after how many iterations
        self.lr_decay_speed = 5  # learning rate decay speed
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 3  # how many games of each self-play epoch
        self.per_game_opt_times = 5  # num of train_steps for each update
        self.is_adjust_lr = True  # whether dynamic changing lr
        self.kl_targ = 0.02  # KL，用于early stop
        self.check_freq = 5  # frequency of checking the performance of current model and saving model
        self.start_game_num = 0  # the starting num of training
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        self.network = ResNet  # the type of network
        self.policy_param = None  # Network parameters
        self.loss_records = [] # 记录损失
