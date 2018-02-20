# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gobang

"""

from __future__ import print_function
import random
import numpy as np
import pickle
from collections import defaultdict, deque
from Game import Game
from Board import Board
from PolicyValueNet import PolicyValueNet  # Pytorch
from AlphaZeroPlayer import AlphaZeroPlayer
from RolloutPlayer import RolloutPlayer

root_data_file = "data/"

class TrainPipeline():
    def __init__(self, init_model=None):
        # params of the board and the game
        self.board_width = 6
        self.board_height = 6
        self.n_in_row = 4
        self.board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params
        self.learn_rate = 5e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1  # how many games of each self-play epoch
        self.epochs = 5  # num of train_steps for each update
        self.is_adjust_lr = True # whether dynamic changing lr
        self.kl_targ = 0.025  # KL散度量，用于early stop
        self.check_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 500
        if init_model:
            print ('init model')
            # start training from an initial policy-value net
            policy_param = pickle.load(open(init_model, 'rb'))
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, net_params=policy_param)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height)
        self.mcts_player = AlphaZeroPlayer(self.policy_value_net.predict, c_puct=self.c_puct,
                                           nplays=self.n_playout, is_selfplay=True)



    def self_play(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data, episode_len = self.game.start_self_play_game(self.mcts_player, temp=self.temp)
            self.episode_len = episode_len
            # augment the data
            play_data = self.augment_data(play_data)
            self.data_buffer.extend(play_data)

    def optimize(self, iteration=0):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]

        if self.is_adjust_lr:
            old_probs, old_v = self.policy_value_net.predict_many(state_batch) # used for adjusting lr

        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.fit(state_batch, mcts_probs_batch, winner_batch,
                                                      self.learn_rate * self.lr_multiplier)
        if self.is_adjust_lr:
            # adaptively adjust the learning rate
            # self.adjust_learning_rate(old_probs, old_v, state_batch, winner_batch)
            self.adjust_learning_rate_2(iteration)

        print("loss:{}, entropy:{}".format(loss, entropy))
        return loss, entropy


    def adjust_learning_rate(self, old_probs, old_v, state_batch, winner_batch):
        '''adjust learning rate based on KL'''
        new_probs, new_v = self.policy_value_net.predict_many(state_batch)
        kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))  # KL散度，相对熵
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:  # kl增大，收敛不好，减小学习率
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:  # kl很小，说明收敛不错；提高学习率
            self.lr_multiplier *= 1.5
        explained_var_old = 1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch))
        explained_var_new = 1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch))

        print("kl:{:.5f},lr:{},explained_var_old:{:.3f},explained_var_new:{:.3f}".format(
                kl, self.learn_rate * self.lr_multiplier, explained_var_old, explained_var_new))


    def adjust_learning_rate_2(self, iteration):
        if (iteration+1) % 200 == 0:
            self.lr_multiplier /= 1.5
        print ("lr:{}".format(self.learn_rate * self.lr_multiplier))


    def evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing games against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = AlphaZeroPlayer(self.policy_value_net.predict, c_puct=self.c_puct,
                                              nplays=self.n_playout)
        pure_mcts_player = RolloutPlayer(c_puct=5, nplays=self.pure_mcts_playout_num)  # 评估的时候应该跟当前最强的战斗
        win_cnt = defaultdict(int)
        for i in range(n_games):
            print ("evaluate game %d" %i)
            winner = self.game.start_game(current_mcts_player, pure_mcts_player, who_first=i % 2, is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(self.pure_mcts_playout_num, win_cnt[1], win_cnt[2],
                                                                  win_cnt[-1]))
        return win_ratio




    def augment_data(self, play_data):
        """
        augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]"""
        extend_data = []
        for state, mcts_porb, winner in play_data:
            '''
            state:
            3*3 board's moves like:
                6 7 8
                3 4 5
                0 1 2
            mcts_porb: flatten
            0,1,2,3,4,5,6,7,8
            winner
            1 or -1
            '''
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])  # i=4就是原来的数据
                equi_mcts_prob = np.rot90(np.flipud(mcts_porb.reshape(self.board_height, self.board_width)),
                                          i)  # 上下翻转成棋盘形状，各个cell的值对应该位置下棋概率
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])  # 水平翻转
                equi_mcts_prob = np.fliplr(equi_mcts_prob)  # equi_mcts_prob和equi_state对应
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
        return extend_data

    def save_model(self, win_ratio, epochs):
        # save
        net_params = self.policy_value_net.get_policy_param()  # get model params
        pickle.dump(net_params, open(root_data_file+"current_policy_{}_epochs_{}.model".format(self.policy_value_net, epochs), 'wb'),
                    pickle.HIGHEST_PROTOCOL)  # save model param to file

        if win_ratio > self.best_win_ratio:
            print("New best policy!!!!!!!!")
            self.best_win_ratio = win_ratio
            pickle.dump(net_params, open(root_data_file+"best_policy_{}_epochs_{}.model".format(self.policy_value_net, epochs), 'wb'),
                        pickle.HIGHEST_PROTOCOL)  # update the best_policy
            if self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000:
                self.pure_mcts_playout_num += 1000  # 增强
                self.best_win_ratio = 0.0

    def run(self):
        """run the training pipeline"""
        loss_records = []
        entropy_records = []
        try:
            for i in range(self.game_batch_num):
                self.self_play(self.play_batch_size) # big step 1
                print("batch i:{}, episode_len:{}".format(i + 1, self.episode_len))

                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.optimize(iteration=i) # big step 2
                    loss_records.append(loss)
                    entropy_records.append(entropy)

                # check the performance of the current model，and save the model params
                if (i + 1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i + 1))
                    win_ratio = self.evaluate() #big step 3
                    self.save_model(win_ratio, i+1)

                # 每500轮保存下损失和熵
                if  (i + 1) % 500 == 0:
                    records = {"loss": loss_records, "entropy":entropy_records}
                    pickle.dump(records, open(root_data_file+"loss_entropy_records.data", 'wb'),
                                pickle.HIGHEST_PROTOCOL)
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()
