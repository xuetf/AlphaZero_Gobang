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
from PolicyValueNet import *  # Pytorch
from AlphaZeroPlayer import AlphaZeroPlayer
from RolloutPlayer import RolloutPlayer
from Config import *
root_data_file = "data/"

class TrainPipeline():
    def __init__(self, config=None):
        # params of the board and the game
        self.config = config if config else Config()

        # Network wrapper
        self.policy_value_net = PolicyValueNet(self.config.board_width, self.config.board_height,
                                               net_params=self.config.policy_param,
                                               Network=self.config.network)

        # 传入policy_value_net的predict方法，神经网络辅助MCTS搜索过程
        self.mcts_player = AlphaZeroPlayer(self.policy_value_net.predict, c_puct=self.config.c_puct,
                                           nplays=self.config.n_playout, is_selfplay=True)


    def self_play(self, n_games=1):
        """
        collect self-play data for training
        n_game: 自我对弈n_game局后，再更新网络
        """
        self.episode_len = 0
        self.augmented_len = 0
        for i in range(n_games):
            winner, play_data, episode_len = self.config.game.start_self_play_game(self.mcts_player, temp=self.config.temp)
            self.episode_len += episode_len # episode_len每局下的回合数
            # augment the data
            play_data = self.augment_data(play_data)
            self.augmented_len += len(play_data)
            self.config.data_buffer.extend(play_data)

    def optimize(self, iteration=0):
        """update the policy-value net"""
        mini_batch = random.sample(self.config.data_buffer, self.config.batch_size)
        state_batch, mcts_probs_batch, winner_batch = list(zip(*mini_batch))

        if self.config.is_adjust_lr:
            old_probs, old_v = self.policy_value_net.predict_many(state_batch) # used for adjusting lr

        for i in range(self.config.per_game_opt_times): # number of opt times
            loss_info = self.policy_value_net.fit(state_batch, mcts_probs_batch, winner_batch,
                                                      self.config.learn_rate * self.config.lr_multiplier)
        if self.config.is_adjust_lr:
            # adaptively adjust the learning rate
            self.adjust_learning_rate(old_probs, old_v, state_batch, winner_batch)
            #self.adjust_learning_rate_2(iteration)

        print("combined loss:{0:.5f}, value loss:{1:.5f}, policy loss:{2:.5f}, entropy:{3:.5f}".
              format(loss_info['combined_loss'], loss_info['value_loss'], loss_info['policy_loss'], loss_info['entropy']))

        return loss_info


    def adjust_learning_rate(self, old_probs, old_v, state_batch, winner_batch):
        '''
        reference paper: PPO:Proximal Policy Optimization
        adjust learning rate based on KL
        '''
        new_probs, new_v = self.policy_value_net.predict_many(state_batch)
        kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))  # KL散度，相对熵
        if kl > self.config.kl_targ * 2 and self.config.lr_multiplier > 0.1:  # kl增大，收敛不好，减小学习率
            self.config.lr_multiplier /= 1.5
        elif kl < self.config.kl_targ / 2 and self.config.lr_multiplier < 10:  # kl很小，说明收敛不错；提高学习率
            self.config.lr_multiplier *= 1.5
        explained_var_old = 1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch))
        explained_var_new = 1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch))

        print("kl:{:.5f},lr:{:.7f},explained_var_old:{:.3f},explained_var_new:{:.3f}".format(
                kl, self.config.learn_rate * self.config.lr_multiplier, explained_var_old, explained_var_new))


    def adjust_learning_rate_2(self, iteration):
        '''衰减法'''
        if (iteration+1) % self.config.lr_decay_per_iterations == 0:
            self.config.lr_multiplier /= self.config.lr_decay_speed
        print ("lr:{}".format(self.config.learn_rate * self.config.lr_multiplier))


    def evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing games against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = AlphaZeroPlayer(self.policy_value_net.predict, c_puct=self.config.c_puct,
                                              nplays=self.config.n_playout)
        pure_mcts_player = RolloutPlayer(c_puct=5, nplays=self.config.pure_mcts_playout_num)  # 可以优化，评估的时候应该跟当前最强的战斗
        win_cnt = defaultdict(int)
        for i in range(n_games):
            print ("evaluate game %d" %i)
            winner = self.config.game.start_game(current_mcts_player, pure_mcts_player, who_first=i % 2, is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(self.config.pure_mcts_playout_num, win_cnt[1], win_cnt[2],
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
                equi_mcts_prob = np.rot90(np.flipud(mcts_porb.reshape(self.config.board_height, self.config.board_width)),
                                          i)  # 上下翻转成棋盘形状，各个cell的值对应该位置下棋概率
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])  # 水平翻转
                equi_mcts_prob = np.fliplr(equi_mcts_prob)  # equi_mcts_prob和equi_state对应
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
        return extend_data

    def save_model(self, win_ratio, epochs, prefix=''):
        # save
        self.config.policy_param = self.policy_value_net.get_policy_param()  # get model params

        if win_ratio > self.config.best_win_ratio:
            print("New best policy!!!!!!!!")
            self.config.best_win_ratio = win_ratio
            pickle.dump(self.config, open("tmp/config-epochs-{0}-{1:.2f}.pkl".format(epochs,win_ratio), 'wb'))

            if self.config.best_win_ratio == 1.0 and self.config.pure_mcts_playout_num < 5000:
                self.config.pure_mcts_playout_num += 1000  # 增强
                self.config.best_win_ratio = 0.0

    def run(self):
        """run the training pipeline"""
        print ("start training from game:{}".format(self.config.start_game_num))
        try:
            for i in range(self.config.start_game_num, self.config.game_batch_num):

                self.self_play(self.config.play_batch_size) # big step 1
                print("iteration i:{}, episode_len:{}, augmented_len:{}, current_buffer_len:{}".format(i + 1,
                                                            self.episode_len, self.augmented_len, len(self.config.data_buffer)))

                if len(self.config.data_buffer) > self.config.batch_size:
                    loss_info = self.optimize(iteration=i) # big step 2
                    self.config.loss_records.append(loss_info)

                self.config.start_game_num = i + 1  # update for restart

                # check the performance of the current model，and save the model params
                if (i + 1) % self.config.check_freq == 0:
                    print("current iteration: {}".format(i + 1))
                    win_ratio = self.evaluate() #big step 3
                    self.save_model(win_ratio, i+1)



        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    config = pickle.load(open('tmp/config-epochs-{0}-{1:.2f}.pkl'.format(50, 0.90),'rb'))
    training_pipeline = TrainPipeline(config=None)
    training_pipeline.run()