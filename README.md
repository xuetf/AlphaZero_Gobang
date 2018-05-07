## Overview
This is a AlphaZero Implementation of Gobang based on Pytorch.

## Code
- Train.py : Run the train process
- Run.py : Play with Human using the trained model
- Player.py: Base class for different Player
- RolloutPlayer.py: Player with MCTS using random rollout policy
- AlphaZeroPlayer.py: AlphaZero Player with MCTS guided by Residual Network
- HumanPlayer.py: Human Player
- MCTS.py: Base class for different MCTS
- AlphaZeroMCTS.py: MCTS guided by Residual Network
- RolloutMCTS.py: MCTS using random rollout policy
- TreeNode.py: MCTS Tree Node
- PolicyValueNet.py: Redisual Network Implementation based on Pytorch
- Board.py: Board Class for Gobang
- Game.py: Game for Gobang

## Running Script
### Run on Linux Server
nohup python -u Train.py > simpleres_train.log 2>&1 &

### Download the trained model
scp root@139.199.21.83:/usr/local/workspace/AlphaZero_Gobang/data/current_policy_resnet_epochs_1500.model /Users/xuetf/Downloads

### Upload -P big!!!
scp -P 8381 local_file_path root@139.199.21.83:/root/


## Reference
[AlphaZero实战](https://zhuanlan.zhihu.com/p/32089487)