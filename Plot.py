import matplotlib.pyplot as plt
from Config import *
from Util import load_config

'''experiment draw according to practice'''

def draw_pk():
    plt.figure(figsize=(10, 10))
    plt.suptitle('PK')
    plt.xlabel('opponent')
    plt.ylabel('times')

    ax = plt.subplot(2, 2, 1)
    ax.set_title('AlphaGobangZero VS RandomRolloutMCTS')
    names = ['Win', 'Lose', 'Tie']
    x = range(len(names))
    y = [50, 0., 0.]
    plt.bar(range(len(names)), y)
    plt.xticks(x, names, rotation=45)
    plt.yticks(range(0, 51, 10))

    ax = plt.subplot(2, 2, 2)
    ax.set_title('AlphaGobangZero VS Human')
    names = ['Win', 'Lose', 'Tie']
    x = range(len(names))
    y = [19, 18, 13]
    plt.bar(range(len(names)), y)
    plt.xticks(x, names, rotation=45)
    plt.yticks(range(0, 51, 10))
    plt.show()


def draw_loss(filename=root_data_file + 'epochs-1500_resnet2.pkl'):
    config = load_config(file_name=filename, only_load_param=False)
    print (config.loss_records)

    combined_loss_list = [loss['combined_loss']for loss in config.loss_records]
    policy_loss_list = [loss['policy_loss'] for loss in config.loss_records]
    value_loss_list = [loss['value_loss'] for loss in config.loss_records]
    entropy_list = [loss['entropy'] for loss in config.loss_records]

    plt.plot(combined_loss_list, color='blue', label='combined_loss')
    plt.plot(policy_loss_list, color='red', label='policy_loss')
    plt.plot(value_loss_list, color='green', label='value_loss')
    plt.plot(entropy_list, color='black', label='entropy')
    plt.legend()
    plt.show()

def draw_epsilon_parameters():
    '''data according to experiment'''
    plt.figure(figsize=(10, 10))
    plt.suptitle(r'$\epsilon$ experiment')

    ax = plt.subplot(2, 3, 1)
    ax.set_title(r'$\epsilon$=0 AlphaGobangZero')
    names = [r'$\epsilon$=0.2', r'$\epsilon$=0.8', 'Random', 'Human']
    x = range(len(names))
    y = [0.167, 0.333, 0.70, 0.30]
    plt.bar(range(len(names)), y)
    plt.xticks(x, names, rotation=45)
    plt.yticks(np.arange(0, 1.01, 0.1))
    plt.xlabel('opponent')
    plt.ylabel('win ratio')

    ax = plt.subplot(2, 3, 2)
    ax.set_title(r'$\epsilon$=0.2 AlphaGobangZero')
    names = [r'$\epsilon$=0', r'$\epsilon$=0.8', 'Random', 'Human']
    x = range(len(names))
    y = [0.833, 0.80, 1.00, 0.50]
    plt.bar(range(len(names)), y)
    plt.xticks(x, names, rotation=45)
    plt.yticks(np.arange(0, 1.01, 0.1))
    plt.xlabel('opponent')


    ax = plt.subplot(2, 3, 3)
    ax.set_title(r'$\epsilon$=0.8 AlphaGobangZero')
    names = [r'$\epsilon$=0', r'$\epsilon$=0.2', 'Random', 'Human']
    x = range(len(names))
    y = [0.667, 0.20, 0.90, 0.333]
    plt.bar(range(len(names)), y)
    plt.xticks(x, names, rotation=45)
    plt.yticks(np.arange(0, 1.01, 0.1))
    plt.xlabel('opponent')

    plt.show()


def draw_n_parameters():
    '''data according to experiment'''
    plt.figure(figsize=(10, 10))
    plt.suptitle('n experiment')

    ax = plt.subplot(2, 3, 1)
    ax.set_title('n=10 AlphaGobangZero')
    names = ['n=400', 'n=1000', 'Random', 'Human']
    x = range(len(names))
    y = [0., 0., 0.6, 0.]
    plt.bar(range(len(names)), y)
    plt.xticks(x, names, rotation=45)
    plt.yticks(np.arange(0, 1.01, 0.1))
    plt.xlabel('opponent')
    plt.ylabel('win ratio')

    ax = plt.subplot(2, 3, 2)
    ax.set_title('n=400 AlphaGobangZero')
    names = ['n=10', 'n=1000', 'Random', 'Human']
    x = range(len(names))
    y = [1.0, 0.4, 1.00, 0.50]
    plt.bar(range(len(names)), y)
    plt.xticks(x, names, rotation=45)
    plt.yticks(np.arange(0, 1.01, 0.1))
    plt.xlabel('opponent')


    ax = plt.subplot(2, 3, 3)
    ax.set_title('n=1000 AlphaGobangZero')
    names = ['n=10', 'n=400', 'Random', 'Human']
    x = range(len(names))
    y = [1.0, 0.6, 1.00, 0.60]
    plt.bar(range(len(names)), y)
    plt.xticks(x, names, rotation=45)
    plt.yticks(np.arange(0, 1.01, 0.1))
    plt.xlabel('opponent')

    plt.show()

def draw_cpuct_parameters():
    '''data according to experiment'''
    plt.figure(figsize=(10, 10))
    plt.suptitle(r'$c_{puct}$ experiment')

    ax = plt.subplot(2, 3, 1)
    ax.set_title(r'$c_{puct}$=1 AlphaGobangZero')
    names = [r'$c_{puct}$=5', r'$c_{puct}$=20', 'Random', 'Human']
    x = range(len(names))
    y = [0.367, 0.40, 1.00, 0.20]
    plt.bar(range(len(names)), y)
    plt.xticks(x, names, rotation=45)
    plt.yticks(np.arange(0, 1.01, 0.1))
    plt.xlabel('opponent')
    plt.ylabel('win ratio')

    ax = plt.subplot(2, 3, 2)
    ax.set_title(r'$c_{puct}$=5 AlphaGobangZero')
    names = [r'$c_{puct}$=1', r'$c_{puct}$=20', 'Random', 'Human']
    x = range(len(names))
    y = [0.633, 0.80, 1.00, 0.50]
    plt.bar(range(len(names)), y)
    plt.xticks(x, names, rotation=45)
    plt.yticks(np.arange(0, 1.01, 0.1))
    plt.xlabel('opponent')


    ax = plt.subplot(2, 3, 3)
    ax.set_title(r'$c_{puct}$=20 AlphaGobangZero')
    names = [r'$c_{puct}$=1', r'$c_{puct}$=5', 'Random', 'Human']
    x = range(len(names))
    y = [0.60, 0.20, 1.00, 0.333]
    plt.bar(range(len(names)), y)
    plt.xticks(x, names, rotation=45)
    plt.yticks(np.arange(0, 1.01, 0.1))
    plt.xlabel('opponent')
    plt.show()

def draw_network():
    '''data according to actual experiment'''
    plt.figure(figsize=(10, 10))
    plt.suptitle('Network Contrast Experiment')
    plt.xlabel('opponent')
    plt.ylabel('win ratio')

    ax = plt.subplot(2, 3, 1)
    ax.set_title('AlphaZero')
    names = ['ConvNet', 'ResNet_1', 'FFNet', 'Random', 'Human']
    x = range(len(names))
    y = [0.367, 0.567, 0.967, 1, 0.533]
    plt.bar(range(len(names)), y)
    plt.xticks(x, names, rotation=45)
    plt.yticks(np.arange(0, 1.01, 0.1))

    ax = plt.subplot(2, 3, 2)
    ax.set_title('ConvNet')
    names = ['AlphaZero', 'ResNet_1', 'FFNet', 'Random', 'Human']
    x = range(len(names))
    y = [0.633, 0.80, 0.933, 1, 0.6]
    plt.bar(range(len(names)), y)
    plt.xticks(x, names, rotation=45)
    plt.yticks(np.arange(0, 1.01, 0.1))

    ax = plt.subplot(2, 3, 3)
    ax.set_title('FFNet')
    names = ['AlphaZero', 'ResNet_1', 'ConvNet', 'Random', 'Human']
    x = range(len(names))
    y = [0, 0.067, 0.033, 0.933, 0.1]
    plt.bar(range(len(names)), y)
    plt.xticks(x, names, rotation=45)
    plt.yticks(np.arange(0, 1.01, 0.1))
    plt.show()

draw_loss()