import matplotlib.pyplot as plt
from Config import *
from Util import load_config



def draw_loss():
    config = load_config(file_name=tmp_data_file + 'config-epochs-270-1.00.pkl', only_load_param=False)
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

draw_loss()