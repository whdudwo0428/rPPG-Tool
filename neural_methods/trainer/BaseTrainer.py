# File: neural_methods/trainer/BaseTrainer.py

import os
import pickle

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator


class BaseTrainer:
    @staticmethod
    def add_trainer_args(parser):
        """Adds arguments to Parser for training process"""
        parser.add_argument('--lr', default=None, type=float)
        parser.add_argument('--model_file_name', default=None, type=str)
        return parser

    def __init__(self):
        pass

    def train(self, data_loader):
        pass

    def valid(self, data_loader):
        pass

    def test(self, data_loader):
        pass

    def save_test_outputs(self, predictions, labels, config):
        output_dir = config.TEST.OUTPUT_SAVE_DIR
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        if config.TOOLBOX_MODE == 'train_and_test':
            filename_id = self.model_file_name
        elif config.TOOLBOX_MODE == 'only_test':
            model_file_root = (
                config.INFERENCE.MODEL_PATH.split("/")[-1]
                .split(".pth")[0]
            )
            filename_id = model_file_root + "_" + config.TEST.DATA.DATASET
        else:
            raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')

        output_path = os.path.join(output_dir, filename_id + '_outputs.pickle')
        data = {
            'predictions': predictions,
            'labels': labels,
            'label_type': config.TEST.DATA.PREPROCESS.LABEL_TYPE,
            'fs': config.TEST.DATA.FS
        }
        with open(output_path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Saving outputs to:', output_path)

    def plot_losses_and_lrs(self, train_loss, valid_loss, lrs, config):
        output_dir = os.path.join(
            config.LOG.PATH, config.TRAIN.DATA.EXP_DATA_NAME, 'plots'
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        if config.TOOLBOX_MODE == 'train_and_test':
            filename_id = self.model_file_name
        else:
            raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')

        # -- Loss Plot --
        plt.figure(figsize=(10, 6))
        epochs = range(len(train_loss))
        plt.plot(epochs, train_loss, label='Training Loss')
        if valid_loss:
            plt.plot(epochs, valid_loss, label='Validation Loss')
        else:
            print("Validation loss list is empty; skipping valid-loss plot.")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{filename_id} Losses')
        plt.legend()
        plt.xticks(epochs)
        ax = plt.gca()
        ax.yaxis.set_major_locator(MaxNLocator(integer=False, prune='both'))
        plt.savefig(os.path.join(output_dir, filename_id + '_losses.pdf'), dpi=300)
        plt.close()

        # -- LR Plot --
        plt.figure(figsize=(6, 4))
        scheduler_steps = range(len(lrs))
        plt.plot(scheduler_steps, lrs, label='Learning Rate')
        plt.xlabel('Scheduler Step')
        plt.ylabel('Learning Rate')
        plt.title(f'{filename_id} LR Schedule')
        plt.legend()
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        plt.savefig(os.path.join(output_dir, filename_id + '_learning_rates.pdf'), bbox_inches='tight', dpi=300)
        plt.close()

        print('Saving plots of losses and learning rates to:', output_dir)
