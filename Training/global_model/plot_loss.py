'''
keras callback to plot loss
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import clear_output

from tensorflow import keras

## callbacks
# updatable plot
# a minimal example (sort of)

class LossPlotter(keras.callbacks.Callback):
    def __init__(self, output_dir, batch_mode=False):
        self.batch_mode = batch_mode
        self.output_dir = output_dir
        super(keras.callbacks.Callback, self).__init__()

    def on_train_begin(self, logs={}):
        self.i = 0
        self.loss_labels = ['loss','loss_clusters', 'loss_windows','loss_softF1',
                            'loss_en_softF1','loss_en_regr', 'loss_regularization', 'loss_is_calo_seed']
        self.losses = {}
        self.val_losses ={}
        for l in self.loss_labels:
            self.losses[l] = []
            self.val_losses[l] = []
        self.figure = None
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.performance_save(logs)
        # if not self.batch_mode:
        self.performance_plot()
        self.save_metrics()

    def performance_save(self, logs):
        self.logs.append(logs)
        
        for l in self.loss_labels:
            self.losses[l].append(logs.get(l))
            self.val_losses[l].append(logs.get("val_" + l))
        self.i += 1

       
    def performance_plot(self):
        self.figure, axs = plt.subplots(2, 4, figsize=(24,12), dpi=100)
        # self.figure.tight_layout()
        for il, l in enumerate(self.loss_labels):
            ax = axs[il // 4][il %4]
            ax.set_title(l)
            ax.plot(list(range(1,self.i+1)), self.losses[l], ".-", label="Train loss")
            ax.plot(list(range(1,self.i+1)), self.val_losses[l], ".-", label="Val loss")
            ax.set_yscale("log")
            ax.set_xlabel("epochs")
            ax.legend()
        
        if not self.batch_mode:
            clear_output(wait=True)
            plt.show()
        self.figure.savefig(self.output_dir+ "/loss_plot.png")
        plt.close(self.figure)


    def save_metrics(self):
        metrics = {}
        for l,v in self.losses.items():
            metrics[l] = v
        for l,v in self.val_losses.items():
            metrics[f"val_{l}"] = v
        df = pd.DataFrame(metrics)
        df.to_csv(self.output_dir + "/metrics_history.csv")
        

    def save_figure(self, fname):
        if self.batch_mode:
            self.performance_plot()
        self.figure.savefig(self.output_dir + '/' + fname)
        plt.close(self.figure)
