import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


class Dashboard:
    def __init__(self, dataset):
        super(Dashboard).__init__()
        self.dataset = dataset
        self.sequence = dataset.seq_len

        self.fig, self.ax = self.init_figure()
        self.data = None
        self.pred = None
        self.upper = None
        self.lower = None

        self.preds = self.initialize()

        self.band_flag = False
        self.upper_band = list()
        self.lower_band = list()
        
        self.area_up1 = None
        self.area_upper = None
        self.area_lower = None
        self.labels = self.initialize()
        self._time = None
        self.time = self.initialize(dataset.time)
        self.label = dataset.label
        self.scope = 240
        self.idx = 0

    def init_figure(self):
        fig, ax = plt.subplots(figsize=(20, 6), facecolor="lightgray")
        fig.suptitle(self.dataset.title, fontsize=25)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")

        return fig, ax

    def concat(self, target, x, denormalize=False):
        if denormalize is True:
            x = self.dataset.denormalize(x)

        x = x.detach().numpy()
        if target is None:
            target = x
        else:
            target = np.concatenate((target, x))

        return target
    
    def initialize(self, value=None):
        if value is None:
            data_list = [list() for x in range(self.sequence)]
            for i in range(len(data_list)):
                data_list[i] = np.zeros(i)
            return data_list
        
        return value

    def data_concat(self, target, x, cond):
        if target is None:
            return x[:cond]
        
        return np.concatenate((target[:-cond], x[:cond + 1]))

    def pred_concat(self, target, y, cond):
        if target is None:
            return y
        
        length = len(target) - self.sequence + cond
        return np.concatenate((target[:length + 2], y[cond + 1:]))
    
    def visualize(self, x, y, cond, normalize=True):
        if normalize:
            x = self.dataset.denormalize(x)
            y = self.dataset.denormalize(y)

        x = x[0].detach().numpy().ravel()
        y = y[0].detach().numpy().ravel()
        
        self.data = self.data_concat(self.data, x, cond)
        self.pred = self.pred_concat(self.pred, y, cond)

        if self.band_flag is False:
            self.area_up1 = y[:cond]
            self.area_up2 = y[:cond]
            self.area_up3 = y[:cond]
            self.area_down1 = y[:cond]
            self.area_down2 = y[:cond]
            self.area_down3 = y[:cond]
            self.band_flag = True
            
            self.upper = y[:cond]
            self.lower = y[:cond]

        std = y.std()
        median = np.median(y[cond:])

        self.upper = np.append(self.upper, y[cond:].max())
        self.lower = np.append(self.lower, y[cond:].min())

        self.area_up1 = np.append(self.area_up1, median + 1 * std)
        self.area_up2 = np.append(self.area_up2, median + 2 * std)
        self.area_up3 = np.append(self.area_up3, median + 3 * std)
        self.area_down1 = np.append(self.area_down1, median - 1 * std)
        self.area_down2 = np.append(self.area_down2, median - 2 * std)
        self.area_down3 = np.append(self.area_down3, median - 3 * std)

        fig, ax = self.fig, self.ax
        try:
            ax.clear()
            ax.grid()
            
            length = len(self.pred) - cond
            plt.axvspan(length - self.sequence, length - self.sequence + cond, facecolor='green', alpha=0.01)
            plt.axvspan(length - self.sequence + cond, length, facecolor='gray', alpha=0.5)

            length = np.arange(len(self.area_up1[cond:]))
            ax.fill_between(length, self.area_down3[cond:], self.area_up3[cond:], color='red', alpha=0.1)
            ax.fill_between(length, self.area_down2[cond:], self.area_up2[cond:], color='blue', alpha=0.2)
            ax.fill_between(length, self.area_down1[cond:], self.area_up1[cond:], color='blue', alpha=0.3)
            ax.plot(self.data[cond:], "r-", linewidth=2, alpha=0.6, label="data")
            ax.plot(self.pred[cond:], "b-", linewidth=3, alpha=0.2, label="pred")


            xtick = np.arange(0, self.scope * 2, 12)
            values = self.time[0:self.scope * 2:12]

            plt.ylim(self.dataset.min, self.dataset.max)
            plt.xticks(xtick, values, rotation=30)

            fig.show()
            fig.canvas.draw()
            fig.canvas.flush_events()
        except (KeyboardInterrupt, AttributeError) as e:
            raise
        

    def _visualize(self, time, data, pred):
        self.idx += 1
        try:
            fig, ax = self.fig, self.ax

            data = data[0].detach().numpy().ravel()
            pred = pred[0].detach().numpy().ravel()
            
            def concat(target, x, loc=None):
                if target is None:
                    if loc is None:
                        target = x[:6]
                    else:
                        target = x[:loc]
                else:
                    if loc is None:
                        target = np.concatenate((target, x[6:7]))
                    else:
                        target = np.concatenate((target, x[loc:loc+1]))
                return target
            
            self.data = concat(self.data, data)
            min_scope = max(self.sequence, self.data.size - self.scope + 1)
            if self.area_lower is None:
                self.area_lower = pred[:6]
                self.area_upper = pred[:6]

            min_axvln = min(0, self.area_lower.size)

            self.area_upper = np.append(self.area_upper, pred[6:].max())
            self.area_lower = np.append(self.area_lower, pred[6:].min())

            ax.clear()
            ax.grid()
            
            for i in range(self.sequence):
                self.labels[i] = concat(self.labels[i], self.label[self.idx, i], loc=0)
                self.preds[i] = np.append(self.preds[i], pred[i])
            
            plt.axvspan(10, 40, facecolor='gray')

            # print(self.dataset.label[min_scope:min_scope+self.scope, :, 0])
            for i in range(7, self.sequence):
                # ax.plot(self.labels[i][min_scope:], alpha=0.4, linewidth=i//2, label=f"Label {i}")
                # print(self.labels[i][min_scope:])
                ax.plot(self.preds[i][min_scope:], alpha=0.4, linewidth=1, label=f"Preds {i}")
                
                
            # print(f"Pred length : {len(self.pred5)}")
            # ax.vlines(x=min_axvln, ymin=pred[4:].min(), ymax=pred[4:].max(), linewidth=3)
            ax.plot(self.data[min_scope:], "r-", alpha=0.6, linewidth=4, label="Actual Data")
            # ax.plot(self.pred2[min_scope:], alpha=0.4, label="Pred 3")
            # ax.plot(self.pred3[min_scope:], alpha=0.4, label="Pred 6")
            # ax.plot(self.pred4[min_scope:], alpha=0.4, label="Pred 9")
            # ax.plot(self.pred5[min_scope:], alpha=0.4, label="Pred 11")
            ax.plot(self.area_upper[min_scope:], "b-", linewidth=2, alpha=0.6, label="Upper")
            ax.plot(self.area_lower[min_scope:], "b-", linewidth=2, alpha=0.6, label="Lower")
            ax.legend()
            
            # ax.fill_between(np.arange(10), self.area_lower[min_scope:], self.area_upper[min_scope:], color='lightgray', alpha=0.5)
            
            xtick = np.arange(0, self.scope, 12)
            values = self.time[min_scope:min_scope + self.scope:12]
            plt.xticks(xtick, values, rotation=30)

            plt.ylim(self.dataset.min, self.dataset.max)
            ax.relim()
            fig.legend()

            fig.show()
            fig.canvas.draw()
            fig.canvas.flush_events()
        except (KeyboardInterrupt, AttributeError) as e:
            fig.close()
            raise 
        
def plt_loss(gen_loss, dis_loss, path, num):
    idx = num * 1000
    gen_loss = np.clip(np.array(gen_loss), a_min=-1500, a_max=1500)
    dis_loss = np.clip(-np.array(dis_loss), a_min=-1500, a_max=1500)
    plt.figure(figsize=(9, 4.5))
    plt.plot(gen_loss[idx : idx + 1001], label="g_loss", alpha=0.7)
    plt.plot(dis_loss[idx : idx + 1001], label="d_loss", alpha=0.7)
    plt.title("Loss")
    plt.legend()
    plt.savefig(
        path + "/Loss_" + str(num) + ".png", bbox_inches="tight", pad_inches=0.5
    )
    plt.close()

def plt_progress(real, fake, epoch, path):
    real = np.squeeze(real)
    fake = np.squeeze(fake)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.flatten()
    fig.suptitle("Data generation, iter:" + str(epoch))
    for i in range(ax.shape[0]):
        ax[i].plot(real[i], color="red", label="Real", alpha=0.7)
        ax[i].plot(fake[i], color="blue", label="Fake", alpha=0.7)
        ax[i].legend()

    plt.savefig(
        path + "/line_generation" + "/Iteration_" + str(epoch) + ".png",
        bbox_inches="tight",
        pad_inches=0.5,
    )
    plt.clf()
