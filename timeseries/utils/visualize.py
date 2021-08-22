import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Dashboard:
    def __init__(self, dataset):
        super(Dashboard).__init__()
        self.dataset = dataset
        self.sequence = dataset.seq_len

        self.fig, self.ax = self.init_figure()
        self.data = None
        self.preds = self.initialize()
        self.pred0 = None
        self.pred1 = None
        self.pred2 = None
        self.pred3 = None
        self.pred4 = None
        self.pred5 = None
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


    def _visualize(self, time, data, pred):
        self.idx += 1
        try:
            fig, ax = self.fig, self.ax

            data = data[0].detach().numpy().ravel()
            pred = pred[0].detach().numpy().ravel()
            
            def concat(target, x, loc=None):
                if target is None:
                    if loc is None:
                        target = x[:3]
                    else:
                        target = x[:loc]
                else:
                    if loc is None:
                        target = np.concatenate((target, x[3:4]))
                    else:
                        target = np.concatenate((target, x[loc:loc+1]))
                return target
            
            self.data = concat(self.data, data)
            min_scope = max(0, self.data.size - self.scope + 1)
            if self.area_lower is None:
                self.area_lower = pred[:4]
                self.area_upper = pred[:4]
            min_axvln = min(0, self.area_lower.size)

            self.area_upper = np.append(self.area_upper, pred[4:].max())
            self.area_lower = np.append(self.area_lower, pred[4:].min())

            self.pred2 = concat(self.pred2, pred, 3)
            self.pred3 = concat(self.pred3, pred, 6)
            self.pred4 = concat(self.pred4, pred, 9)
            self.pred5 = concat(self.pred5, pred, 11)
            # self._time = concat(self._time, time)
            ax.clear()
            ax.grid()
            
            for i in range(self.sequence):
                self.labels[i] = concat(self.labels[i], self.label[self.idx, i])
                self.preds[i] = np.append(self.preds[i], pred[i])

            # print(self.dataset.label[min_scope:min_scope+self.scope, :, 0])
            for i in range(self.sequence):
                # ax.plot(self.labels[i][min_scope:], alpha=0.4, linewidth=(12-i), label=f"Label {i}")
                ax.plot(self.preds[i][min_scope:], alpha=0.4, linewidth=i//2, label=f"Preds {i}")
                
                
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
            
            xtick = np.arange(0, self.scope, 12)
            # ax.fill_between(np.arange(10), self.area_lower[min_scope:], self.area_upper[min_scope:], color='lightgray', alpha=0.5)
            
            values = self.time[min_scope:min_scope + self.scope:12]
            plt.xticks(xtick, values, rotation=30)

            # ax.relim()
            # ax.autoscale_view()

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
