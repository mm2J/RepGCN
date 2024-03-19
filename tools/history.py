class History():
    def __init__(self, dir):
        self.csv_dir = os.path.join(dir, 'metrics_outputs.csv')
        self.pic_dir = os.path.join(dir, 'loss-acc.png')
        self.losses_epoch = []
        self.acc_epoch = []
        self.epoch_outputs = [['Epoch', 'Train Loss', 'Avg Acc', 'NM Avg Acc', 'BG Avg Acc', 'CL Avg Acc']]
        self.temp_data = []

    def update(self, data, mode):
        if mode == 'train':
            self.temp_data.append(data)
            self.losses_epoch.append(data)
        elif mode == 'test':
            self.temp_data.extend(
                [data.get('avg_acc'),
                 data.get('nm_avg_acc', 0.0),
                 data.get('bg_avg_acc', 0.0),
                 data.get('cl_avg_acc', 0.0)])
            self.acc_epoch.append(data.get('avg_acc'))

    def after_epoch(self, meta):
        '''
        保存每周期的 'Train Loss', 'Avg Acc', 'NM#5-6', 'BG#1-2', 'CL#1-2'
        '''
        acc_epoch = []
        epoch_outputs = []
        with open(self.csv_dir, 'w', newline='') as f:
            writer = csv.writer(f)
            for i in range(len(meta['train_info']['train_loss'])):
                temp_data = [i + 1, meta['train_info']['train_loss'][i],
                             meta['train_info']['acc_result'][i].get('avg_acc', 0.0),
                             mean(meta['train_info']['acc_result'][i].get('nm_avg_acc', 0.0)),
                             mean(meta['train_info']['acc_result'][i].get('bg_avg_acc', 0.0)),
                             mean(meta['train_info']['acc_result'][i].get('cl_avg_acc', 0.0))]
                acc_epoch.append(meta['train_info']['acc_result'][i].get('avg_acc'))
                epoch_outputs.append(temp_data)
            writer.writerows(epoch_outputs)

        '''
        绘制每周期Train Loss以及Validation Accuracy
        '''
        total_epoch = range(1, len(meta['train_info']['train_loss']) + 1)

        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.plot(total_epoch, meta['train_info']['train_loss'], 'red', linewidth=2, label='Train loss')
        ax1.grid(True)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Acc')
        ax2.plot(total_epoch, acc_epoch, 'blue', linewidth=2, label='Avg Acc')
        fig.legend()
        fig.tight_layout()
        plt.savefig(self.pic_dir)
        plt.close("all")