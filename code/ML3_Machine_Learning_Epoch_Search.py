"""Code for finding the best model over epochs"""

import time
import torch
import warnings
import pandas as pd
import os.path as osp
import seaborn as sns
from util.helper import save_image, print_data_info, print_gpu_info
from DP1_Data_Preprocessing import KinaseDataset
from configs.ML3_Configs import configs
from torch_geometric.data import DataLoader
from ML1_Machine_Learning import GraphML

warnings.filterwarnings("ignore")


class EarlyStopping():
    # I referred to this class EarlyStopping from the link below
    #   https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch

    def __init__(self, min_delta=0.2, tolerance=20):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, test_loss):
        if (test_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


class EpochSearch(GraphML):
    def __init__(self, cf, OUT_DIR):
        super().__init__(cf, OUT_DIR)

    def run(self):
        # 1. Load Data
        dataset = KinaseDataset()

        # 2. Load Model
        model = self.cf['net'](dataset, self.cf)

        print(model)
        print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

        # 3. Set loss function
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        print(self.optimizer)

        # 4. Set GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(self.device)

        print_gpu_info(self.device)

        # 5. Load batch data
        self.train_loader = DataLoader(dataset[:int(len(dataset) * 0.8)], batch_size=self.cf['batch_size'], shuffle=True)
        self.test_loader = DataLoader(dataset[int(len(dataset) * 0.8):], batch_size=self.cf['batch_size'])

        print_data_info(dataset)
        print('Train data size:', len(self.train_loader.dataset), 'Test data size:', len(self.test_loader.dataset))

        # 6. Training, testing, and finding the best model
        print("Starting...")

        losses = []
        start = time.time()
        early_stopping = EarlyStopping()

        for epoch in range(self.cf['epoch_size']):
            train_loss = self.train(model)
            test_loss, _, _ = self.test(model)
            losses.append([train_loss, test_loss])

            # Check early stopping
            early_stopping(train_loss, test_loss)

            if early_stopping.early_stop:
                # Save the learned model
                torch.save(model, osp.join(self.OUT_DIR + f'[Selected] {model.name}-{epoch}.pt'))
                break

            if epoch % 5 == 0:
                print(f"Epoch {epoch} | Train Loss {format(train_loss, '.4f')} | Test Loss {format(test_loss, '.4f')} | {format((time.time() - start), '.2f')} sec")
                start = time.time()

        # 7. Draw an image to show test and training loss
        sns.lineplot(data=pd.DataFrame(losses, columns=['train_loss', 'test_loss']))
        save_image(self.OUT_DIR, f'{model.name}-training_loss')


if __name__ == '__main__':
    EpochSearch(configs[0], '../data/output/ML3/').run()
