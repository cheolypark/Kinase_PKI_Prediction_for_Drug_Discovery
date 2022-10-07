"""Code for performing a single machine learning."""

import os
import time
import torch
import warnings
import pandas as pd
import os.path as osp
import seaborn as sns
from util.helper import save_image, print_data_info, print_gpu_info
from sklearn.metrics import mean_squared_error, mean_absolute_error
from DP1_Data_Preprocessing import KinaseDataset
from configs.ML1_Configs import configs
from torch_geometric.data import DataLoader
from datetime import datetime

warnings.filterwarnings("ignore")

#===================================================================================================================
# Code reference:
#  I referred to the code in the link below.
#  And I edited this code to address the kinase GNN training
#   https://colab.research.google.com/drive/16GBgwYR2ECiXVxA1BoLxYshKczNMeEAQ?usp=sharing
#===================================================================================================================


class GraphML():
    def __init__(self, cf, OUT_DIR):
        self.cf = cf
        self.experiment_time = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
        self.OUT_DIR = osp.join(OUT_DIR + f'{self.experiment_time}/')

        os.makedirs(self.OUT_DIR, exist_ok=True)

    def train(self, model):
        model.train()
        loss = None

        for data in self.train_loader:
            model.change_data_to_address_this_model(data)

            data = data.to(self.device)

            self.optimizer.zero_grad()

            out = model(data)

            loss = self.loss_fn(out.squeeze(), data.y)
            loss.backward()

            self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def test(self, model):
        model.eval()

        y_pred, y_true = [], []

        for data in self.test_loader:
            model.change_data_to_address_this_model(data)

            data = data.to(self. device)
            out = model(data)

            y_pred.append(out.view(-1).cpu())
            y_true.append(data.y.view(-1).cpu().to(torch.float))

        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)

        loss = self.loss_fn(y_pred, y_true)

        return loss.item(), y_true, y_pred

    def test_final(self, model):
        loss, y_true, y_pred = self.test(model)

        y_true = y_true.tolist()
        y_pred = y_pred.tolist()

        df_test = pd.DataFrame(data={'y_true': y_true, 'y_pred': y_pred})

        print(df_test)

        # Compute prediction performance
        mae = mean_absolute_error(df_test.y_true, df_test.y_pred)
        mse = mean_squared_error(df_test.y_true, df_test.y_pred)
        rmse = mean_squared_error(df_test.y_true, df_test.y_pred, squared = False)

        mae = float("{:.4f}".format(mae))
        mse = float("{:.4f}".format(mse))
        rmse = float("{:.4f}".format(rmse))

        key = ['NET', 'MAE', 'MSE', 'RMSE']
        value = [self.cf['net'].name, mae, mse, rmse]

        for k, v in self.cf.items():
            if type(self.cf[k]) != type:
                key.append(k)
                value.append(v)

        df_performance = pd.DataFrame(data={'key': key, 'value': value})
        df_performance.to_csv(osp.join(self.OUT_DIR + f'{model.name}.csv'), index=False)

        # Draw a result image showing data points between true and prediction
        sns.scatterplot(data=df_test, x="y_true", y="y_pred")
        save_image(self.OUT_DIR, f'{model.name}-comparison_with_ground_truth')

    def run(self):
        # 1. Load Data
        dataset = KinaseDataset()

        # 2. Load Model
        model = self.cf['net'](dataset, self.cf)

        print(model)
        print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

        # 3. Set loss function
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # lr=0.00005

        print(self.optimizer)

        # 4. Set GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = 'cpu'
        model = model.to(self.device)

        print_gpu_info(self.device)

        # 5. Load batch data
        self.train_loader = DataLoader(dataset[:int(len(dataset) * 0.8)], batch_size=self.cf['batch_size'], shuffle=True)
        self.test_loader = DataLoader(dataset[int(len(dataset) * 0.8):], batch_size=self.cf['batch_size'])

        print_data_info(dataset)
        print('Train data size:', len(self.train_loader.dataset), 'Test data size:', len(self.test_loader.dataset))

        print("Starting training...")

        losses = []
        start = time.time()

        for epoch in range(self.cf['epoch_size']):
            train_loss = self.train(model)
            test_loss, _, _ = self.test(model)
            losses.append([train_loss, test_loss])

            if epoch % 5 == 0:
                print(f"Epoch {epoch} | Train Loss {format(train_loss, '.4f')} | Test Loss {format(test_loss, '.4f')} | {format((time.time() - start), '.2f')} sec")
                start = time.time()

        # 7. Draw an image to show test and training loss
        df_loss = pd.DataFrame(losses, columns=['train_loss', 'test_loss'])
        sns.lineplot(data=df_loss)
        save_image(self.OUT_DIR, f'{model.name}-training_loss')

        # 8. Save test and training loss
        df_loss.to_csv(osp.join(self.OUT_DIR + f'[loss] {model.name}.csv'), index=False)

        # 9. Save the learned model
        torch.save(model, osp.join(self.OUT_DIR + f'{model.name}.pt'))

        # 10. Final Test
        self.test_final(model)


if __name__ == '__main__':
    GraphML(configs[0], '../data/output/ML1/').run()
