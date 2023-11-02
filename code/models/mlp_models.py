import copy
import os

import torch
import torch.nn as nn
import numpy as np

from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm, trange
from sklearn.metrics import r2_score, mean_squared_error


class MLPRegression(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))

        for i in range(1, len(hidden_dims)):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(p=dropout))
        self.layers.append(nn.Linear(hidden_dims[-1], 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.flatten() # remove the output dim which is 1

class MultiTargetMLPRegression(MLPRegression):
    def __init__(self, input_dim, hidden_dims, dropout, n_target):
        super().__init__(input_dim, hidden_dims, dropout)
        # Re-use the original regression architecture,
        # Only change the last layer to have multiple targets
        # (Not sure if it's a good thing to do though...)
        self.layers[-1] = nn.Linear(hidden_dims[-1], n_target)

    def forward(self, x):
        target_idx = x[:, -1]
        x = x[:, :-1] # last dim will be an index for the metric name
        
        for layer in self.layers:
            x = layer(x)

        # supposedly after this x is (batch_size, n_targets)
        # but we only select the relevant targets
        # so we use torch.gather()
        selected_targets = torch.gather(x, 1, target_idx.view(-1, 1).long())

        return selected_targets.flatten()
        
def train(args, logger, model, X_train, y_train, X_dev, y_dev):
    # does early stopping if X_dev is not None
    # otherwise just train all the way till the end

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if torch.cuda.is_available():
        model.cuda()

    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    loss_fn = torch.nn.MSELoss()

    best_dev_rmse, best_r2_score, best_state_dict, best_epoch = 1e10, -1, None, None

    for epoch in trange(30):
        model.train()

        for batch in tqdm(dataloader, disable=not args.verbose):

            if torch.cuda.is_available():
                batch = [item.cuda() for item in batch]

            X, y_true = batch
            y_pred = model(X)
            loss = loss_fn(y_pred, y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_rmse, r2_score, _ = eval_(model, X_train, y_train)
        logger.info("[Epoch {}] train_rmse: {:.4f}, r2_score: {:.4f}".format(epoch, train_rmse, r2_score))

        if X_dev is not None:
            dev_rmse, r2_score, _ = eval_(model, X_dev, y_dev)
            logger.info("[Epoch {}] dev_rmse: {:.4f}, r2_score: {:.4f}".format(epoch, dev_rmse, r2_score))

            if dev_rmse < best_dev_rmse:
                best_dev_rmse = dev_rmse
                best_r2_score = r2_score
                best_state_dict = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                logger.info("Saving best model at epoch {}".format(epoch))

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # if args.save_model:
    # save_filename = os.path.join(args.output_dir, "model.pt")
    # torch.save(best_state_dict, save_filename)

    return best_dev_rmse, best_r2_score

def eval_(model, X_eval, y_eval):
    model.eval()

    dataset = TensorDataset(X_eval, y_eval)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_predictions = []

    for batch in dataloader:

        if torch.cuda.is_available():
            batch = [item.cuda() for item in batch]
            
        X, y_true = batch

        with torch.no_grad():
            y_pred = model(X)

        all_predictions.append(y_pred)

    # flatten the predictions
    all_predictions = torch.concat(all_predictions, axis=0).cpu().numpy()

    mse = mean_squared_error(y_true=y_eval, y_pred=all_predictions) # np.mean((y_pred - y_dev)**2)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true=y_eval, y_pred=all_predictions)

    return rmse, r2, all_predictions

def train_for_search(args, logger, model, X_train, y_train, X_dev, y_dev):
    # several optimization for maximal GPU utilization
    assert torch.cuda.is_available()
    X_train_gpu = X_train.cuda()
    y_train_gpu = y_train.cuda()
    X_dev_gpu = X_dev.cuda()
    y_dev_gpu = y_dev.cuda()
    model.cuda()

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    dataset = TensorDataset(X_train_gpu, y_train_gpu)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    loss_fn = torch.nn.MSELoss()

    best_dev_rmse, best_r2_score, best_state_dict, best_epoch = 1e10, -1, None, None

    for epoch in trange(30):
        model.train()

        for batch in tqdm(dataloader, disable=not args.verbose):
            X, y_true = batch
            y_pred = model(X)
            loss = loss_fn(y_pred, y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # omitted because we don't need this during search
        # train_rmse, r2_score, _ = eval_(model, X_train, y_train)
        # logger.info("[Epoch {}] train_rmse: {:.4f}, r2_score: {:.4f}".format(epoch, train_rmse, r2_score))

        if X_dev is not None:
            dev_rmse, r2_score, _ = eval_for_search(model, X_dev_gpu, y_dev_gpu, X_dev, y_dev)
            
            if dev_rmse < best_dev_rmse:
                best_dev_rmse = dev_rmse
                best_r2_score = r2_score
                # we don't need this for search
                # best_state_dict = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                # logger.info("Saving best model at epoch {}".format(epoch))
                # logger.info("[Epoch {}] dev_rmse: {:.4f}, r2_score: {:.4f}".format(epoch, dev_rmse, r2_score))


    # we don't need this for search
    # if best_state_dict is not None:
    #     model.load_state_dict(best_state_dict)

    return best_dev_rmse, best_r2_score

def eval_for_search(model, X_eval_gpu, y_eval_gpu, X_eval, y_eval):
    model.eval()

    dataset = TensorDataset(X_eval_gpu, y_eval_gpu)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_predictions = []

    for batch in dataloader:

        if torch.cuda.is_available():
            batch = [item.cuda() for item in batch]
            
        X, y_true = batch

        with torch.no_grad():
            y_pred = model(X)

        all_predictions.append(y_pred)

    # flatten the predictions
    all_predictions = torch.concat(all_predictions, axis=0).cpu().numpy()

    mse = mean_squared_error(y_true=y_eval, y_pred=all_predictions) # np.mean((y_pred - y_dev)**2)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true=y_eval, y_pred=all_predictions)

    return rmse, r2, all_predictions