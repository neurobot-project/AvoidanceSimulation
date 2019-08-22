import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np


class Trainer(object):
    def __init__(self, model, memory, device, batch_size):
        """
        Train the trainable model of a policy
        """
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss().to(device)
        self.memory = memory
        self.data_loader = None
        self.batch_size = batch_size
        self.optimizer = None

    def set_learning_rate(self, learning_rate):
        logging.info('Current learning rate: %f', learning_rate)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)

    def optimize_epoch(self, num_epochs):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        average_epoch_loss = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            for data in self.data_loader:
                inputs, values = data
                inputs = Variable(inputs)
                values = Variable(values)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, values)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data.item()

            average_epoch_loss = epoch_loss / len(self.memory)
            logging.debug('Average loss in epoch %d: %.2E', epoch, average_epoch_loss)

        return average_epoch_loss

    def optimize_batch(self, num_batches):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        losses = 0
        for _ in range(num_batches):
            inputs, values = next(iter(self.data_loader))
            inputs = Variable(inputs)
            values = Variable(values)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, values)
            loss.backward()
            self.optimizer.step()
            losses += loss.data.item()

        average_loss = losses / num_batches
        logging.debug('Average loss : %.2E', average_loss)

        return average_loss


class SFTrainer(object):
    def __init__(self, model, memory, device, batch_size):
        """
        Train the trainable model of a policy
        """
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss().to(device)
        self.memory = memory
        self.data_loader = None
        self.batch_size = batch_size
        self.optimizer = None
        self.w_lr = 0.001

    def set_learning_rate(self, learning_rate):
        logging.info('Current learning rate: %f', learning_rate)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)

    def optimize_epoch(self, num_epochs):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        average_epoch_loss = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            w_losses = []
            for data in self.data_loader:
                states, sf_values, actions, next_states, rewards = data
                states = Variable(states)
                sf_values = Variable(sf_values)

                self.optimizer.zero_grad()
                outputs = self.model(states)
                loss = self.criterion(outputs, sf_values)
                if torch.sum(torch.isnan(loss)):
                    print(data)
                    print("NaN detected!: epoch opt :: "+str(epoch))
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data.item()

                w_vec = self.model.w_vec.detach().cpu().numpy().reshape(-1)
                phis = states.detach().cpu().numpy()
                r_loss = rewards.detach().cpu().numpy().reshape(-1) - np.matmul(phis, w_vec)
                w_loss = np.mean(np.multiply(r_loss.reshape(-1, 1), phis), axis=0)
                w_vec = w_vec + self.w_lr*w_loss
                self.model.w_vec = torch.tensor(w_vec.reshape(1, -1))
                w_losses.append(np.mean(w_loss))
            
            # print(outputs)
            average_epoch_loss = epoch_loss / len(self.memory)
            logging.debug('Average loss in epoch %d: %.2E', epoch, average_epoch_loss)
            logging.debug('Average w loss in epoch %d: %.2E', epoch, np.mean(np.array(w_losses)))

        return average_epoch_loss

    def optimize_batch(self, num_batches):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        losses = 0
        w_losses = 0
        for _ in range(num_batches):
            states, sf_values, actions, next_states, rewards = next(iter(self.data_loader))
            # inputs, values = next(iter(self.data_loader))
            states = Variable(states)
            sf_values = Variable(sf_values)

            # values = Variable(values)

            self.optimizer.zero_grad()
            outputs = self.model(states)
            loss = self.criterion(outputs, sf_values)
            if torch.sum(torch.isnan(loss)):
                print("NaN detected!: epoch opt :: "+str(epoch))
            loss.backward()
            self.optimizer.step()
            losses += loss.data.item()

            w_vec = self.model.w_vec.detach().cpu().numpy().reshape(-1)
            phis = states.detach().cpu().numpy()
            r_loss = rewards.detach().cpu().numpy().reshape(-1) - np.matmul(phis, w_vec)
            w_loss = np.mean(np.multiply(r_loss.reshape(-1, 1), phis), axis=0)
            w_vec = w_vec + self.w_lr*w_loss
            self.model.w_vec = torch.tensor(w_vec.reshape(1, -1))
            w_losses += np.mean(w_losses)
            # print(loss.data.item())
        
        # print(outputs)
        average_loss = losses / num_batches
        average_w_loss = w_losses / num_batches
        logging.debug('Average loss : %.2E', average_loss)
        logging.debug('Average w loss : %.2E', average_w_loss)

        return average_loss