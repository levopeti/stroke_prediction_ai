import torch

from tqdm import tqdm
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

class TrainLoop(object):
    def __init__(self,
                 model: Module,
                 optimizer: Optimizer,
                 loss: Module,
                 train_loader: DataLoader,
                 valid_loader: DataLoader,
                 num_epoch: int):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_epoch = num_epoch

    def run_loop(self):
        for epoch in range(self.num_epoch):
            # print("\nepoch: {}".format(epoch))
            self.model.train()

            correct = 0
            total = 0

            tq = tqdm(total=(len(self.train_loader)))
            tq.set_description('epoch {}'.format(epoch))
            for batch_idx, (x, y) in enumerate(self.train_loader):
                # print(batch_idx, x, y)
                self.optimizer.zero_grad()

                model_out = torch.squeeze(self.model(x.float()))

                # print(model_out.shape, y.shape)
                loss = self.loss(model_out, y)  # index of the max log-probability
                loss.backward()

                self.optimizer.step()

                total += y.size(0)

                _, predicted = model_out.max(1)
                correct += predicted.eq(y).sum().item()

                acc = correct / total

                tq.update(1)
                tq.set_postfix(acc='{:.1f}%'.format(acc * 100),
                               loss='{:.2f}'.format(loss.item()))
                # print("{:.2f}%, loss: {:.2f}, acc: {:.1f}%".format(batch_idx/len(self.train_loader), loss.item(), acc * 100))

            tq.close()
            self.validation_step()

    def validation_step(self):
        with torch.no_grad():
            self.model.eval()
            correct = 0
            loss = list()

            total = 0

            for batch_idx, (x, y) in enumerate(self.valid_loader):
                total += y.size(0)

                model_out = torch.squeeze(self.model(x.float()))
                # print(model_out.shape, y.shape)
                loss.append(self.loss(model_out, y).item())

                _, predicted = model_out.max(1)
                correct += predicted.eq(y).sum().item()
                acc = correct / total

            print("val_acc: {:.1f}%, val_loss: {:.2f}".format(acc * 100, sum(loss) / len(loss)))