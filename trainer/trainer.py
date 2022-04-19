import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from torch.autograd import Variable
from tqdm import tqdm


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 train_loader, val_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.train_loader = train_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_loader)
        else:
            # iteration-based training
            self.train_loader = inf_loop(train_loader)
            self.len_epoch = len_epoch
        self.val_loader = val_loader
        self.do_validation = self.val_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = 100     #int(np.sqrt(train_loader.batch_size))

        self.log = MetricTracker().result()
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, batch in enumerate(self.train_loader):
            data = Variable(batch['image']).to(self.device)
            target = Variable(batch['label']).to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Epoch: {}, Step: {}, step_train_loss: {:.4f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        train_log = self.train_metrics.result()
        self.log.update(**{'train_'+k : v for k, v in train_log.items()})

        # add epoch_avg_train_loss into tensorboard
        try:
            print(self.log['train_loss'])
            print(self.log['train_accuracy'])
        except:
            print('train_metrics error')
        #self.writer.add_scalar('epoch_avg_train_loss', self.log['train_loss'])

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            self.log.update(**{'val_'+k : v for k, v in val_log.items()})

            # add epoch_avg_train_loss into tensorboard
            try:
                print(self.log['val_loss'])
                print(self.log['val_accuracy'])
            except:
                print('valid_metrics error')
            #self.writer.add_scalar('epoch_avg_val_loss', self.log['val_loss'])

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return self.log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader)):
                data = Variable(batch['image']).to(self.device)
                target = Variable(batch['label']).to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.val_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_loader, 'n_samples'):
            current = batch_idx * self.train_loader.batch_size
            total = self.train_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

