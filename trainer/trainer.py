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

        if self.model.__class__.__name__ == 'FirstHalfUNet':
            metrics_list = ['loss'] + list(self.metric_ftns.keys())
        else:
            metrics_list = ['loss', 'loss_mask', 'loss_label'] + list(self.metric_ftns.keys())
        self.train_metrics = MetricTracker(*metrics_list, writer=self.writer)
        self.valid_metrics = MetricTracker(*metrics_list, writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, batch in enumerate(self.train_loader):
            if self.model.__class__.__name__ == 'FirstHalfUNet':
                img = Variable(batch['image']).to(self.device)
                label_tl = Variable(batch['label']).to(self.device)

                label_pred = self.model(img)
                loss = self.criterion(label_pred, label_tl)
            else:
                img = Variable(batch['image']).to(self.device)
                label_tl = Variable(batch['label']).to(self.device)
                mask_tl = Variable(batch['mask']).to(self.device)

                mask_pred, label_pred = self.model(img)
                loss_mask = self.criterion(mask_pred, mask_tl)
                loss_label = self.criterion(label_pred, label_tl)
                loss = loss_mask + loss_label

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            if self.model.__class__.__name__ == 'UNet':
                self.train_metrics.update('loss_mask', loss_mask.item())
                self.train_metrics.update('loss_label', loss_label.item())
                self.train_metrics.update('iou', self.metric_ftns['iou'](label_pred, label_tl))           
            self.train_metrics.update('accuracy', self.metric_ftns['accuracy'](label_pred, label_tl))

            if batch_idx % self.log_step == 0:
                if self.model.__class__.__name__ == 'FirstHalfUNet':
                    self.logger.debug('Epoch: {}, Step: {}, step_train_loss: {:.4f}'.format(
                        epoch,
                        self._progress(batch_idx),
                        loss.item()
                        ))
                else:
                    self.logger.debug('Epoch: {}, Step: {}, step_train_loss: {:.4f}, step_train_loss_mask: {:.4f}, step_train_loss_label: {:.4f}'.format(
                        epoch,
                        self._progress(batch_idx),
                        loss.item(),
                        loss_mask.item(),
                        loss_label.item()
                        ))
                self.writer.add_image('input', make_grid(img.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        train_log = self.train_metrics.result()
        self.log.update(**{'train_'+k : v for k, v in train_log.items()})

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            self.log.update(**{'val_'+k : v for k, v in val_log.items()})

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
                if self.model.__class__.__name__ == 'FirstHalfUNet':
                    img = Variable(batch['image']).to(self.device)
                    label_tl = Variable(batch['label']).to(self.device)

                    label_pred = self.model(img)
                    loss = self.criterion(label_pred, label_tl)
                else:
                    img = Variable(batch['image']).to(self.device)
                    label_tl = Variable(batch['label']).to(self.device)
                    mask_tl = Variable(batch['mask']).to(self.device)

                    mask_pred, label_pred = self.model(img)
                    loss_mask = self.criterion(mask_pred, mask_tl)
                    loss_label = self.criterion(label_pred, label_tl)
                    loss = loss_mask + loss_label

                self.writer.set_step((epoch - 1) * len(self.val_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                if self.model.__class__.__name__ == 'UNet':
                    self.valid_metrics.update('loss_mask', loss_mask.item())
                    self.valid_metrics.update('loss_label', loss_label.item())
                    self.valid_metrics.update('iou', self.metric_ftns['iou'](label_pred, label_tl))           
                self.valid_metrics.update('accuracy', self.metric_ftns['accuracy'](label_pred, label_tl))

                self.writer.add_image('input', make_grid(img.cpu(), nrow=8, normalize=True))

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

