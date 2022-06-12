import torch

def accuracy(prob, labels):
    '''
        Calculate the accuracy (= sum(no. of sample that prediction is fully correct)/batch_size) 

        Args:
            prob: a batch of predictions in probability from model
            labels: a batch of labels from data loader
    '''
    prob = prob.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    preds = prob>0.5
    return ((labels==preds).sum(axis=1)==4).sum() / len(preds)
    # return (((labels==preds).sum(axis=1)==4).sum() / len(preds)).item()

def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def iou(pr, gt, eps=1e-7, threshold=0.5, ignore_channels=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    return ((intersection + eps) / union).item()