import torch

def accuracy(prob, labels):
    '''
        Calculate the accuracy (= sum(no. of sample that prediction is fully correct)/batch_size) 

        Args:
            prob: a batch of predictions in probability from model
            labels: a batch of labels from data loader
    '''
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    preds = prob>0.5
    return ((labels==preds).sum(axis=1)==4).sum() / len(preds)
    # return (((labels==preds).sum(axis=1)==4).sum() / len(preds)).item()