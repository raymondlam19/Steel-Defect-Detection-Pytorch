import torch

def accuracy(prob, labels):
    '''
        Calculate the accuracy (= sum(no. of sample that prediction is fully correct)/batch_size) 

        Args:
            prob: a batch of predictions in probability from model
            labels: a batch of labels from data loader
    '''
    preds = (prob>0.5).float()
    return (((labels==preds).sum(axis=1)==4).sum() / len(preds)).item()
    #return torch.tensor(((labels==preds).sum(axis=1)==4).sum() / len(preds)).item()

# def accuracy(output, target):
#     with torch.no_grad():
#         pred = torch.argmax(output, dim=1)
#         assert pred.shape[0] == len(target)
#         correct = 0
#         correct += torch.sum(pred == target).item()
#     return correct / len(target)


# def top_k_acc(output, target, k=3):
#     with torch.no_grad():
#         pred = torch.topk(output, k, dim=1)[1]
#         assert pred.shape[0] == len(target)
#         correct = 0
#         for i in range(k):
#             correct += torch.sum(pred[:, i] == target).item()
#     return correct / len(target)
