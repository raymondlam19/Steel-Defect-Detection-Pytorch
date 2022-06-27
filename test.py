import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from torch.autograd import Variable
import numpy as np
import pandas as pd

from utils import build_rles

def main(config):
    logger = config.get_logger('test')

    data_loader = config.init_obj('data_loader', module_data)
    test_loader = data_loader.test_loader

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)

    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    df_out = pd.DataFrame()

    logger.info("Predicting testset...")
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            imgids = data['ImageId']
            images = Variable(data['image']).to(device)
            if checkpoint['arch']=="FirstHalfUNet":
                preds_prob = model(images)
            else:
                preds_prob, _ = model(images)
            preds = (preds_prob>0.5).float()
            preds = preds.detach().cpu().numpy()
            
            if checkpoint['arch']=="FirstHalfUNet":
                pred_1 = preds[:, 0]
                pred_2 = preds[:, 1]
                pred_3 = preds[:, 2]
                pred_4 = preds[:, 3]
            else:
                pred_1 = build_rles(preds[:, 0, :, :])
                pred_2 = build_rles(preds[:, 1, :, :])
                pred_3 = build_rles(preds[:, 2, :, :])
                pred_4 = build_rles(preds[:, 3, :, :])
                
            df = pd.DataFrame({
                'ImageId': imgids,
                '1': pred_1,
                '2': pred_2,
                '3': pred_3,
                '4': pred_4})
            
            df_out = pd.concat([df_out, df], axis=0, ignore_index=True)
    
    if checkpoint['arch']=="FirstHalfUNet":
        df_out.to_csv('data/testset_label_output.csv', index=False)
        logger.info("Saved testset_label_output.csv")
    else:
        df_out.to_csv('data/testset_rle_output.csv', index=False)
        logger.info("Saved testset_rle_output.csv")

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
