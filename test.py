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
    # df_out = pd.DataFrame({
    #     'ImageId': pd.Series(dtype='str'),
    #     '1': pd.Series(dtype='str'),
    #     '2': pd.Series(dtype='str'),
    #     '3': pd.Series(dtype='str'),
    #     '4': pd.Series(dtype='str')})

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            imgids = data['ImageId']
            images = Variable(data['image']).to(device)
            preds_prob = model(images)
            preds = (preds_prob>0.5).float()
            preds = preds.detach().cpu().numpy()
            
            pred_rle_1 = build_rles(preds[:, 0, :, :])
            pred_rle_2 = build_rles(preds[:, 1, :, :])
            pred_rle_3 = build_rles(preds[:, 2, :, :])
            pred_rle_4 = build_rles(preds[:, 3, :, :])

            df = pd.DataFrame({
                'ImageId': imgids,
                '1': pred_rle_1,
                '2': pred_rle_2,
                '3': pred_rle_3,
                '4': pred_rle_4})

            df_out = pd.concat([df_out, df], axis=0, ignore_index=True)
    
    df_out.to_csv('data/testset_rle_output.csv', index=False)
    print('Saved testset_rle_output.csv')

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
