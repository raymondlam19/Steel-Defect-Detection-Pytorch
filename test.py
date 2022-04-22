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


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    # data_loader = getattr(module_data, config['data_loader']['type'])(
    #     config['data_loader']['args']['data_dir'],
    #     batch_size=512,
    #     shuffle=False,
    #     validation_split=0.0,
    #     training=False,
    #     num_workers=2
    # )

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
    
    testset_imgids = []
    testset_preds = []

    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):
            imgids = data['ImageId']
            images = Variable(data['image']).to(device)
            preds_prob = model(images)
            preds = (preds_prob>0.5).float()

            testset_imgids.extend(imgids)
            testset_preds.extend(preds.cpu().detach().tolist())

    testset_imgids = np.array(testset_imgids)
    testset_preds = np.array(testset_preds)

    #overwrite the testset.csv
    testset_output = pd.DataFrame({
        'ImageId': testset_imgids,
        '1': testset_preds[:,0],
        '2': testset_preds[:,1],
        '3': testset_preds[:,2],
        '4': testset_preds[:,3]
    })
    testset_output.to_csv('data/testset_output.csv', index=False)
    print('Saved testset_output.csv')

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
