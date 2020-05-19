''' Calculate Inception Moments
 This script iterates over the dataset and calculates the moments of the
 activations of the Inception net (needed for FID), and also returns
 the Inception Score of the training data.

 Note that if you don't shuffle the data, the IS of true data will be under-
 estimated as it is label-ordered. By default, the data is not shuffled
 so as to reduce non-determinism. '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import utils
import inception_utils
from tqdm import tqdm, trange
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
import datetime
import datasets as dset

def prepare_parser():
  usage = 'Calculate and store inception metrics.'
  parser = ArgumentParser(description=usage)
  parser.add_argument(
    '--dataset', type=str, default='I128_hdf5',
    help='Which Dataset to train on, out of I128, I256, C10, C100...'
         'Append _hdf5 to use the hdf5 version of the dataset. (default: %(default)s)')
  parser.add_argument(
    '--data_root', type=str, default='data',
    help='Default location where data is stored (default: %(default)s)')
  parser.add_argument(
    '--batch_size', type=int, default=64,
    help='Default overall batchsize (default: %(default)s)')
  parser.add_argument(
    '--parallel', action='store_true', default=False,
    help='Train with multiple GPUs (default: %(default)s)')
  parser.add_argument(
    '--augment', action='store_true', default=False,
    help='Augment with random crops and flips (default: %(default)s)')
  parser.add_argument(
    '--num_workers', type=int, default=8,
    help='Number of dataloader workers (default: %(default)s)')
  parser.add_argument(
    '--time_steps', type=int, default=12,
    help='how many frames per video (default: %(default)s)')
  parser.add_argument(
    '--frames_between_clips', type=int, default=12,
    help='How many frames between the beginning of this clip and the beginning of the last clip. It should be the same as time_steps for most cases'
         '(default: %(default)s)')
  parser.add_argument(
    '--shuffle', action='store_true', default=False,
    help='Shuffle the data? (default: %(default)s)')
  parser.add_argument(
    '--seed', type=int, default=0,
    help='Random seed to use.')
  parser.add_argument(
    '--logs_root', type=str, default='logs',
    help='Default location to store logs (default: %(default)s)')
  return parser

def run(config):

  unique_id = datetime.datetime.now().strftime('%Y%m-%d%H-%M%S-')
  tensorboard_path = os.path.join(config['logs_root'], 'tensorboard_logs', unique_id)
  os.makedirs(tensorboard_path)
  writer = SummaryWriter(log_dir=tensorboard_path)

  # Get loader
  config['drop_last'] = False
  frame_size = utils.imsize_dict[config['dataset']] #112
  print('Dataset:',config['dataset'],'Frame size:',frame_size)
  loaders = utils.get_video_data_loaders(num_epochs = 1, frame_size = frame_size, **config)

  # Load inception net
  net = inception_utils.load_r2plus1d_18_net(parallel=config['parallel'])
  pool, logits, labels = [], [], []
  device = 'cuda'
  accu_correct, accu_total = 0, 0
  # norm_mean = [-0.43216/0.22803, -0.394666/0.22145, -0.37645/0.216989]
  # norm_std = [1/0.22803, 1/0.22145, 1/0.216989]
  # norm_mean = torch.tensor([0.43216, 0.394666, 0.37645]).to(device)
  # norm_std  = torch.tensor([0.22803, 0.22145, 0.216989]).to(device)
  norm_mean = torch.tensor([0.5, 0.5, 0.5]).to(device)
  norm_std  = torch.tensor([0.5, 0.5, 0.5]).to(device)
  transform_tensorboard = dset.VideoNormalize(norm_mean, norm_std)

  for i, (x, y) in enumerate(tqdm(loaders[0])):
    x = x.to(device) #[B,T,C,H,W]
    # print('x shape',x.shape)
    if i % 100 == 0:
      # t_x = transform_tensorboard(x[0].permute(1,0,2,3).contiguous()).permute(1,0,2,3).unsqueeze(0)
      t_x = x * norm_std[None,None,:,None,None] + norm_mean[None,None,:,None,None]
      print('Range:', t_x.min(),t_x.max())
      writer.add_video('Loaded Data', t_x, i)
    with torch.no_grad():
      pool_val, logits_val = net(x)
      pool += [np.asarray(pool_val.cpu())]
      logits += [np.asarray(F.softmax(logits_val, 1).cpu())]
      labels += [np.asarray(y.cpu())]
      accu_correct += int(sum(F.softmax(logits_val, 1).cpu().argmax(1).squeeze() == y.cpu().squeeze()))
      accu_total += len(np.asarray(y.cpu()).squeeze())
      if i % 10 == 0:
        print(F.softmax(logits_val, 1).cpu()[0])
        print('Accumulated coorect predictions:', accu_correct)
        print('Accumulated total number of samples:', accu_total)
        print('Accumulated prediction accuracy is:', accu_correct/accu_total)

  pool, logits, labels = [np.concatenate(item, 0) for item in [pool, logits, labels]]
  # uncomment to save pool, logits, and labels to disk
  # print('Saving pool, logits, and labels to disk...')
  # np.savez(config['dataset']+'_inception_activations.npz',
  #           {'pool': pool, 'logits': logits, 'labels': labels})
  # Calculate inception metrics and report them
  print('Calculating inception metrics...')
  IS_mean, IS_std = inception_utils.calculate_inception_score(logits)
  print('Training data from dataset %s has IS of %5.5f +/- %5.5f' % (config['dataset'], IS_mean, IS_std))
  # Prepare mu and sigma, save to disk. Remove "hdf5" by default
  # (the FID code also knows to strip "hdf5")
  print('Calculating means and covariances...')
  mu, sigma = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
  print('Saving calculated means and covariances to disk...')
  np.savez(config['dataset'].strip('_hdf5')+'_inception_moments.npz', **{'mu' : mu, 'sigma' : sigma})

def main():
  # parse command line
  parser = prepare_parser()
  config = vars(parser.parse_args())
  print(config)
  run(config)


if __name__ == '__main__':
    main()
