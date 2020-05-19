#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import os

import numpy as np

import chainer
import chainer.cuda
import cv2 as cv
from c3d_ft import C3DVersion1
from chainer import Variable
from chainer import cuda
from tqdm import tqdm

sys.path.insert(0, '.')  # isort:skip
# from infer import get_models  # isort:skip
# from infer import make_video  # isort:skip
import random
import skvideo.io

def calc_inception(ys):
    N, C = ys.shape
    p_all = np.mean(ys, axis=0, keepdims=True)
    kl = np.sum(ys * np.log(ys + 1e-7) - ys * np.log(p_all + 1e-7)) / N
    return np.exp(kl)


def main():
    parser = argparse.ArgumentParser(description='inception score')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--result_dir', type=str, default='/home/jugatl2/tgan/evaluation/UCF101/')
    parser.add_argument('--iter', type=int, default=100000)
    parser.add_argument('--calc_iter', type=int, default=10000)
    # parser.add_argument('--mean', type=str, default='/mnt/sakura201/mitmul/codes/tgan2_orig/inception/mean2.npz')
    parser.add_argument('--mean', type=str, default='./mean2.npz')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--interpolation', type=str, default='INTER_CUBIC')
    parser.add_argument('--clip_length', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    np.random.seed(args.seed)

    inter_method = args.interpolation
    args.interpolation = getattr(cv, args.interpolation)

    cuda.get_device(args.gpu).use()
    chainer.cuda.cupy.random.seed(args.seed)
    xp = chainer.cuda.cupy

    c3dmodel = C3DVersion1('auto')
    c3dmodel.to_gpu()

    # load model
    # fsgen, vgen, _ = get_models(args.result_dir, args.iter)       # no model
    # if args.gpu >= 0:
    #     fsgen.to_gpu()
    #     vgen.to_gpu()
    batchsize = args.batch_size

    mean = np.load(args.mean)['mean'].astype('f')                       #what mean do we use? 
    mean = mean.reshape((3, 1, 16, 128, 171))[:, :, :, :, 21:21 + 128]

    # generator
    ys = []
    for i in tqdm(range(args.calc_iter // batchsize)):
        for k in range(batchsize):
            while True:
                filename = random.choice(os.listdir(args.result_dir))
                if filename.endswith('.avi'):
                    break
            random_file = skvideo.io.vread(os.path.join(args.result_dir, filename))
            num_frames = random_file.shape[0]
            start_frame = random.randint(0, num_frames - args.clip_length)
            random_clip = random_file[start_frame:start_frame+args.clip_length, : , :]
            random_clip_ = np.zeros((args.clip_length, 128, 128, 3))
            for t in range(args.clip_length):
                random_clip_[t] = np.asarray(
                    cv.resize(random_clip[t], (128, 128), interpolation=args.interpolation))
            if k == 0:
                x = random_clip_
            elif k ==1:
                x = np.concatenate((np.expand_dims(x, axis=0), np.expand_dims(random_clip_, axis=0)))
            else:
                x = np.concatenate((x, np.expand_dims(random_clip_, axis=0)))

        n, f, h, w, c = x.shape
        # x = x.reshape(n * f, h, w, c)              #folding time dim into batch dim 
        # # x = x * 128 + 128                                                   #what's going on here?
        # x_ = np.zeros((n * f, 128, 128, 3))
        # for t in range(n * f):                                              #frame by frame
        #     x_[t] = np.asarray(
        #         cv.resize(x[t], (128, 128), interpolation=args.interpolation))
        x = x.transpose(4, 0, 1, 2, 3).reshape(3, n, f, 128, 128)
        x = x[::-1] - mean  # mean file is BGR-order while model outputs RGB-order
        x = x[:, :, :, 8:8 + 112, 8:8 + 112].astype('f')                    #what's going on here?
        x = x.transpose(1, 0, 2, 3, 4)
        with chainer.using_config('train', False) and \
                chainer.no_backprop_mode():
            # C3D takes an image with BGR order
            y = c3dmodel(Variable(xp.asarray(x)),
                         layers=['prob'])['prob'].data.get()                #get labels for generated videos
            ys.append(y)
    ys = np.asarray(ys).reshape((-1, 101))

    score = calc_inception(ys)
    with open('{}/inception_iter-{}_{}.txt'.format(args.result_dir, args.iter, inter_method), 'w') as fp:
        print(args.result_dir, args.iter, args.calc_iter, args.mean, score, file=fp)
        print(args.result_dir, args.iter, 'score:{}'.format(score))

    return 0


if __name__ == '__main__':
    sys.exit(main())
