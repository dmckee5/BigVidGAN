''' Datasets
    This file contains definitions for our CIFAR, ImageFolder, and HDF5 datasets
'''
import os
import os.path
import sys
from PIL import Image
import numpy as np
from tqdm import tqdm, trange

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.io as io
from torchvision.datasets.utils import download_url, check_integrity
import torch.utils.data as data
from torch.utils.data import DataLoader
# from torchvision.datasets.video_utils import VideoClips
from VideoClips2 import VideoClips
from torchvision.datasets.utils import list_dir
import numbers
from glob import glob
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
  images = []
  dir = os.path.expanduser(dir)
  print
  for target in tqdm(sorted(os.listdir(dir))):
    d = os.path.join(dir, target)
    if not os.path.isdir(d):
      continue

    for root, _, fnames in sorted(os.walk(d)):
      for fname in sorted(fnames):
        if is_image_file(fname):
          path = os.path.join(root, fname)
          item = (path, class_to_idx[target])
          images.append(item)

  return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
  with open(path, 'rb') as f:
    img = Image.open(f)
    return img.convert('RGB')


def accimage_loader(path):
  import accimage
  try:
    return accimage.Image(path)
  except IOError:
    # Potentially a decoding problem, fall back to PIL.Image
    return pil_loader(path)


def default_loader(path):
  from torchvision import get_image_backend
  if get_image_backend() == 'accimage':
    return accimage_loader(path)
  else:
    return pil_loader(path)


class ImageFolder(data.Dataset):
  """A generic data loader where the images are arranged in this way: ::

      root/dogball/xxx.png
      root/dogball/xxy.png
      root/dogball/xxz.png

      root/cat/123.png
      root/cat/nsdf3.png
      root/cat/asd932_.png

  Args:
      root (string): Root directory path.
      transform (callable, optional): A function/transform that  takes in an PIL image
          and returns a transformed version. E.g, ``transforms.RandomCrop``
      target_transform (callable, optional): A function/transform that takes in the
          target and transforms it.
      loader (callable, optional): A function to load an image given its path.

   Attributes:
      classes (list): List of the class names.
      class_to_idx (dict): Dict with items (class_name, class_index).
      imgs (list): List of (image path, class_index) tuples
  """

  def __init__(self, root, transform=None, target_transform=None,
               loader=default_loader, load_in_mem=False,
               index_filename='imagenet_imgs.npz', **kwargs):
    classes, class_to_idx = find_classes(root)
    # Load pre-computed image directory walk
    if os.path.exists(index_filename):
      print('Loading pre-saved Index file %s...' % index_filename)
      imgs = np.load(index_filename)['imgs']
    # If first time, walk the folder directory and save the
    # results to a pre-computed file.
    else:
      print('Generating  Index file %s...' % index_filename)
      imgs = make_dataset(root, class_to_idx)
      np.savez_compressed(index_filename, **{'imgs' : imgs})
    if len(imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                           "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

    self.root = root
    self.imgs = imgs
    self.classes = classes
    self.class_to_idx = class_to_idx
    self.transform = transform
    self.target_transform = target_transform
    self.loader = loader
    self.load_in_mem = load_in_mem

    if self.load_in_mem:
      print('Loading all images into memory...')
      self.data, self.labels = [], []
      for index in tqdm(range(len(self.imgs))):
        path, target = imgs[index][0], imgs[index][1]
        self.data.append(self.transform(self.loader(path)))
        self.labels.append(target)


  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is class_index of the target class.
    """
    if self.load_in_mem:
        img = self.data[index]
        target = self.labels[index]
    else:
      path, target = self.imgs[index]
      img = self.loader(str(path))
      if self.transform is not None:
        img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    # print(img.size(), target)
    return img, int(target)

  def __len__(self):
    return len(self.imgs)

  def __repr__(self):
    fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
    fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
    fmt_str += '    Root Location: {}\n'.format(self.root)
    tmp = '    Transforms (if any): '
    fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    tmp = '    Target Transforms (if any): '
    fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    return fmt_str


''' ILSVRC_HDF5: A dataset to support I/O from an HDF5 to avoid
    having to load individual images all the time. '''
import h5py as h5
import torch
class ILSVRC_HDF5(data.Dataset):
  def __init__(self, root, transform=None, target_transform=None,
               load_in_mem=False, train=True,download=False, validate_seed=0,
               val_split=0, **kwargs): # last four are dummies

    self.root = root
    self.num_imgs = len(h5.File(root, 'r')['labels'])

    # self.transform = transform
    self.target_transform = target_transform

    # Set the transform here
    self.transform = transform

    # load the entire dataset into memory?
    self.load_in_mem = load_in_mem

    # If loading into memory, do so now
    if self.load_in_mem:
      print('Loading %s into memory...' % root)
      with h5.File(root,'r') as f:
        self.data = f['imgs'][:]
        self.labels = f['labels'][:]

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is class_index of the target class.
    """
    # If loaded the entire dataset in RAM, get image from memory
    if self.load_in_mem:
      img = self.data[index]
      target = self.labels[index]

    # Else load it from disk
    else:
      with h5.File(self.root,'r') as f:
        img = f['imgs'][index]
        target = f['labels'][index]


    # if self.transform is not None:
        # img = self.transform(img)
    # Apply my own transform
    img = ((torch.from_numpy(img).float() / 255) - 0.5) * 2

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, int(target)

  def __len__(self):
      return self.num_imgs
      # return len(self.f['imgs'])

import pickle
class CIFAR10(dset.CIFAR10):

  def __init__(self, root, train=True,
           transform=None, target_transform=None,
           download=True, validate_seed=0,
           val_split=0, load_in_mem=True, **kwargs):
    self.root = os.path.expanduser(root)
    self.transform = transform
    self.target_transform = target_transform
    self.train = train  # training set or test set
    self.val_split = val_split

    if download:
      self.download()

    if not self._check_integrity():
      raise RuntimeError('Dataset not found or corrupted.' +
                           ' You can use download=True to download it')

    # now load the picked numpy arrays
    self.data = []
    self.labels= []
    for fentry in self.train_list:
      f = fentry[0]
      file = os.path.join(self.root, self.base_folder, f)
      fo = open(file, 'rb')
      if sys.version_info[0] == 2:
        entry = pickle.load(fo)
      else:
        entry = pickle.load(fo, encoding='latin1')
      self.data.append(entry['data'])
      if 'labels' in entry:
        self.labels += entry['labels']
      else:
        self.labels += entry['fine_labels']
      fo.close()

    self.data = np.concatenate(self.data)
    # Randomly select indices for validation
    if self.val_split > 0:
      label_indices = [[] for _ in range(max(self.labels)+1)]
      for i,l in enumerate(self.labels):
        label_indices[l] += [i]
      label_indices = np.asarray(label_indices)

      # randomly grab 500 elements of each class
      np.random.seed(validate_seed)
      self.val_indices = []
      for l_i in label_indices:
        self.val_indices += list(l_i[np.random.choice(len(l_i), int(len(self.data) * val_split) // (max(self.labels) + 1) ,replace=False)])

    if self.train=='validate':
      self.data = self.data[self.val_indices]
      self.labels = list(np.asarray(self.labels)[self.val_indices])

      self.data = self.data.reshape((int(50e3 * self.val_split), 3, 32, 32))
      self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    elif self.train:
      # print(np.shape(self.data))
      if self.val_split > 0:
        self.data = np.delete(self.data,self.val_indices,axis=0)
        self.labels = list(np.delete(np.asarray(self.labels),self.val_indices,axis=0))

      self.data = self.data.reshape((int(50e3 * (1.-self.val_split)), 3, 32, 32))
      self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
    else:
      f = self.test_list[0][0]
      file = os.path.join(self.root, self.base_folder, f)
      fo = open(file, 'rb')
      if sys.version_info[0] == 2:
        entry = pickle.load(fo)
      else:
        entry = pickle.load(fo, encoding='latin1')
      self.data = entry['data']
      if 'labels' in entry:
        self.labels = entry['labels']
      else:
        self.labels = entry['fine_labels']
      fo.close()
      self.data = self.data.reshape((10000, 3, 32, 32))
      self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

  def __getitem__(self, index):
    """
    Args:
        index (int): Index
    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target = self.data[index], self.labels[index]

    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    img = Image.fromarray(img)

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

  def __len__(self):
      return len(self.data)

class videoCIFAR10(CIFAR10):
    def __init__(self, root, train=True,
         transform=None, target_transform=None,
         download=True, validate_seed=0,
         val_split=0, load_in_mem=True, **kwargs):
         super().__init__(root, train,
         transform, target_transform,
         download, validate_seed,
         val_split, load_in_mem, **kwargs)
         self.time_steps = kwargs['time_steps']
    def __getitem__(self,index):
        img, target = super().__getitem__(index)
        return torch.unsqueeze(img, dim=0).repeat(self.time_steps,1,1,1), target

    def __len__(self):
        return super().__len__()



class UCF101(data.Dataset):

  # def __init__(self, root, transform=None, video_len=12):
  #   self.file = os.path.expanduser(root)
  #   self.video_len = video_len
  #   self.transform =transform
  #   with h5.File(self.file, 'r') as f:
  #     self.data_len = len(f['labels'])

  # def __getitem__(self, index):
  #   with h5.File(self.file, 'r') as f:
  #     start, stop = f['timestamp'][index]
  #     labels = f['labels'][index]
  #     if stop - start > self.video_len:
  #       start_rand = np.random.randint(start, stop - self.video_len)
  #       video = f['videos'][index][start_rand: stop]
  #     else:
  #       video = f['videos'][index][start: stop]
  #       while video.shape[0] < self.video_len:
  #         video = np.vstack((video, video[-1]))
  #       # print('Added data for %s'%(video_name))
  #     if self.transform is not None:
  #       video = self.transform(video)
  #   return video, label

  # def __len__(self):
  #   return self.data_len
  # torchvision.datasets.UCF101(root, annotation_path, frames_per_clip, step_between_clips=1, fold=1, train=True, transform=None)

  def __init__(self, root, extensions=None, clip_length_in_frames=12, frames_between_clips=12, frame_rate=12, transforms = None):
    # print(root, clip_length_in_frames, frames_between_clips)
    if extensions == None:
      extensions = ('avi','mp4')
    classes = list(sorted(list_dir(root)))
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    self.samples = self.make_dataset(root, class_to_idx, extensions, is_valid_file=None)
    video_list = [x[0] for x in self.samples]

    self.video_clips = VideoClips(sorted(glob(root+'/**/*')), clip_length_in_frames, frames_between_clips,frame_rate=frame_rate,num_workers=8)
    self.transforms = transforms

  def make_dataset(self, dir, class_to_idx, extensions=None, is_valid_file=None):
    samples = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return x.lower().endswith(extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    samples.append(item)

    return samples

  def __getitem__(self, index):
    # index = 0
    clip, audio, info, video_idx = self.video_clips.get_clip(index)
    # print('NUM_CLIPS!!!: ', self.video_clips.num_clips(), 'NUM_VIDEOS: ', self.video_clips.num_videos())
    # print('VideoClips files: ', ' | '.join(self.video_clips.video_paths))
    # print('video_idx: ', video_idx, 'index: ', index)
    if self.transforms != None:
      clip = self.transforms(clip)
    label = self.samples[video_idx][1]

    return clip, label

  def __len__(self):

    return self.video_clips.num_clips()

  # def __init__():
    # self.video_dataset = data.dataset.UCF101(root, annotation_path, frames_per_clip=12, step_between_clips=10)
    # return self.video_dataset

def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tensor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True
#xiaodan: added by xiaodan
def resize(clip, target_size, interpolation_mode="area"):
    assert len(target_size) == 2, "target size should be tuple (height, width)"
    return torch.nn.functional.interpolate(
        clip, size=target_size, mode=interpolation_mode
    )
#xiaodan: added by xiaodan
class VideoResizedCenterCrop(object):
  """Crops the given video at the center.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
    """

  def __init__(self, size):
    if isinstance(size, numbers.Number):
      self.size = (int(size), int(size))
    else:
      self.size = size

  def __call__(self, clip):
    """
    Args:
        clip (tensor): Clip to be cropped. [C,T,H,W]
    Returns:
        clip (tensor): Cropped clip. [C,T,H,W]
      """
    # print(clip.dtype)
    clip = clip.float()
    h, w = clip.shape[-2:]
    th, tw = self.size
    min_shape, min_shape_i= min((v,i) for i,v in enumerate([h,w]))
    if min_shape_i == 0:
      target_size = (th,int(round(tw*w/h)))
    else:
      target_size = (int(round(th*h/w)),tw)

    resized_clip = resize(clip,target_size)
    # print('clip and resized clip sizes',clip.shape,resized_clip.shape)
    rh, rw = resized_clip.shape[-2:]
    i = int(round((rh - th) / 2.))
    j = int(round((rw - tw) / 2.))
    # print('i and j','(',i,',',i+th,')','(',j,',',j+tw,')')
    return resized_clip[..., i:(i + th), j:(j + tw)]


  def __repr__(self):
    return self.__class__.__name__ + '(size={0})'.format(self.size)

class VideoCenterCrop(object):
  """Crops the given video at the center.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
    """

  def __init__(self, size):
    if isinstance(size, numbers.Number):
      self.size = (int(size), int(size))
    else:
      self.size = size

  def __call__(self, clip):
    """
    Args:
        clip (tensor): Clip to be cropped. [C,T,H,W]
    Returns:
        clip (tensor): Cropped clip. [C,T,H,W]
      """
    h, w = clip.shape[-2:]
    th, tw = self.size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return clip[..., i:(i + th), j:(j + tw)]


  def __repr__(self):
    return self.__class__.__name__ + '(size={0})'.format(self.size)

class VideoNormalize(object):
  """docstring for VideoNormalize"""
  def __init__(self, mean, std, inplace=False):
    super(VideoNormalize, self).__init__()
    self.mean = mean
    self.std = std
    self.inplace = inplace


  def __call__(self, clip):
    """
    Args:
        clip (torch.tensor): video clip to be normalized. Size is (C, T, H, W)
    """
    return self.normalize(clip, self.mean, self.std, self.inplace)

  def __repr__(self):
    return self.__class__.__name__ + '(mean={0}, std={1}, inplace={2})'.format(
      self.mean, self.std, self.inplace)

  def normalize(self, clip, mean, std, inplace=False):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (C, T, H, W)
        mean (tuple): pixel RGB mean. Size is (3)
        std (tuple): pixel standard deviation. Size is (3)
    Returns:
        normalized clip (torch.tensor): Size is (C, T, H, W)
    """
    assert _is_tensor_video_clip(clip), "clip should be a 4D torch.tensor"
    if not inplace:
      clip = clip.clone()
    mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return clip.permute(1, 0, 2, 3)

class ToTensorVideo(object):
  """
  Convert tensor data type from uint8 to float, divide value by 255.0 and
  permute the dimenions of clip tensor
  """
  def __init__(self):
    pass

  def __call__(self, clip):
    """
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, H, W, C)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W)
    """
    _is_tensor_video_clip(clip)
    if not clip.dtype == torch.uint8:
      raise TypeError("clip tensor should have data type uint8. Got %s" % str(clip.dtype))
    return clip.float().permute(3, 0, 1, 2) / 255.0

  def __repr__(self):
    return self.__class__.__name__

class CIFAR100(CIFAR10):
    base_folder = 'cifar-100-python'
    url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
