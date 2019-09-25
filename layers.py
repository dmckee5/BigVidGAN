''' Layers
    This file contains various layers for the BigGAN models.
'''
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P

from sync_batchnorm import SynchronizedBatchNorm2d as SyncBN2d


# Projection of x onto y
def proj(x, y):
  return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
  for y in ys:
    x = x - proj(x, y)
  return x


# Apply num_itrs steps of the power method to estimate top N singular values.
def power_iteration(W, u_, update=True, eps=1e-12):
  # Lists holding singular vectors and values
  us, vs, svs = [], [], []
  for i, u in enumerate(u_):
    # Run one step of the power iteration
    with torch.no_grad():
      v = torch.matmul(u, W)
      # Run Gram-Schmidt to subtract components of all other singular vectors
      v = F.normalize(gram_schmidt(v, vs), eps=eps)
      # Add to the list
      vs += [v]
      # Update the other singular vector
      u = torch.matmul(v, W.t())
      # Run Gram-Schmidt to subtract components of all other singular vectors
      u = F.normalize(gram_schmidt(u, us), eps=eps)
      # Add to the list
      us += [u]
      if update:
        u_[i][:] = u
    # Compute this singular value and add it to the list
    svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    #svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
  return svs, us, vs


# Convenience passthrough function
class identity(nn.Module):
  def forward(self, input):
    return input


# Spectral normalization base class
class SN(object):
  def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
    # Number of power iterations per step
    self.num_itrs = num_itrs
    # Number of singular values
    self.num_svs = num_svs
    # Transposed?
    self.transpose = transpose
    # Epsilon value for avoiding divide-by-0
    self.eps = eps
    # Register a singular vector for each sv
    for i in range(self.num_svs):
      self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
      self.register_buffer('sv%d' % i, torch.ones(1))

  # Singular vectors (u side)
  @property
  def u(self):
    return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

  # Singular values;
  # note that these buffers are just for logging and are not used in training.
  @property
  def sv(self):
   return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]

  # Compute the spectrally-normalized weight
  def W_(self):
    W_mat = self.weight.view(self.weight.size(0), -1)
    if self.transpose:
      W_mat = W_mat.t()
    # Apply num_itrs power iterations
    for _ in range(self.num_itrs):
      svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps)
    # Update the svs
    if self.training:
      with torch.no_grad(): # Make sure to do this in a no_grad() context or you'll get memory leaks!
        for i, sv in enumerate(svs):
          self.sv[i][:] = sv
    return self.weight / svs[0]


# 2D Conv layer with spectral norm
class SNConv2d(nn.Conv2d, SN):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
             padding=0, dilation=1, groups=1, bias=True,
             num_svs=1, num_itrs=1, eps=1e-12):
    nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
                     padding, dilation, groups, bias)
    SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)
  def forward(self, x):
    return F.conv2d(x, self.W_(), self.bias, self.stride,
                    self.padding, self.dilation, self.groups)

# 3D Conv layer with spectral norm
class SNConv3d(nn.Conv3d, SN):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
             padding=0, dilation=1, groups=1, bias=True,
             num_svs=1, num_itrs=1, eps=1e-12):
    nn.Conv3d.__init__(self, in_channels, out_channels, kernel_size, stride,
                     padding, dilation, groups, bias)
    SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)
  def forward(self, x):
    return F.conv3d(x, self.W_(), self.bias, self.stride,
                    self.padding, self.dilation, self.groups)

# Linear layer with spectral norm
class SNLinear(nn.Linear, SN):
  def __init__(self, in_features, out_features, bias=True,
               num_svs=1, num_itrs=1, eps=1e-12):
    nn.Linear.__init__(self, in_features, out_features, bias)
    SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)
  def forward(self, x):
    return F.linear(x, self.W_(), self.bias)


# Embedding layer with spectral norm
# We use num_embeddings as the dim instead of embedding_dim here
# for convenience sake
class SNEmbedding(nn.Embedding, SN):
  def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
               max_norm=None, norm_type=2, scale_grad_by_freq=False,
               sparse=False, _weight=None,
               num_svs=1, num_itrs=1, eps=1e-12):
    nn.Embedding.__init__(self, num_embeddings, embedding_dim, padding_idx,
                          max_norm, norm_type, scale_grad_by_freq,
                          sparse, _weight)
    SN.__init__(self, num_svs, num_itrs, num_embeddings, eps=eps)
  def forward(self, x):
    return F.embedding(x, self.W_())


class FullAttention(nn.Module):
  def __init__(self, ch, time_steps, which_conv=SNConv2d, name='full_attention'):
    super(FullAttention, self).__init__()  #assume input tensor as (BT, C, W, H)
    # Channel multiplier
    self.ch = ch
    self.T = time_steps
    self.which_conv = which_conv
    self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
    self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
    # Learnable gain parameter
    self.gamma = P(torch.tensor(0.), requires_grad=True)
  def forward(self, x, y=None):
    # Apply convs
    theta = self.theta(x)
    phi = self.phi(x).contiguous().view(-1, self.T, self.ch//8, x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)
    g = self.g(x).contiguous().view(-1, self.T, self.ch//2, x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)
    # Perform reshapes
    theta = theta.contiguous().view(-1, self.T, self.ch//8, x.shape[2], x.shape[3])
    theta.permute(0, 2, 1, 3, 4)
    theta = theta.contiguous().view(-1, self.ch//8, self.T*x.shape[2]*x.shape[3])
    phi = phi.contiguous().view(-1, self.ch//8, self.T*x.shape[2]*x.shape[3])
    g = g.contiguous().view(-1, self.ch//2, self.T*x.shape[2]*x.shape[3])
    # Matmul and softmax to get attention maps
    beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
    # Attention map times g path
    o = self.o(torch.bmm(g, beta.transpose(1,2)).contiguous().view(-1, self.ch // 2, self.T, x.shape[2], x.shape[3]))
    return self.gamma * o + x

# A non-local block as used in SA-GAN
# Note that the implementation as described in the paper is largely incorrect;
# refer to the released code for the actual implementation.
class Attention(nn.Module):
  def __init__(self, ch, which_conv=SNConv2d, name='attention'):
    super(Attention, self).__init__()
    # Channel multiplier
    self.ch = ch
    self.which_conv = which_conv
    self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
    self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
    # Learnable gain parameter
    self.gamma = P(torch.tensor(0.), requires_grad=True)
  def forward(self, x, y=None):
    # Apply convs
    theta = self.theta(x)
    phi = F.max_pool2d(self.phi(x), [2,2])
    g = F.max_pool2d(self.g(x), [2,2])
    # Perform reshapes
    theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
    phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
    g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
    # Matmul and softmax to get attention maps
    beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
    # Attention map times g path
    o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
    return self.gamma * o + x

class SelfAttention_width(nn.Module):# Assume [B,T,C,H,W]
  def __init__(self, ch, time_steps,which_conv=SNConv2d, name='attention_width'):
    super(Attention, self).__init__()
    # Channel multiplier
    self.ch = ch
    self.T = time_steps
    self.which_conv = which_conv
    self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
    self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
    # Learnable gain parameter
    self.gamma = P(torch.tensor(0.), requires_grad=True)
  def forward(self, x, y=None):
    # x:[BT,C,H,W]
    # Apply convs
    theta = self.theta(x)
    phi = self.phi(x)
    g = self.g(x)
    # Perform reshapes
    theta = theta.contiguous().view(-1,self.T,*x.shape[1:])
    theta = theta.permute(0,1,3,2,4)
    theta = theta.contiguous().view(-1, self. ch // 8, x.shape[3]) #[BTH,C//8,W]
    phi = phi.contiguous().view(-1,self.T,*x.shape[1:])
    phi = phi.permute(0,1,3,2,4)
    phi = phi.contiguous().view(-1, self. ch // 8,x.shape[3]) # [BTH,C//8,W]
    g = g.contiguous().view(-1,self.T,*x.shape[1:])
    g = g.permute(0,1,3,2,4)
    g = g.contiguous().view(-1, self. ch // 2, x.shape[3]) # [BTH,C//2,W]
    # Matmul and softmax to get attention maps
    beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
    # Attention map times g path
    attn_g = torch.bmm(g, beta.transpose(1,2)).contiguous().view(-1, self.T, x.shape[2],self.ch // 2, x.shape[3]).permute(0,1,3,2,4)
    o = self.o(attn_g.contiguous().view(*x.shape))
    return self.gamma * o + x

class SelfAttention_height(nn.Module):# Assume [B,T,C,H,W]
  def __init__(self, ch, time_steps,which_conv=SNConv2d, name='attention_height'):
    super(Attention, self).__init__()
    # Channel multiplier
    self.ch = ch
    self.T = time_steps
    self.which_conv = which_conv
    self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
    self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
    # Learnable gain parameter
    self.gamma = P(torch.tensor(0.), requires_grad=True)
  def forward(self, x, y=None):
    # x:[BT,C,H,W]
    # Apply convs
    theta = self.theta(x)
    phi = self.phi(x)
    g = self.g(x)
    # Perform reshapes
    theta = theta.contiguous().contiguous().view(-1,self.T,*x.shape[1:]) #[B,T,C,H,W]
    theta = theta.permute(0,1,4,2,3) #[B,T,W,C,H]
    theta = theta.contiguous().contiguous().view(-1, self. ch // 8, x.shape[2]) #[BTW,C//8,H]
    phi = phi.contiguous().contiguous().view(-1,self.T,*x.shape[1:])#[B,T,C,H,W]
    phi = phi.permute(0,1,4,2,3) #[B,T,W,C,H]
    phi = phi.contiguous().contiguous().view(-1, self. ch // 8,x.shape[2]) #[BTW,C//8,H]
    g = g.contiguous().contiguous().view(-1,self.T,*x.shape[1:])#[B,T,C,H,W]
    g = g.permute(0,1,4,2,3)#[B,T,W,C,H]
    g = g.contiguous().contiguous().view(-1, self. ch // 2, x.shape[2]) # [BTW,C//2,H]
    # Matmul and softmax to get attention maps
    beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)#[BTW,H,|H|]
    # Attention map times g path
    attn_g = torch.bmm(g, beta.transpose(1,2)).contiguous().contiguous().view(-1, self.T, x.shape[3],self.ch // 2, x.shape[2]).permute(0,1,3,4,2) #[B,T,C,H,W]
    o = self.o(attn_g.contiguous().contiguous().view(*x.shape))# [BT,C,H,W]
    return self.gamma * o + x

class SelfAttention_time(nn.Module):# Assume [B,T,C,H,W]
  def __init__(self, ch, time_steps,which_conv=SNConv2d, name='attention_time'):
    super(Attention, self).__init__()
    # Channel multiplier
    self.ch = ch
    self.T = time_steps
    self.which_conv = which_conv
    self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
    self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
    # Learnable gain parameter
    self.gamma = P(torch.tensor(0.), requires_grad=True)
  def forward(self, x, y=None):
    # x:[BT,C,H,W]
    # Apply convs
    theta = self.theta(x) #[BT,C//8,H,W]
    phi = self.phi(x) #[BT,C//8,H,W]
    g = self.g(x) #[BT,C//2,H,W]
    # Perform reshapes
    theta = theta.contiguous().view(-1,self.T,*x.shape[1:]) #[B,T,C//8,H,W]
    theta = theta.permute(0,3,4,2,1) #[B,H,W,C//8,T]
    theta = theta.contiguous().view(-1, self. ch // 8, self.T) #[BHW,C//8,T]
    phi = phi.contiguous().view(-1,self.T,*x.shape[1:])#[B,T,C//8,H,W]
    phi = phi.permute(0,3,4,2,1) #[B,H,W,C//8,T]
    phi = phi.contiguous().view(-1, self. ch // 8,self.T) #[BHW,C//8,T]
    g = g.contiguous().view(-1,self.T,*x.shape[1:])#[B,T,C,H,W]
    g = g.permute(0,3,4,2,1)#[B,H,W,C//2,T]
    g = g.contiguous().view(-1, self. ch // 2, self.T) # [BHW,C//2,T]
    # Matmul and softmax to get attention maps
    beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)#[BHW,T,|T|]
    # Attention map times g path
    attn_g = torch.bmm(g, beta.transpose(1,2)).contiguous().view(-1, x.shape[2], x.shape[3],self.ch // 2, self.T).permute(0,4,3,1,2) #[B,T,C,H,W]
    o = self.o(attn_g.contiguous().view(*x.shape))# [BT,C,H,W]
    return self.gamma * o + x

# Fused batchnorm op
def fused_bn(x, mean, var, gain=None, bias=None, eps=1e-5):
  # Apply scale and shift--if gain and bias are provided, fuse them here
  # Prepare scale
  scale = torch.rsqrt(var + eps)
  # If a gain is provided, use it
  if gain is not None:
    scale = scale * gain
  # Prepare shift
  shift = mean * scale
  # If bias is provided, use it
  if bias is not None:
    shift = shift - bias
  return x * scale - shift
  #return ((x - mean) / ((var + eps) ** 0.5)) * gain + bias # The unfused way.


# Manual BN
# Calculate means and variances using mean-of-squares minus mean-squared
def manual_bn(x, gain=None, bias=None, return_mean_var=False, eps=1e-5):
  # Cast x to float32 if necessary
  float_x = x.float()
  # Calculate expected value of x (m) and expected value of x**2 (m2)
  # Mean of x
  m = torch.mean(float_x, [0, 2, 3], keepdim=True)
  # Mean of x squared
  m2 = torch.mean(float_x ** 2, [0, 2, 3], keepdim=True)
  # Calculate variance as mean of squared minus mean squared.
  var = (m2 - m **2)
  # Cast back to float 16 if necessary
  var = var.type(x.type())
  m = m.type(x.type())
  # Return mean and variance for updating stored mean/var if requested
  if return_mean_var:
    return fused_bn(x, m, var, gain, bias, eps), m.squeeze(), var.squeeze()
  else:
    return fused_bn(x, m, var, gain, bias, eps)


# My batchnorm, supports standing stats
class myBN(nn.Module):
  def __init__(self, num_channels, eps=1e-5, momentum=0.1):
    super(myBN, self).__init__()
    # momentum for updating running stats
    self.momentum = momentum
    # epsilon to avoid dividing by 0
    self.eps = eps
    # Momentum
    self.momentum = momentum
    # Register buffers
    self.register_buffer('stored_mean', torch.zeros(num_channels))
    self.register_buffer('stored_var',  torch.ones(num_channels))
    self.register_buffer('accumulation_counter', torch.zeros(1))
    # Accumulate running means and vars
    self.accumulate_standing = False

  # reset standing stats
  def reset_stats(self):
    self.stored_mean[:] = 0
    self.stored_var[:] = 0
    self.accumulation_counter[:] = 0

  def forward(self, x, gain, bias):
    if self.training:
      out, mean, var = manual_bn(x, gain, bias, return_mean_var=True, eps=self.eps)
      # If accumulating standing stats, increment them
      if self.accumulate_standing:
        self.stored_mean[:] = self.stored_mean + mean.data
        self.stored_var[:] = self.stored_var + var.data
        self.accumulation_counter += 1.0
      # If not accumulating standing stats, take running averages
      else:
        self.stored_mean[:] = self.stored_mean * (1 - self.momentum) + mean * self.momentum
        self.stored_var[:] = self.stored_var * (1 - self.momentum) + var * self.momentum
      return out
    # If not in training mode, use the stored statistics
    else:
      mean = self.stored_mean.view(1, -1, 1, 1)
      var = self.stored_var.view(1, -1, 1, 1)
      # If using standing stats, divide them by the accumulation counter
      if self.accumulate_standing:
        mean = mean / self.accumulation_counter
        var = var / self.accumulation_counter
      return fused_bn(x, mean, var, gain, bias, self.eps)


# Simple function to handle groupnorm norm stylization
def groupnorm(x, norm_style):
  # If number of channels specified in norm_style:
  if 'ch' in norm_style:
    ch = int(norm_style.split('_')[-1])
    groups = max(int(x.shape[1]) // ch, 1)
  # If number of groups specified in norm style
  elif 'grp' in norm_style:
    groups = int(norm_style.split('_')[-1])
  # If neither, default to groups = 16
  else:
    groups = 16
  return F.group_norm(x, groups)


# Class-conditional bn
# output size is the number of channels, input size is for the linear layers
# Andy's Note: this class feels messy but I'm not really sure how to clean it up
# Suggestions welcome! (By which I mean, refactor this and make a pull request
# if you want to make this more readable/usable).
class ccbn(nn.Module):
  def __init__(self, output_size, input_size, which_linear, eps=1e-5, momentum=0.1,
               cross_replica=False, mybn=False, norm_style='bn',):
    super(ccbn, self).__init__()
    self.output_size, self.input_size = output_size, input_size
    # Prepare gain and bias layers
    self.gain = which_linear(input_size, output_size)
    self.bias = which_linear(input_size, output_size)
    # epsilon to avoid dividing by 0
    self.eps = eps
    # Momentum
    self.momentum = momentum
    # Use cross-replica batchnorm?
    self.cross_replica = cross_replica
    # Use my batchnorm?
    self.mybn = mybn
    # Norm style?
    self.norm_style = norm_style

    if self.cross_replica:
      self.bn = SyncBN2d(output_size, eps=self.eps, momentum=self.momentum, affine=False)
    elif self.mybn:
      self.bn = myBN(output_size, self.eps, self.momentum)
    elif self.norm_style in ['bn', 'in']:
      self.register_buffer('stored_mean', torch.zeros(output_size))
      self.register_buffer('stored_var',  torch.ones(output_size))


  def forward(self, x, y):
    # Calculate class-conditional gains and biases
    gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
    bias = self.bias(y).view(y.size(0), -1, 1, 1)
    # If using my batchnorm
    if self.mybn or self.cross_replica:
      return self.bn(x, gain=gain, bias=bias)
    # else:
    else:
      if self.norm_style == 'bn':
        out = F.batch_norm(x, self.stored_mean, self.stored_var, None, None,
                          self.training, 0.1, self.eps)
      elif self.norm_style == 'in':
        out = F.instance_norm(x, self.stored_mean, self.stored_var, None, None,
                          self.training, 0.1, self.eps)
      elif self.norm_style == 'gn':
        out = groupnorm(x, self.normstyle)
      elif self.norm_style == 'nonorm':
        out = x
      return out * gain + bias
  def extra_repr(self):
    s = 'out: {output_size}, in: {input_size},'
    s +=' cross_replica={cross_replica}'
    return s.format(**self.__dict__)

# Class-conditional bn for 5D tensor
# output size is the number of channels, input size is for the linear layers
# xiaodan: added by Xiaodan
class ccbn5D(nn.Module):
  def __init__(self, output_size, input_size, which_linear, time_steps, eps=1e-5, momentum=0.1,
               cross_replica=False, mybn=False, norm_style='bn',):
    super(ccbn, self).__init__()
    self.output_size, self.input_size = output_size, input_size
    # Prepare gain and bias layers
    self.gain = which_linear(input_size, output_size)
    self.bias = which_linear(input_size, output_size)
    self.time_steps = time_steps #xiaodan: added by xiaodan
    # epsilon to avoid dividing by 0
    self.eps = eps
    # Momentum
    self.momentum = momentum
    # Use cross-replica batchnorm?
    self.cross_replica = cross_replica
    # Use my batchnorm?
    self.mybn = mybn
    # Norm style?
    self.norm_style = norm_style

    if self.cross_replica:
      self.bn = SyncBN2d(output_size, eps=self.eps, momentum=self.momentum, affine=False)
    elif self.mybn:
      self.bn = myBN(output_size, self.eps, self.momentum)
    elif self.norm_style in ['bn', 'in']:
      self.register_buffer('stored_mean', torch.zeros(output_size))
      self.register_buffer('stored_var',  torch.ones(output_size))


  def forward(self, x, y): #x:[BT,C,H,W]
    # xiaodan: make x back to [B,T,C,H,W] and reshape into [B,TC,H,W]
    x_TC = x.contiguous().view(-1,self.time_steps,*x.shape[1:]).contiguous()\
                                            .view(x.shape[0],-1,*x.shape[2:])
    # Calculate class-conditional gains and biases
    gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
    bias = self.bias(y).view(y.size(0), -1, 1, 1)
    # If using my batchnorm
    if self.mybn or self.cross_replica:
      return self.bn(x_TC, gain=gain, bias=bias).contiguous()\ # [B,TC,H,W]
                .view(-1,self.time_steps,*x.shape[1:])\ # [B,T,C,H,W]
                .contiguous().view(*x.shape)  # [BT,C,H,W]
    # else:
    else:
      if self.norm_style == 'bn':
        out = F.batch_norm(x_TC, self.stored_mean, self.stored_var, None, None,
                          self.training, 0.1, self.eps)
      elif self.norm_style == 'in':
        out = F.instance_norm(x_TC, self.stored_mean, self.stored_var, None, None,
                          self.training, 0.1, self.eps)
      elif self.norm_style == 'gn':
        out = groupnorm(x_TC, self.normstyle)
      elif self.norm_style == 'nonorm':
        out = x_TC
      return out.contiguous()\ # [B,TC,H,W]
                .view(-1,self.time_steps,*x.shape[1:])\ # [B,T,C,H,W]
                .contiguous().view(*x.shape) * gain + bias\  # [BT,C,H,W]
  def extra_repr(self):
    s = 'out: {output_size}, in: {input_size},'
    s +=' cross_replica={cross_replica}'
    return s.format(**self.__dict__)


# Normal, non-class-conditional BN
class bn(nn.Module):
  def __init__(self, output_size,  eps=1e-5, momentum=0.1,
                cross_replica=False, mybn=False):
    super(bn, self).__init__()
    self.output_size= output_size
    # Prepare gain and bias layers
    self.gain = P(torch.ones(output_size), requires_grad=True)
    self.bias = P(torch.zeros(output_size), requires_grad=True)
    # epsilon to avoid dividing by 0
    self.eps = eps
    # Momentum
    self.momentum = momentum
    # Use cross-replica batchnorm?
    self.cross_replica = cross_replica
    # Use my batchnorm?
    self.mybn = mybn

    if self.cross_replica:
      self.bn = SyncBN2d(output_size, eps=self.eps, momentum=self.momentum, affine=False)
    elif mybn:
      self.bn = myBN(output_size, self.eps, self.momentum)
     # Register buffers if neither of the above
    else:
      self.register_buffer('stored_mean', torch.zeros(output_size))
      self.register_buffer('stored_var',  torch.ones(output_size))

  def forward(self, x, y=None):
    if self.cross_replica or self.mybn:
      gain = self.gain.view(1,-1,1,1)
      bias = self.bias.view(1,-1,1,1)
      return self.bn(x, gain=gain, bias=bias)
    else:
      return F.batch_norm(x, self.stored_mean, self.stored_var, self.gain,
                          self.bias, self.training, self.momentum, self.eps)


# Generator blocks
# Note that this class assumes the kernel size and padding (and any other
# settings) have been selected in the main generator module and passed in
# through the which_conv arg. Similar rules apply with which_bn (the input
# size [which is actually the number of channels of the conditional info] must
# be preselected)
class GBlock(nn.Module):
  def __init__(self, in_channels, out_channels,
               which_conv=nn.Conv2d, which_bn=bn, activation=None,
               upsample=None):
    super(GBlock, self).__init__()

    self.in_channels, self.out_channels = in_channels, out_channels
    self.which_conv, self.which_bn = which_conv, which_bn
    self.activation = activation
    self.upsample = upsample
    # Conv layers
    self.conv1 = self.which_conv(self.in_channels, self.out_channels)
    self.conv2 = self.which_conv(self.out_channels, self.out_channels)
    self.learnable_sc = in_channels != out_channels or upsample
    if self.learnable_sc:
      self.conv_sc = self.which_conv(in_channels, out_channels,
                                     kernel_size=1, padding=0)
    # Batchnorm layers
    self.bn1 = self.which_bn(in_channels)
    self.bn2 = self.which_bn(out_channels)
    # upsample layers
    self.upsample = upsample

  def forward(self, x, y):
    h = self.activation(self.bn1(x, y))
    if self.upsample:
      h = self.upsample(h)
      x = self.upsample(x)
    h = self.conv1(h)
    h = self.activation(self.bn2(h, y))
    h = self.conv2(h)
    if self.learnable_sc:
      x = self.conv_sc(x)
    return h + x


# Residual block for the discriminator
class DBlock(nn.Module):
  def __init__(self, in_channels, out_channels, which_conv=SNConv2d, wide=True,
               preactivation=False, activation=None, downsample=None,):
    super(DBlock, self).__init__()
    self.in_channels, self.out_channels = in_channels, out_channels
    # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
    self.hidden_channels = self.out_channels if wide else self.in_channels
    self.which_conv = which_conv
    self.preactivation = preactivation
    self.activation = activation
    self.downsample = downsample

    # Conv layers
    self.conv1 = self.which_conv(self.in_channels, self.hidden_channels)
    self.conv2 = self.which_conv(self.hidden_channels, self.out_channels)
    self.learnable_sc = True if (in_channels != out_channels) or downsample else False
    if self.learnable_sc:
      self.conv_sc = self.which_conv(in_channels, out_channels,
                                     kernel_size=1, padding=0)
  def shortcut(self, x):
    if self.preactivation:
      if self.learnable_sc:
        x = self.conv_sc(x)
      if self.downsample:
        x = self.downsample(x)
    else:
      if self.downsample:
        x = self.downsample(x)
      if self.learnable_sc:
        x = self.conv_sc(x)
    return x

  def forward(self, x):
    if self.preactivation:
      # h = self.activation(x) # NOT TODAY SATAN
      # Andy's note: This line *must* be an out-of-place ReLU or it
      #              will negatively affect the shortcut connection.
      h = F.relu(x)
    else:
      h = x
    h = self.conv1(h)
    h = self.conv2(self.activation(h))
    if self.downsample:
      h = self.downsample(h)

    return h + self.shortcut(x)

# dogball
