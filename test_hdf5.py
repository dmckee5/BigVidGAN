import h5py as h5
with h5.File('../../data/UCF101.hdf5', 'r') as f:
    print('I can print')
