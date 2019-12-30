# import zarr
import pandas as pd
import os
import random
import numpy as np 
# import dask as da
import matplotlib.pyplot as plt
from tqdm import tqdm

class vid2frame_dataset(data.dataset):
	"""docstring for video_dataset"""
	def __init__(self, data_root=None, save_path=None, label_csv_path=None, cache_csv_path, extensions=None, clip_length_in_frames=12, frame_rate=12, transforms = None):
		super(video_dataset, self).__init__()
		"""	
			The constructor for Rectangle class

		Parameters
	    ----------
		data_root : str(or None)
			The path to the directory with all the videos 
		save_path : str(or None)
			The path to the directory where the frames should be saved
        label_csv_path : str(or None)
        	The path to the csv file which contains class labels
        cache_csv_path : str(or None)
        	The path to the csv file where the cache will be saved
        extensions : list(or None)
        	The path to the csv file where the cache will be saved
		"""

		self.data_root = data_root
		# self.zarr_root = zarr_root
		self.label_csv_path = label_csv_path
		self.cache_csv_path = cache_csv_path
		self.save_path = save_path
		# self.zarr_file = zarr.open(zarr_root, 'a')
		self.extensions = extensions
		self.clip_length_in_frames = clip_length_in_frames
		self.frame_rate = frame_rate
		self.transforms = transforms
		self.frame_cache_exists = False

		if 'videos' in self.zarr_file.keys():
			self.zarr_file_exists = True
			self.videos = self.zarr_file['videos'].data
			self.labels = self.zarr_file['labels'].data
			self.cache_df = pd.from_csv(self.cache_csv_path)

		if self.zarr_file_exists == False:
			self.label_df = pd.from_csv(self.csv_path)
			columns = ['path', 'label']
			self.cache_df = pd.DataFrame(columns=columns)
			self.create_frame_cache()
			

	def __getitem__(index):
		frame_path = self.cache_df['path']
		label = self.cache_df['label']
		num_frames = len(os.listdir(frame_path))
		start_frame = random.randint(num_frames-self.clip_length_in_frames)
		clip = np.empty((0, **plt.imread(os.path.join(frame_path, str(start_frame) + '.jpg')).shape))
		for frame in range(start_frame, start_frame+self.clip_length_in_frames):
			frame = plt.imread(os.path.join(frame_path, str(frame) + '.jpg'))
			clip = np.concatenate(clip, frame)
		if self.transforms != None:
			clip = self.transforms(self.videos[index])
		return clip, label

	def  __len__():
		return len(self.cache_df)

	def vid2frame(self, file, frame_path):
		command = "ffmpeg  -loglevel panic -i {} -q:v 1 -vf fps={} {}/%06d.jpg".format(os.path.join(self.data_root, file), self.frame_rate, frame_path)
		try:
			os.system(command)
		except:
			return False
		return True

	def create_frame_cache(self):
		# self.zarr_file.create_froup('videos')
		# self.zarr_file.create_froup('labels')
		# num_clips = 0
		def is_video(file):
			if self.extensions == None:
				self.extensions = ['avi', 'mp4']
			for ext in self.extensions:
				if file.endswith(ext):
					return True
			return False

		for file in tqdm(os.listdir(self.data_root)):
			#create jpg frames from video
			if is_video(file):
				frame_path = os.path.join(self.save_path,file[:-4])
				os.makedirs(frame_path)
				files_written = self.vid2frame(file, frame_path)
				if not files_written:
					raise RuntimeError('Failed to convert file {}'.format(file))
					break
				label = str(self.label_df[self.label_df['youtube_id'] == file[:-4]]['label'])
				self.cache_df.loc[len(self.cache_df)] = [frame_path, label]
		#save cache to disk
		self.cache_df.to_csv(self.cache_csv_path)



			# save frames to zarr
			# num_frames = len(os.listdir(frame_path))
			# if num_frames >= self.clip_length_in_frames:
			# 	start_frame = random.randint(num_frames-self.clip_length_in_frames)
			# 	clip = np.empty((0, **plt.imread(os.path.join(frame_path, str(start_frame) + '.jpg')).shape))
			# 	for frame in range(start_frame, start_frame+self.clip_length_in_frames):
			# 		frame = plt.imread(os.path.join(frame_path, str(frame) + '.jpg'))
			# 		clip = np.concatenate(clip, frame)
			# 	clip = np.expand_dims(clip, axis=0)
			# 	label = np.array(str(self.label_df[self.label_df['youtube_id'] == file[:-4]]['label']))
			# 	label = np.expand_dims(label, axis=0)
			# 	if num_clips == 0:
			# 		videos = self.zarr_file['videos'].array('data', shape=clip.shape, chunks=((**clip.shape[1:])), data=clip, compressor=None)
			# 		labels = self.zarr_file['labels'].array('data', shape=label.shape, chunks=True, data=label, compressor=None)
			# 		# self.zarr_file = zarr.open(self.zarr_root, 'r+')
			# 	else:
			# 		videos.append(clip, axis=0)
			# 		labels.append(label, axis=0)
