import os
import pandas as pd

root = '/home/ubuntu/kinetics-400/kinetics'
video_path = os.path.join(root, 'Kinetics_trimmed_videos_train_merge')

df = pd.read_csv(os.path.join(root, 'csv', 'kinetics-400_train.csv'))
file_label = {}

for index, row in df.iterrows():
	file_label[row['youtube_id']] = row['label']
print('Map created!')

count = 0
for video in os.listdir(video_path):
	if video.endswith('.mp4'):
		label = file_label[video[:-4]]
		label = label.replace("'", "")
		label_path = os.path.join(video_path, label)
		if os.path.isdir(label_path) == False:
			print('Creating: ', label)
			os.makedirs(label_path)
		os.system("mv '{}/{}' '{}/{}'".format(video_path, video, label_path, video))
		count += 1
	if count % 1000 == 0:
		print('Completed {} videos'.format(count))