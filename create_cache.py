import os 
import pandas as pd 
import numpy as np 

frame_root = '/home/ubuntu/kinetics-400/kinetics/frames'
cache_csv = '/home/ubuntu/kinetics-400/kinetics/file_cache.csv'
class_csv = '/home/ubuntu/kinetics-400/kinetics/csv/kinetics-400_train.csv'


class_df = pd.read_csv(class_csv)

columns = ['path', 'label']
cache_df = pd.DataFrame(columns=columns)


count = 0
for video_dir in os.listdir(frame_root):
	label = str(class_df[class_df['youtube_id'] == video_dir]['label'].values[0]).split(',')[0]
	path = os.path.join(frame_root, video_dir)
	cache_df = cache_df.append({'path':path, 'label': label}, ignore_index=True)
	if count % 1000 == 0:
		print('completed: {}'.format(count))
	count += 1

cache_df.to_csv(cache_csv)