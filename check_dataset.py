import os
from skvideo import io
from multiprocessing import Pool

root = '/home/ubuntu/nfs/data/kinetics-400/train/Kinetics_trimmed_videos_train_merge'
count = 0

label_path = []
for label in os.listdir(root):
	label_path.append(os.path.join(root, label))

def file_check(label_path):
	for video in os.listdir(label_path):
		try:
			vid = io.vread(os.path.join(label_path, video))
		except:
			print("video: {} in class: {} doesn't open".format(video, label_path))
	print('Finished {}'.format(label_path))
	# return "Checked class: {}".format(label_path)

p = Pool(10)
print(p.map(file_check, [label for label in label_path[178:]]))