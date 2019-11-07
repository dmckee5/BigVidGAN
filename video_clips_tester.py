from collections import Counter
# from torchvision.datasets.video_utils import VideoClips
import os
from torchvision.datasets.utils import list_dir
from glob import glob
from VideoClips2 import VideoClips
import numpy as np
def tester(root, extensions=None, clip_length_in_frames=12, frames_between_clips=12, frame_rate=12, transforms = None):
    # print(root, clip_length_in_frames, frames_between_clips)
    def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
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

    if extensions == None:
      extensions = ('avi','mp4')
    classes = list(sorted(list_dir(root)))
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    idx_to_class = {i: classes[i] for i in range(len(classes))}
    samples = make_dataset(root, class_to_idx, extensions, is_valid_file=None)
    video_list = [x[0] for x in samples]
    video_idxs = []
    video_clips = VideoClips(sorted(glob(root+'/**/*')), clip_length_in_frames, frames_between_clips,frame_rate=12,num_workers=8)
    print('Num samples: ', len(samples), video_clips.num_videos())
    print('Num clips: ', video_clips.num_clips())
    smaller_set = set()
    # for sample in samples:
    # 	if sample[0] not in smaller_set:
    # 		smaller_set.add(sample[0])
    # count = 0
    # for video_path in video_clips.video_paths:
    #     if video_path in smaller_set:
    #         print(video_path)
    #     else:
    #         smaller_set.add(video_path)
    #     if count % 1000 == 0:
    #         print(count)
    #     count += 1
    diff = 0
    idx = 0
    previous_video_idx=0
    for i in range(video_clips.num_clips()):
        clip, audio, info, video_idx = video_clips.get_clip(i)
        video_idxs.append(video_idx)
        if i % 500 == 0:
            print(i)
        if previous_video_idx+1 != video_idx:
            print(i,video_idx, video_clips.video_paths[video_idx])
            print('\n')
            diff += 1
        idx += 1
        previous_video_idx = video_idx

    # counts = Counter(video_clips.video_paths)
    # print('Finding duplicates')
    # for key, val in counts.items():
    # 	if val > 1:
    		# print(key)
    np.save('video_idxs.npy',np.array(video_idxs))

root = '/home/ubuntu/nfs/data/UCF-101_copy2'
files = sorted(glob(root+'/**/*'))
extensions = set()
for file in files:
	if file[-3:] not in extensions:
		extensions.add(file[-3:])
print(extensions)
tester(root, None, 12, 1000000)
