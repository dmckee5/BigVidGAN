
from torchvision.datasets.video_utils import VideoClips

class MyVideoDataset(object):
    def __init__(self, video_paths):
        self.video_clips = VideoClips(video_paths,
                                      clip_length_in_frames=16,
                                      frames_between_clips=1,
                                      frame_rate=15)

    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        return video, audio
    
    def __len__(self):
        return self.video_clips.num_clips()

a = MyVideoDataset('/home/nfs/data/UCF-101_copy2')
