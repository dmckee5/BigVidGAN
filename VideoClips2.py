
import bisect
from fractions import Fraction
import math
import torch
# from torchvision.io import _read_video_from_file,_probe_video_from_file
from torchvision.io import read_video_timestamps, read_video

# from .utils import tqdm
from torch.utils.model_zoo import tqdm

default_timebase = Fraction(0, 1)
def _read_video_from_file(
    filename,
    seek_frame_margin=0.25,
    read_video_stream=True,
    video_width=0,
    video_height=0,
    video_min_dimension=0,
    video_pts_range=(0, -1),
    video_timebase=default_timebase,
    read_audio_stream=True,
    audio_samples=0,
    audio_channels=0,
    audio_pts_range=(0, -1),
    audio_timebase=default_timebase,
):
    """
    Reads a video from a file, returning both the video frames as well as
    the audio frames
    Args
    ----------
    filename : str
        path to the video file
    seek_frame_margin: double, optional
        seeking frame in the stream is imprecise. Thus, when video_start_pts is specified,
        we seek the pts earlier by seek_frame_margin seconds
    read_video_stream: int, optional
        whether read video stream. If yes, set to 1. Otherwise, 0
    video_width/video_height/video_min_dimension: int
        together decide the size of decoded frames
        - when video_width = 0, video_height = 0, and video_min_dimension = 0, keep the orignal frame resolution
        - when video_width = 0, video_height = 0, and video_min_dimension != 0, keep the aspect ratio and resize
            the frame so that shorter edge size is video_min_dimension
        - When video_width = 0, and video_height != 0, keep the aspect ratio and resize the frame
            so that frame video_height is $video_height
        - When video_width != 0, and video_height == 0, keep the aspect ratio and resize the frame
            so that frame video_height is $video_width
        - When video_width != 0, and video_height != 0, resize the frame so that frame video_width and video_height
            are set to $video_width and $video_height, respectively
    video_pts_range : list(int), optional
        the start and end presentation timestamp of video stream
    video_timebase: Fraction, optional
        a Fraction rational number which denotes timebase in video stream
    read_audio_stream: int, optional
        whether read audio stream. If yes, set to 1. Otherwise, 0
    audio_samples: int, optional
        audio sampling rate
    audio_channels: int optional
        audio channels
    audio_pts_range : list(int), optional
        the start and end presentation timestamp of audio stream
    audio_timebase: Fraction, optional
        a Fraction rational number which denotes time base in audio stream
    Returns
    -------
    vframes : Tensor[T, H, W, C]
        the `T` video frames
    aframes : Tensor[L, K]
        the audio frames, where `L` is the number of points and
            `K` is the number of audio_channels
    info : Dict
        metadata for the video and audio. Can contain the fields video_fps (float)
        and audio_fps (int)
    """
    _validate_pts(video_pts_range)
    _validate_pts(audio_pts_range)

    result = torch.ops.video_reader.read_video_from_file(
        filename,
        seek_frame_margin,
        0,  # getPtsOnly
        read_video_stream,
        video_width,
        video_height,
        video_min_dimension,
        video_pts_range[0],
        video_pts_range[1],
        video_timebase.numerator,
        video_timebase.denominator,
        read_audio_stream,
        audio_samples,
        audio_channels,
        audio_pts_range[0],
        audio_pts_range[1],
        audio_timebase.numerator,
        audio_timebase.denominator,
    )
    vframes, _vframe_pts, vtimebase, vfps, vduration, aframes, aframe_pts, atimebase, \
        asample_rate, aduration = result
    info = _fill_info(vtimebase, vfps, vduration, atimebase, asample_rate, aduration)
    if aframes.numel() > 0:
        # when audio stream is found
        aframes = _align_audio_frames(aframes, aframe_pts, audio_pts_range)
    return vframes, aframes, info


def _read_video_timestamps_from_file(filename):
    """
    Decode all video- and audio frames in the video. Only pts
    (presentation timestamp) is returned. The actual frame pixel data is not
    copied. Thus, it is much faster than read_video(...)
    """
    result = torch.ops.video_reader.read_video_from_file(
        filename,
        0,  # seek_frame_margin
        1,  # getPtsOnly
        1,  # read_video_stream
        0,  # video_width
        0,  # video_height
        0,  # video_min_dimension
        0,  # video_start_pts
        -1,  # video_end_pts
        0,  # video_timebase_num
        1,  # video_timebase_den
        1,  # read_audio_stream
        0,  # audio_samples
        0,  # audio_channels
        0,  # audio_start_pts
        -1,  # audio_end_pts
        0,  # audio_timebase_num
        1,  # audio_timebase_den
    )
    _vframes, vframe_pts, vtimebase, vfps, vduration, _aframes, aframe_pts, atimebase, \
        asample_rate, aduration = result
    info = _fill_info(vtimebase, vfps, vduration, atimebase, asample_rate, aduration)

    vframe_pts = vframe_pts.numpy().tolist()
    aframe_pts = aframe_pts.numpy().tolist()
    return vframe_pts, aframe_pts, info


def _probe_video_from_file(filename):
    """
    Probe a video file.
    Return:
        info [dict]: contain video meta information, including video_timebase,
            video_duration, video_fps, audio_timebase, audio_duration, audio_sample_rate
    """
    result = torch.ops.video_reader.probe_video_from_file(filename)
    vtimebase, vfps, vduration, atimebase, asample_rate, aduration = result
    info = _fill_info(vtimebase, vfps, vduration, atimebase, asample_rate, aduration)
    return info
def unfold(tensor, size, step, dilation=1):
    """
    similar to tensor.unfold, but with the dilation
    and specialized for 1d tensors
    Returns all consecutive windows of `size` elements, with
    `step` between windows. The distance between each element
    in a window is given by `dilation`.
    """
    assert tensor.dim() == 1
    o_stride = tensor.stride(0)
    numel = tensor.numel()
    new_stride = (step * o_stride, dilation * o_stride)
    new_size = ((numel - (dilation * (size - 1) + 1)) // step + 1, size)
    if new_size[0] < 1:
        new_size = (0, size)
    return torch.as_strided(tensor, new_size, new_stride)
class VideoClips(object):
    """
    Given a list of video files, computes all consecutive subvideos of size
    `clip_length_in_frames`, where the distance between each subvideo in the
    same video is defined by `frames_between_clips`.
    If `frame_rate` is specified, it will also resample all the videos to have
    the same frame rate, and the clips will refer to this frame rate.
    Creating this instance the first time is time-consuming, as it needs to
    decode all the videos in `video_paths`. It is recommended that you
    cache the results after instantiation of the class.
    Recreating the clips for different clip lengths is fast, and can be done
    with the `compute_clips` method.
    Arguments:
        video_paths (List[str]): paths to the video files
        clip_length_in_frames (int): size of a clip in number of frames
        frames_between_clips (int): step (in frames) between each clip
        frame_rate (int, optional): if specified, it will resample the video
            so that it has `frame_rate`, and then the clips will be defined
            on the resampled video
        num_workers (int): how many subprocesses to use for data loading.
            0 means that the data will be loaded in the main process. (default: 0)
    """
    def __init__(self, video_paths, clip_length_in_frames=16, frames_between_clips=1,
                 frame_rate=None, _precomputed_metadata=None, num_workers=0,
                 _video_width=0, _video_height=0, _video_min_dimension=0,
                 _audio_samples=0):

        self.video_paths = video_paths
        self.num_workers = num_workers

        # these options are not valid for pyav backend
        self._video_width = _video_width
        self._video_height = _video_height
        self._video_min_dimension = _video_min_dimension
        self._audio_samples = _audio_samples

        if _precomputed_metadata is None:
            self._compute_frame_pts()
        else:
            self._init_from_metadata(_precomputed_metadata)
        self.compute_clips(clip_length_in_frames, frames_between_clips, frame_rate)

    def _compute_frame_pts(self):
        self.video_pts = []
        self.video_fps = []

        # strategy: use a DataLoader to parallelize read_video_timestamps
        # so need to create a dummy dataset first
        class DS(object):
            def __init__(self, x):
                self.x = x

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                return read_video_timestamps(self.x[idx])

        import torch.utils.data
        dl = torch.utils.data.DataLoader(
            DS(self.video_paths),
            batch_size=16,
            num_workers=self.num_workers,
            collate_fn=lambda x: x)

        with tqdm(total=len(dl)) as pbar:
            for batch in dl:
                pbar.update(1)
                clips, fps = list(zip(*batch))
                clips = [torch.as_tensor(c) for c in clips]
                self.video_pts.extend(clips)
                self.video_fps.extend(fps)

    def _init_from_metadata(self, metadata):
        self.video_paths = metadata["video_paths"]
        assert len(self.video_paths) == len(metadata["video_pts"])
        self.video_pts = metadata["video_pts"]
        assert len(self.video_paths) == len(metadata["video_fps"])
        self.video_fps = metadata["video_fps"]

    @property
    def metadata(self):
        _metadata = {
            "video_paths": self.video_paths,
            "video_pts": self.video_pts,
            "video_fps": self.video_fps
        }
        return _metadata

    def subset(self, indices):
        video_paths = [self.video_paths[i] for i in indices]
        video_pts = [self.video_pts[i] for i in indices]
        video_fps = [self.video_fps[i] for i in indices]
        metadata = {
            "video_paths": video_paths,
            "video_pts": video_pts,
            "video_fps": video_fps
        }
        return type(self)(video_paths, self.num_frames, self.step, self.frame_rate,
                          _precomputed_metadata=metadata, num_workers=self.num_workers,
                          _video_width=self._video_width,
                          _video_height=self._video_height,
                          _video_min_dimension=self._video_min_dimension,
                          _audio_samples=self._audio_samples)

    @staticmethod
    def compute_clips_for_video(video_pts, num_frames, step, fps, frame_rate):
        if fps is None:
            # if for some reason the video doesn't have fps (because doesn't have a video stream)
            # set the fps to 1. The value doesn't matter, because video_pts is empty anyway
            fps = 1
        if frame_rate is None:
            frame_rate = fps
        total_frames = len(video_pts) * (float(frame_rate) / fps)
        idxs = VideoClips._resample_video_idx(int(math.floor(total_frames)), fps, frame_rate)
        video_pts = video_pts[idxs]
        clips = unfold(video_pts, num_frames, step)
        if isinstance(idxs, slice):
            idxs = [idxs] * len(clips)
        else:
            idxs = unfold(idxs, num_frames, step)
        return clips, idxs

    def compute_clips(self, num_frames, step, frame_rate=None):
        """
        Compute all consecutive sequences of clips from video_pts.
        Always returns clips of size `num_frames`, meaning that the
        last few frames in a video can potentially be dropped.
        Arguments:
            num_frames (int): number of frames for the clip
            step (int): distance between two clips
        """
        self.num_frames = num_frames
        self.step = step
        self.frame_rate = frame_rate
        self.clips = []
        self.resampling_idxs = []
        # print('length of video_pts',len(self.video_pts),'length of video_paths',len(self.video_paths))
        for video_pts, fps in zip(self.video_pts, self.video_fps):
            clips, idxs = self.compute_clips_for_video(video_pts, num_frames, step, fps, frame_rate)
            self.clips.append(clips)
            self.resampling_idxs.append(idxs)
        clip_lengths_list = []
        for vi, v in enumerate(self.clips):
            clip_lengths_list.append(len(v))
            # if len(v) != 1:
                # print(vi,v,self.video_paths[vi])
                # print()
        clip_lengths = torch.as_tensor(clip_lengths_list)
        self.cumulative_sizes = clip_lengths.cumsum(0).tolist()

    def __len__(self):
        return self.num_clips()

    def num_videos(self):
        return len(self.video_paths)

    def num_clips(self):
        """
        Number of subclips that are available in the video list.
        """
        return self.cumulative_sizes[-1]

    def get_clip_location(self, idx):
        """
        Converts a flattened representation of the indices into a video_idx, clip_idx
        representation.
        """
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if video_idx == 0:
            clip_idx = idx
        else:
            clip_idx = idx - self.cumulative_sizes[video_idx - 1]
        return video_idx, clip_idx

    @staticmethod
    def _resample_video_idx(num_frames, original_fps, new_fps):
        step = float(original_fps) / new_fps
        if step.is_integer():
            # optimization: if step is integer, don't need to perform
            # advanced indexing
            step = int(step)
            return slice(None, None, step)
        idxs = torch.arange(num_frames, dtype=torch.float32) * step
        idxs = idxs.floor().to(torch.int64)
        return idxs

    def get_clip(self, idx):
        """
        Gets a subclip from a list of videos.
        Arguments:
            idx (int): index of the subclip. Must be between 0 and num_clips().
        Returns:
            video (Tensor)
            audio (Tensor)
            info (Dict)
            video_idx (int): index of the video in `video_paths`
        """
        if idx >= self.num_clips():
            raise IndexError("Index {} out of range "
                             "({} number of clips)".format(idx, self.num_clips()))
        video_idx, clip_idx = self.get_clip_location(idx)
        video_path = self.video_paths[video_idx]
        clip_pts = self.clips[video_idx][clip_idx]

        # from torchvision import get_video_backend
        backend = 'pyav' #xiaodan: hard coded to be pyav

        if backend == "pyav":
            # check for invalid options
            if self._video_width != 0:
                raise ValueError("pyav backend doesn't support _video_width != 0")
            if self._video_height != 0:
                raise ValueError("pyav backend doesn't support _video_height != 0")
            if self._video_min_dimension != 0:
                raise ValueError("pyav backend doesn't support _video_min_dimension != 0")
            if self._audio_samples != 0:
                raise ValueError("pyav backend doesn't support _audio_samples != 0")

        if backend == "pyav":
            start_pts = clip_pts[0].item()
            end_pts = clip_pts[-1].item()
            video, audio, info = read_video(video_path, start_pts, end_pts)
        else:
            info = _probe_video_from_file(video_path)
            video_fps = info["video_fps"]
            audio_fps = None

            video_start_pts = clip_pts[0].item()
            video_end_pts = clip_pts[-1].item()

            audio_start_pts, audio_end_pts = 0, -1
            audio_timebase = Fraction(0, 1)
            if "audio_timebase" in info:
                audio_timebase = info["audio_timebase"]
                audio_start_pts = pts_convert(
                    video_start_pts,
                    info["video_timebase"],
                    info["audio_timebase"],
                    math.floor,
                )
                audio_end_pts = pts_convert(
                    video_end_pts,
                    info["video_timebase"],
                    info["audio_timebase"],
                    math.ceil,
                )
                audio_fps = info["audio_sample_rate"]
            video, audio, info = _read_video_from_file(
                video_path,
                video_width=self._video_width,
                video_height=self._video_height,
                video_min_dimension=self._video_min_dimension,
                video_pts_range=(video_start_pts, video_end_pts),
                video_timebase=info["video_timebase"],
                audio_samples=self._audio_samples,
                audio_pts_range=(audio_start_pts, audio_end_pts),
                audio_timebase=audio_timebase,
            )

            info = {"video_fps": video_fps}
            if audio_fps is not None:
                info["audio_fps"] = audio_fps

        if self.frame_rate is not None:
            resampling_idx = self.resampling_idxs[video_idx][clip_idx]
            if isinstance(resampling_idx, torch.Tensor):
                resampling_idx = resampling_idx - resampling_idx[0]
            video = video[resampling_idx]
            info["video_fps"] = self.frame_rate
        assert len(video) == self.num_frames, "{} x {}".format(video.shape, self.num_frames)
        return video, audio, info, video_idx
