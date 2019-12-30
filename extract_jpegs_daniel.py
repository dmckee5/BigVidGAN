# copied from TimeCycle preprocessing package
import os
import subprocess
from joblib import delayed
from joblib import Parallel

# folder_path = '/data/dbmckee2/kinetics-400/train/Kinetics_trimmed_videos_train_merge/'
output_path = '/home/ubuntu/kinetics-400/kinetics/frames'
file_src = '/home/ubuntu/kinetics-400/kinetics/Kinetics_trimmed_videos_train_merge/'


file_list = []

# f = open(file_src, 'r')
for file in os.listdir(file_src):
    if file.endswith('mp4'):
        # line = line[:-1]
        file_list.append(os.path.join(file_src, file))
# f.close()


def download_clip(inname):

    status = False
    inname = '%s' % inname
    outname = '%s' % os.path.basename(inname).replace('.mp4', '') #[:-4]
    outname = os.path.join(output_path, outname)
    #os.makedirs(outname)
    # command = "ffmpeg  -loglevel panic -i {} -q:v 1 -vf fps=12 {}/%06d.jpg".format( inname, outname)
    command = "ffmpeg -loglevel verbose -i {} -q:v 1 -vf fps=12 {}/%06d.jpg".format( inname, outname)
    # ffmpeg  -loglevel panic -i /scratch/xiaolonw/kinetics/data/train/making_tea/DImSF2kwc5g_000083_000093.mp4 -q:v 1 -vf fps=12 /nfs.yoda/xiaolonw/kinetics/jpg_outs/%06d.jpg
    try:
        output = subprocess.check_output(command, shell=True,
                                         stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        print(err.output)
        return status, err.output

    # Check if the video was successfully saved.
    status = os.path.exists(inname[:-4])
    return status, 'Downloaded'


def download_clip_wrapper(row):
    """Wrapper for parallel processing purposes."""

    videoname = os.path.splitext(os.path.basename(row))[0]

    inname = row #folder_path  + '/' + videoname + '/clip.mp4'
    outname = output_path + '/' +videoname

    if os.path.isdir(outname) is False:
        try:
            os.makedirs( outname, 0o755 )
        except:
            print(outname)

    downloaded, log = download_clip(inname)
    return downloaded


# def main(input_csv, output_dir, trim_format='%06d', num_jobs=24, tmp_dir='/tmp/kinetics'):

# if __name__ == '__main__':

status_lst = Parallel(n_jobs=36)(delayed(download_clip_wrapper)(row) for row in file_list)
# status_lst = [download_clip_wrapper(row) for row in file_list] #Parallel(n_jobs=10)(delayed(download_clip_wrapper)(row) for row in file_list)
