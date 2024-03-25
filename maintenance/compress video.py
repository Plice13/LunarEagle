import os
from moviepy.editor import VideoFileClip

def compress_video(input_path, output_path, bitrate='1000k'):
    clip = VideoFileClip(input_path)
    compressed_clip = clip.resize(width=clip.size[0] // 32, height=clip.size[1] // 32)  # Reduce size by half
    compressed_clip.write_videofile(output_path, bitrate=bitrate)
    clip.close()

# Example usage
folder = r'C:\Users\PlicEduard'
input_video_path = os.path.join(folder, 'S1170017.MP4')
output_video_path = os.path.join(folder, 'output.mp4')
compress_video(input_video_path, output_video_path)
