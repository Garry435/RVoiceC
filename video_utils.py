import subprocess
import os

def seperate_av(input_video,output_video,output_audio):
    command = f"ffmpeg -i {input_video} -an -vcodec copy {output_video}"
    subprocess.call(command, shell=True)
    command = f"ffmpeg -i {input_video} -vn -acodec libmp3lame {output_audio}"
    subprocess.call(command, shell=True)
    print(f"Audio extracted successfully. Output audio: {output_audio}") if os.path.exists(output_audio) else print("Error: Output audio file not found.")
    print(f"Video extracted successfully. Output Video: {output_video}") if os.path.exists(output_video) else print("Error: Output video file not found.")
def merge_av(output_video,input_video,input_audio):
    command = f'ffmpeg -i {input_video} -i {input_audio} -c:v copy -c:a aac -strict experimental {output_video}'
    subprocess.call(command, shell=True)
    print(f"Video merged successfully. Output Video: {output_video}") if os.path.exists(output_video) else print("Error: Output video file not found.")
