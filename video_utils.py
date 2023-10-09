from moviepy.editor import VideoFileClip,AudioFileClip, clips_array

def separate_av(input_file, output_audio_path, output_video_path):
    try:
        video_clip = VideoFileClip(input_file)
        audio_clip = video_clip.audio
        video_clip_without_audio = video_clip.set_audio(None)
        audio_clip.write_audiofile(output_audio_path)
        video_clip_without_audio.write_videofile(output_video_path, codec="libx264")
        video_clip.close()
        audio_clip.close()
        print("Separation complete.")
    except Exception as e:
        print(f"Error: {e}")

def merge_av(video_path, audio_path, output_path):
    try:
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)

        video_clip = video_clip.set_audio(audio_clip)

        video_clip.write_videofile(output_path, codec="libx264")

        print("Merging complete.")
    except Exception as e:
        print(f"Error: {e}")
