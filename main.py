# from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
# import shutil
# import os
# import uuid
# from datetime import datetime
# from moviepy.editor import VideoFileClip, concatenate_videoclips
# import whisper
# from transformers import MarianMTModel, MarianTokenizer, pipeline
# from scenedetect import VideoManager, SceneManager
# from scenedetect.detectors import ContentDetector
# from gtts import gTTS
# from pydub import AudioSegment
# import subprocess
# import logging
# from pymongo import MongoClient

# # Initialize FastAPI app
# app = FastAPI()

# # Setup logging
# logging.basicConfig(level=logging.INFO)

# # MongoDB connection settings
# MONGO_URI = "mongodb://127.0.0.1:27017/"  # Update as per your MongoDB setup
# DATABASE_NAME = "video_processing_data"
# client = MongoClient(MONGO_URI)
# db = client[DATABASE_NAME]
# video_collection = db["videos"]

# # Allowed video extensions
# ALLOWED_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mkv", ".mov"]

# # Ensure directories exist for uploads and outputs
# BASE_UPLOAD_FOLDER = "uploads"
# BASE_OUTPUT_FOLDER = "outputs"
# os.makedirs(BASE_UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(BASE_OUTPUT_FOLDER, exist_ok=True)

# # Helper function: Check if the file is a video
# def is_video_file(filename: str):
#     ext = os.path.splitext(filename)[1].lower()
#     return ext in ALLOWED_VIDEO_EXTENSIONS

# # Helper function: Create a unique folder
# def create_unique_output_folder():
#     unique_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}"
#     unique_folder = os.path.join(BASE_OUTPUT_FOLDER, unique_id)
#     os.makedirs(unique_folder, exist_ok=True)
#     return unique_folder

# # Scene detection
# def detect_scenes(video_path):
#     video_manager = VideoManager([video_path])
#     scene_manager = SceneManager()
#     scene_manager.add_detector(ContentDetector(threshold=20.0))

#     video_manager.set_downscale_factor()
#     video_manager.start()
#     scene_manager.detect_scenes(frame_source=video_manager)
#     scene_list = scene_manager.get_scene_list()
#     video_manager.release()

#     scenes = [(scene[0].get_seconds(), scene[1].get_seconds()) for scene in scene_list]
#     if not scenes:
#         raise ValueError("No scenes detected, adjust detection settings.")
#     logging.info(f"Detected {len(scenes)} scenes.")
#     return scenes

# # Summarize video
# def create_summary(video_path, scene_list, output_folder):
#     video = VideoFileClip(video_path)
#     key_clips = [video.subclip(start, end) for start, end in scene_list]
#     key_clips = sorted(key_clips, key=lambda clip: clip.duration, reverse=True)
#     summary_clips = key_clips[:10] if len(key_clips) > 10 else key_clips
#     summary_clip = concatenate_videoclips(summary_clips, method="compose")

#     summarized_video_path = os.path.join(output_folder, "summarized_video.mp4")
#     summary_clip.write_videofile(summarized_video_path, codec="libx264")

#     video.close()
#     logging.info(f"Summarized video created at {summarized_video_path}")
#     return summarized_video_path

# # Transcription
# def transcribe_audio(video_path, output_folder):
#     audio_path = os.path.join(output_folder, "extracted_audio.mp3")
#     video = VideoFileClip(video_path)

#     if video.audio is not None:
#         video.audio.write_audiofile(audio_path)
#     else:
#         raise ValueError("The video does not contain any audio.")

#     video.close()

#     model = whisper.load_model("large")
#     result = model.transcribe(audio_path)
#     transcription = result["text"]

#     transcription_file = os.path.join(output_folder, "transcription.txt")
#     with open(transcription_file, "w") as f:
#         f.write(transcription)

#     logging.info(f"Transcription completed and saved to {transcription_file}")
#     return transcription, result["segments"]

# # Translation
# def translate_text_to_hindi(transcription_segments, output_folder):
#     model_name = "Helsinki-NLP/opus-mt-en-hi"
#     tokenizer = MarianTokenizer.from_pretrained(model_name)
#     model = MarianMTModel.from_pretrained(model_name)

#     translated_segments = []
#     for segment in transcription_segments:
#         text = segment["text"]
#         inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#         translated = model.generate(**inputs)
#         hindi_text = tokenizer.decode(translated[0], skip_special_tokens=True)
#         translated_segments.append((hindi_text, segment["start"], segment["end"]))

#     logging.info(f"Translation to Hindi completed for {len(translated_segments)} segments.")
#     return translated_segments

# def create_hindi_audio(translated_segments, output_folder, delay=500):
#     hindi_audio_path = os.path.join(output_folder, "hindi_audio.mp3")
#     combined_audio = AudioSegment.silent(duration=0)

#     for hindi_text, start, end in translated_segments:
#         tts = gTTS(hindi_text, lang='hi')
#         segment_audio_path = os.path.join(output_folder, f"hindi_segment_{start}_{end}.mp3")
#         tts.save(segment_audio_path)

#         segment_audio = AudioSegment.from_file(segment_audio_path)
#         desired_duration = (end - start) * 1000
#         if len(segment_audio) > desired_duration:
#             segment_audio = segment_audio[:desired_duration]
#         else:
#             segment_audio = segment_audio + AudioSegment.silent(duration=desired_duration - len(segment_audio))
#         segment_audio += AudioSegment.silent(duration=delay)

#         combined_audio += segment_audio

#     combined_audio.export(hindi_audio_path, format="mp3")

#     logging.info(f"Hindi audio created and saved at {hindi_audio_path}")
#     return hindi_audio_path

# # Step 6: Merge summarized video and Hindi audio using ffmpeg (Fix to include Hindi audio and exclude original audio)
# def merge_audio_video_ffmpeg(video_path, hindi_audio_path, output_folder):
#     output_video_path = os.path.join(output_folder, "final_output_with_hindi_audio.mp4")

#     # FFmpeg command to map the video and Hindi audio and disable the original audio
#     command = (
#         f'ffmpeg -i "{video_path}" -i "{hindi_audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac '
#         f'-shortest -y "{output_video_path}"'
#     )

#     # Log the command being executed for troubleshooting
#     logging.info(f"Running FFmpeg command: {command}")

#     try:
#         # Capture FFmpeg logs and errors
#         subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     except subprocess.CalledProcessError as e:
#         logging.error(f"Error during FFmpeg execution: {e.stderr.decode('utf-8')}")
#         raise Exception(f"FFmpeg merge failed: {e.stderr.decode('utf-8')}")

#     logging.info(f"Final video with Hindi audio created at {output_video_path}")
#     return output_video_path


# # Summarize text using BART
# def summarize_text(transcription: str, output_folder: str, max_input_length=1024, max_output_length=150):
#     summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
#     chunks = [transcription[i:i+max_input_length] for i in range(0, len(transcription), max_input_length)]
#     summary_parts = []
#     for chunk in chunks:
#         summary = summarizer(chunk, max_length=max_output_length, min_length=50, do_sample=False)
#         summary_parts.append(summary[0]['summary_text'])
#     full_summary = " ".join(summary_parts)
#     summary_file_path = os.path.join(output_folder, "summarized_transcription.txt")
#     with open(summary_file_path, "w") as f:
#         f.write(full_summary)
#     logging.info(f"Transcription summary saved to {summary_file_path}")
#     return full_summary, summary_file_path


# # Background task for video processing
# def process_video(video_path, output_folder, filename):
#     try:
#         # Scene detection
#         scenes = detect_scenes(video_path)

#         # Summarize video
#         summarized_video_path = create_summary(video_path, scenes, output_folder)

#         # Transcription
#         transcription, transcription_segments = transcribe_audio(summarized_video_path, output_folder)

#         # Summarize transcription
#         transcription_summary, summary_file_path = summarize_text(transcription, output_folder)

#         # Store data in MongoDB
#         video_data = {
#             "original_filename": filename,
#             "uploaded_at": datetime.now(),
#             "output_folder": output_folder,
#             "summarized_video_path": summarized_video_path,
#             "transcription": transcription,
#             "summarized_transcription": transcription_summary,
#         }
#         video_collection.insert_one(video_data)

#         logging.info("Video processing completed successfully.")

#     except Exception as e:
#         logging.error(f"An error occurred during video processing: {str(e)}")


# # FastAPI endpoint
# @app.post("/uploadvideo/")
# async def upload_video(file: UploadFile = File(...), background_tasks: BackgroundTasks):
#     try:
#         if not is_video_file(file.filename):
#             raise HTTPException(status_code=400, detail="Invalid file format. Please upload a valid video file.")

#         output_folder = create_unique_output_folder()
#         video_path = os.path.join(output_folder, file.filename)
#         with open(video_path, "wb") as f:
#             shutil.copyfileobj(file.file, f)

#         # Add video processing to background tasks
#         background_tasks.add_task(process_video, video_path, output_folder, file.filename)

#         # Return immediate response
#         return {"message": "Video is being processed in the background. You will be notified when it's done."}

#     except Exception as e:
#         logging.error(f"An error occurred: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

from fastapi import FastAPI, File, UploadFile, HTTPException
import shutil
import os
import uuid
from datetime import datetime
from moviepy.editor import VideoFileClip, concatenate_videoclips
import whisper
from transformers import MarianMTModel, MarianTokenizer, pipeline
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from gtts import gTTS
from pydub import AudioSegment
import subprocess
import logging
from pymongo import MongoClient

# Initialize FastAPI app
app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)

# MongoDB connection settings
MONGO_URI = "mongodb://127.0.0.1:27017/"  # Update as per your MongoDB setup
DATABASE_NAME = "video_processing_data"
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
video_collection = db["videos"]

# Allowed video extensions
ALLOWED_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mkv", ".mov"]

# Ensure directories exist for uploads and outputs
BASE_UPLOAD_FOLDER = "uploads"
BASE_OUTPUT_FOLDER = "outputs"
os.makedirs(BASE_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(BASE_OUTPUT_FOLDER, exist_ok=True)

# Helper function: Check if the file is a video
def is_video_file(filename: str):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_VIDEO_EXTENSIONS

# Helper function: Create a unique folder
def create_unique_output_folder():
    unique_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}"
    unique_folder = os.path.join(BASE_OUTPUT_FOLDER, unique_id)
    os.makedirs(unique_folder, exist_ok=True)
    return unique_folder

# Scene detection
def detect_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=20.0))

    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    video_manager.release()

    scenes = [(scene[0].get_seconds(), scene[1].get_seconds()) for scene in scene_list]
    if not scenes:
        raise ValueError("No scenes detected, adjust detection settings.")
    logging.info(f"Detected {len(scenes)} scenes.")
    return scenes

# Summarize video
def create_summary(video_path, scene_list, output_folder):
    video = VideoFileClip(video_path)
    key_clips = [video.subclip(start, end) for start, end in scene_list]
    key_clips = sorted(key_clips, key=lambda clip: clip.duration, reverse=True)
    summary_clips = key_clips[:10] if len(key_clips) > 10 else key_clips
    summary_clip = concatenate_videoclips(summary_clips, method="compose")

    summarized_video_path = os.path.join(output_folder, "summarized_video.mp4")
    summary_clip.write_videofile(summarized_video_path, codec="libx264")

    video.close()
    logging.info(f"Summarized video created at {summarized_video_path}")
    return summarized_video_path

# Transcription
def transcribe_audio(video_path, output_folder):
    audio_path = os.path.join(output_folder, "extracted_audio.mp3")
    video = VideoFileClip(video_path)

    if video.audio is not None:
        video.audio.write_audiofile(audio_path)
    else:
        raise ValueError("The video does not contain any audio.")

    video.close()

    model = whisper.load_model("large")
    result = model.transcribe(audio_path)
    transcription = result["text"]

    transcription_file = os.path.join(output_folder, "transcription.txt")
    with open(transcription_file, "w") as f:
        f.write(transcription)

    logging.info(f"Transcription completed and saved to {transcription_file}")
    return transcription, result["segments"]

# Translation
def translate_text_to_hindi(transcription_segments, output_folder):
    model_name = "Helsinki-NLP/opus-mt-en-hi"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    translated_segments = []
    for segment in transcription_segments:
        text = segment["text"]
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**inputs)
        hindi_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        translated_segments.append((hindi_text, segment["start"], segment["end"]))

    logging.info(f"Translation to Hindi completed for {len(translated_segments)} segments.")
    return translated_segments

def create_hindi_audio(translated_segments, output_folder, delay=500):
    hindi_audio_path = os.path.join(output_folder, "hindi_audio.mp3")
    combined_audio = AudioSegment.silent(duration=0)

    for hindi_text, start, end in translated_segments:
        tts = gTTS(hindi_text, lang='hi')
        segment_audio_path = os.path.join(output_folder, f"hindi_segment_{start}_{end}.mp3")
        tts.save(segment_audio_path)

        segment_audio = AudioSegment.from_file(segment_audio_path)
        desired_duration = (end - start) * 1000
        if len(segment_audio) > desired_duration:
            segment_audio = segment_audio[:desired_duration]
        else:
            segment_audio = segment_audio + AudioSegment.silent(duration=desired_duration - len(segment_audio))
        segment_audio += AudioSegment.silent(duration=delay)

        combined_audio += segment_audio

    combined_audio.export(hindi_audio_path, format="mp3")

    logging.info(f"Hindi audio created and saved at {hindi_audio_path}")
    return hindi_audio_path

def merge_audio_video_ffmpeg(video_path, hindi_audio_path, output_folder):
    output_video_path = os.path.join(output_folder, "final_output_with_hindi_audio.mp4")

    command = (
        f'ffmpeg -i "{video_path}" -i "{hindi_audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac '
        f'-shortest -y "{output_video_path}"'
    )

    logging.info(f"Running FFmpeg command: {command}")

    try:
        subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during FFmpeg execution: {e.stderr.decode('utf-8')}")
        raise Exception(f"FFmpeg merge failed: {e.stderr.decode('utf-8')}")

    logging.info(f"Final video with Hindi audio created at {output_video_path}")
    return output_video_path

# Summarize text using BART
def summarize_text(transcription: str, output_folder: str, max_input_length=1024, max_output_length=150):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    chunks = [transcription[i:i+max_input_length] for i in range(0, len(transcription), max_input_length)]
    summary_parts = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=max_output_length, min_length=50, do_sample=False)
        summary_parts.append(summary[0]['summary_text'])
    full_summary = " ".join(summary_parts)
    summary_file_path = os.path.join(output_folder, "summarized_transcription.txt")
    with open(summary_file_path, "w") as f:
        f.write(full_summary)
    logging.info(f"Transcription summary saved to {summary_file_path}")
    return full_summary, summary_file_path

# FastAPI endpoint
@app.post("/uploadvideo/")
async def upload_video(file: UploadFile = File(...)):
    try:
        if not is_video_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload a valid video file.")

        output_folder = create_unique_output_folder()
        video_path = os.path.join(output_folder, file.filename)
        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        scenes = detect_scenes(video_path)
        summarized_video_path = create_summary(video_path, scenes, output_folder)

        transcription, transcription_segments = transcribe_audio(summarized_video_path, output_folder)
        transcription_summary, summary_file_path = summarize_text(transcription, output_folder)

        translated_segments = translate_text_to_hindi(transcription_segments, output_folder)
        hindi_audio_path = create_hindi_audio(translated_segments, output_folder)

        final_video_path = merge_audio_video_ffmpeg(summarized_video_path, hindi_audio_path, output_folder)

        video_data = {
            "original_filename": file.filename,
            "uploaded_at": datetime.now(),
            "output_folder": output_folder,
            "summarized_video_path": summarized_video_path,
            "transcription": transcription,
            "summarized_transcription": transcription_summary,
            "final_video_path": final_video_path,
        }
        video_collection.insert_one(video_data)

        return {
            "message": "Video processed successfully",
            "summary": transcription_summary,
            "summarized_video": summarized_video_path,
            "final_video": final_video_path,
        }

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
