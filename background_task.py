import os
from youtube_transcript_api import YouTubeTranscriptApi
from db import Video, Audio, AudioChunk
from sqlalchemy.orm import Session
from openai import OpenAI
from pydub import AudioSegment
import math
from dotenv import load_dotenv
from utils.chunk_text import chunk_text
from utils.create_embedding import create_embedding

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def process_video_transcript(v_id: str, db: Session):
    print(f"Starting video transcript processing for video ID: {v_id}")
    try:
        video = db.query(Video).filter(Video.v_id == v_id).first()
        if not video:
            print(f"Video not found in database: {v_id}")
            return

        print("Fetching YouTube transcript...")
        transcript_info = YouTubeTranscriptApi.get_transcript(v_id)
        transcript = " ".join([elem["text"] for elem in transcript_info])

        print("Updating database with transcript...")
        video.transcript = transcript
        video.status = "completed"
        db.commit()
        print("Video transcript processing completed successfully")

    except Exception as e:
        error_msg = f"Error processing video transcript: {str(e)}"
        print(error_msg)
        try:
            video.status = "error"
            video.transcript = (
                f"Error: {str(e)}"  # Store error message in transcript field
            )
            db.commit()
        except Exception as db_error:
            print(f"Failed to update error status in database: {str(db_error)}")


def process_audio_transcription(file_location: str, audio_id: str, db: Session):
    print(f"Starting audio transcription for file: {file_location}")
    try:
        audio_record = db.query(Audio).filter(Audio.id == audio_id).first()
        if not audio_record:
            print(f"Audio record not found in database: {audio_id}")
            return

        transcripts = []
        try:
            print("Loading audio file...")
            audio = AudioSegment.from_file(file_location)

            MAX_DURATION = 10 * 60 * 1000  # 10 minutes in milliseconds
            total_segments = math.ceil(len(audio) / MAX_DURATION)
            print(f"Audio will be processed in {total_segments} segments")

            for i in range(total_segments):
                print(f"Processing segment {i+1}/{total_segments}")
                start = i * MAX_DURATION
                end = min((i + 1) * MAX_DURATION, len(audio))
                segment = audio[start:end]

                temp_path = f"{file_location}_segment_{i}.mp3"
                print(f"Exporting segment to {temp_path}")
                segment.export(
                    temp_path, format="mp3", parameters=["-ac", "1", "-ar", "16000"]
                )

                file_size = os.path.getsize(temp_path) / (1024 * 1024)  # Size in MB
                print(f"Segment size: {file_size:.2f}MB")

                if file_size > 20:
                    raise Exception(
                        f"Segment too large ({file_size:.2f}MB) even after compression"
                    )

                try:
                    print("Transcribing segment...")
                    with open(temp_path, "rb") as audio_file:
                        transcription = client.audio.transcriptions.create(
                            model="whisper-1", file=audio_file
                        )
                    transcripts.append(transcription.text)
                    print(f"Segment {i+1} transcribed successfully")
                except Exception as whisper_error:
                    raise Exception(f"Whisper API error: {str(whisper_error)}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        print(f"Removed temporary file: {temp_path}")

            print("Combining transcripts and updating database...")
            final_transcript = " ".join(transcripts)
            audio_record.transcript = final_transcript
            audio_record.status = "completed"

            # Check transcript length
            estimated_tokens = len(final_transcript) // 4
            if estimated_tokens > 7000:
                print("Transcript exceeds token limit, creating chunks...")
                chunks = chunk_text(final_transcript)
                for chunk in chunks:
                    embedding = create_embedding(chunk)
                    chunk_record = AudioChunk(
                        audio_id=audio_id, chunk_text=chunk, embedding=embedding
                    )
                    db.add(chunk_record)

            db.commit()
            print("Audio transcription completed successfully")

            # After getting transcript:
            try:
                print("Chunking and embedding transcript...")
                chunks = chunk_text(final_transcript)
                for chunk in chunks:
                    embedding = create_embedding(chunk)
                    chunk_record = AudioChunk(
                        audio_id=audio_id, chunk_text=chunk, embedding=embedding
                    )
                    db.add(chunk_record)
                db.commit()
                print("Transcript chunking and embedding completed successfully")
            except Exception as e:
                print(f"Error processing transcript chunks: {str(e)}")

        except Exception as e:
            error_msg = f"Transcription error: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)

    except Exception as e:
        error_msg = f"Error processing audio transcript: {str(e)}"
        print(error_msg)
        try:
            audio_record.status = "error"
            audio_record.transcript = (
                f"Error: {str(e)}"  # Store error message in transcript field
            )
            db.commit()
        except Exception as db_error:
            print(f"Failed to update error status in database: {str(db_error)}")
    finally:
        try:
            if os.path.exists(file_location):
                os.remove(file_location)
                print(f"Removed original file: {file_location}")
        except Exception as cleanup_error:
            print(f"Failed to remove file {file_location}: {str(cleanup_error)}")
