import os
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from pytube import extract
from pydantic import BaseModel
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from background_task import process_video_transcript, process_audio_transcription
from db import get_db, Video, Audio, AudioChunk
from utils.token_utils import count_tokens, smart_chunk_selection

app = FastAPI()

load_dotenv()

endpoint = "https://models.inference.ai.azure.com"
model_name = "AI21-Jamba-1.5-Mini"
token = os.getenv("GITHUB_TOKEN")

if not token:
    raise HTTPException(status_code=404, detail="Token not found")

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)


class URLRequest(BaseModel):
    url: str


@app.get("/")
async def root():
    return {"status": "This service is running!"}


@app.post("/transcript/url")
async def getTranscriptfromURL(
    request: URLRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    url = request.url
    v_id = extract.video_id(url)

    # Check if video already exists
    existing_video = db.query(Video).filter(Video.v_id == v_id).first()
    if existing_video:
        if existing_video.status == "completed":
            response = client.complete(
                messages=[
                    SystemMessage(
                        content="""
                            Summarize the provided transcript in one paragraph, not exceeding 100 words. 
                            Always start with phrases like 'This video is about...' or 'In this video...'. 
                            The summary should be concise and informative.
                        """
                    ),
                    UserMessage(content=existing_video.transcript),
                ],
                temperature=1.0,
                top_p=1.0,
                max_tokens=200,
                model=model_name,
            )
            return {"summary": response.choices[0].message.content}
        elif existing_video.status == "processing":
            return {
                "message": "Video transcript is still being processed",
                "video_id": str(existing_video.id),
            }
        else:
            return {
                "message": f"Video processing failed with status: {existing_video.status}",
                "video_id": str(existing_video.id),
            }

    try:
        new_video = Video(v_id=v_id, v_url=url, status="processing")
        db.add(new_video)
        db.commit()

        background_tasks.add_task(process_video_transcript, v_id, db)
        return {
            "message": "Video transcript processing started",
            "video_id": str(new_video.id),
        }

    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/transcript/audio")
async def getTranscriptfromAudio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    try:
        # Validate file type
        valid_extensions = (".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm")
        if not file.filename.endswith(valid_extensions):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported formats: {', '.join(valid_extensions)}",
            )

        # Create uploads directory
        folder = "audio/uploads"
        os.makedirs(folder, exist_ok=True)

        # Save the uploaded file
        file_location = os.path.join(folder, file.filename)
        with open(file_location, "wb+") as f:
            f.write(file.file.read())

        # Create new audio entry
        new_audio = Audio(filename=file.filename, status="processing")
        db.add(new_audio)
        db.commit()

        # Process audio in background
        background_tasks.add_task(
            process_audio_transcription, file_location, str(new_audio.id), db
        )

        return {
            "message": "Audio transcript processing started",
            "audio_id": str(new_audio.id),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/transcript/audio/{audio_id}")
async def getAudioTranscriptionStatus(audio_id: str, db: Session = Depends(get_db)):
    audio = db.query(Audio).filter(Audio.id == audio_id).first()
    if not audio:
        raise HTTPException(status_code=404, detail="Audio not found")

    if audio.status == "completed":
        transcript_text = audio.transcript
        token_count = count_tokens(transcript_text)
        print(f"Transcript length: {token_count} tokens")

        if token_count > 8000:
            chunks = db.query(AudioChunk).filter(AudioChunk.audio_id == audio_id).all()
            if chunks:
                transcript_text = smart_chunk_selection(chunks)
            else:
                print("Warning: Long transcript with no chunks available")
                return {
                    "status": "error",
                    "message": "Transcript too long for processing",
                }

        response = client.complete(
            messages=[
                SystemMessage(
                    content="""
                        Summarize the provided transcript in one paragraph, not exceeding 100 words. 
                        Always start with phrases like 'This audio is about...' or 'In this audio...'. 
                        The summary should be concise and informative.
                    """
                ),
                UserMessage(content=transcript_text),
            ],
            temperature=1.0,
            top_p=1.0,
            model=model_name,
        )
        return {"status": audio.status, "summary": response.choices[0].message.content}

    return {"status": audio.status}
