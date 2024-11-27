## Transcript Summarization using FastAPI, AI21 Jambo 1.5 Mini, and OpenAI Whisper for Audio Transcription

This mini-project demonstrates how to use FastAPI to create an application for summarizing transcripts using AI21 Jambo 1.5 Mini and OpenAI-Whisper for audio transcription. The application allows users to paste a YouTube video link or upload an audio file to get a summary of their transcript.

### Components

- **FastAPI**: A modern web framework for building APIs with Python.
- **AI21 Jambo 1.5 Mini**: A state-of-the-art, hybrid SSM-Transformer.
- **Faster-Whisper**: A fast reimplementation of OpenAI's Whisper model using CTranslate2.

### How to Test the Application

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   ```

2. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set the API keys**:

   - Create a `.env` file in the root directory.
   - Add the following environment variables to the `.env` file:

     ```bash
     GITHUB_TOKEN=<your-github-token>
     OPENAI_API_KEY=<your-openai-api-key>
     ```

     - Replace `<your-github-token>` with your GitHub token. This token will allow you to access the models in the GitHub Marketplace, like AI21 Jambo 1.5 Mini.
     - Replace `<your-openai-api-key>` with your OpenAI API key. This key is necessary for accessing the OpenAI Whisper model.

4. **Start the FastAPI server**:

   ```bash
   uvicorn main:app --reload
   ```

   This will start the FastAPI server, and you can access the application at `http://localhost:8000`.

5. **Test the application**:
   - Open the Swagger UI at `http://localhost:8000/docs`.
   - Paste a YouTube video link or upload an audio file. There is a sample audio file in the `/audio/test` directory.
   - The application will return a summary of the transcript.
