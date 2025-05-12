
import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai
import time
from pathlib import Path
import tempfile
from dotenv import load_dotenv
import os
import re # For YouTube URL parsing

# For YouTube transcripts
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

# For website content fetching
import requests
from bs4 import BeautifulSoup

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# Page configuration
st.set_page_config(
    page_title="Multimodal AI Agent - Content Summarizer",
    page_icon="📝",
    layout="wide"
)

st.title("A.I based Summarizer Agent 📝🎥🎤🖬")
st.header("Multimodal Content Summarizer")

@st.cache_resource
def initialize_agent():
    return Agent(
        name="Content AI Summarizer",
        model=Gemini(model="gemini-1.5-flash-latest"),
        tools=[DuckDuckGo()],
        markdown=True,
    )

# Initialize the agent
summarizer_agent = initialize_agent()

# --- Predefined Prompts ---
PREDEFINED_PROMPTS = {
    "General Summary": "Provide a concise summary of the content in about 100 words.",
    "Key Points": "List the key points or main takeaways from the content.",
    "Explain Simply": "Explain the main topic of the content in simple terms.",
    "Actionable Insights": "What are the actionable insights from this content?",
    "Custom Prompt": ""
}

# --- Helper Functions ---
def get_youtube_video_id(url):
    """Extracts YouTube video ID from various URL formats."""
    regex = r"(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return None

def fetch_youtube_transcript(video_id):
    """Fetches transcript for a given YouTube video ID."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        target_language_codes = ['en']  # Default to English, as no preference was specified.
        try:
            # Try to find a manually created transcript in the target languages
            transcript = transcript_list.find_manually_created_transcript(target_language_codes)
        except NoTranscriptFound:
            # If no manual transcript is found in the target languages,
            # try to find an auto-generated transcript in the target languages.
            # This will raise NoTranscriptFound if no transcript is available in the target_language_codes,
            # which will be caught by the outer exception handler.
            transcript = transcript_list.find_generated_transcript(target_language_codes)
        
        full_transcript = " ".join([item.text for item in transcript.fetch()])
        return full_transcript
    except TranscriptsDisabled:
        return "Transcripts are disabled for this video."
    except NoTranscriptFound:
        return "No transcript found for this video. It might be a music video, very short, or processing."
    except Exception as e:
        return f"Could not retrieve transcript: {str(e)}"

def fetch_website_text(url):
    """Fetches and extracts text content from a website URL."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, "html.parser")
        
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s\s+", " ", text)
        return text
    except requests.exceptions.RequestException as e:
        return f"Error fetching website content: {str(e)}"
    except Exception as e:
        return f"Error parsing website content: {str(e)}"

# --- UI Elements & Logic ---
st.sidebar.header("Input Options")
input_type = st.sidebar.selectbox(
    "Select content type to summarize:",
    ("Video File (MP4, MOV, AVI)", "YouTube Video URL", "Website URL"),
    key="input_type_selector"
)

st.sidebar.header("Summarization Prompt")
selected_prompt_key = st.sidebar.selectbox(
    "Choose a prompt or select 'Custom Prompt' to write your own:",
    list(PREDEFINED_PROMPTS.keys()),
    key="prompt_selector"
)

if selected_prompt_key == "Custom Prompt":
    user_query = st.text_area(
        "Enter your custom summarization request:",
        placeholder="e.g., What are the main arguments presented?",
        help="Provide specific questions or insights you want.",
        key="custom_query_text_area"
    )
else:
    user_query = st.text_area(
        "Your selected prompt (edit if needed):",
        value=PREDEFINED_PROMPTS[selected_prompt_key],
        help="This is the selected predefined prompt. You can edit it if necessary.",
        key="predefined_query_text_area"
    )


if input_type == "Video File (MP4, MOV, AVI)":
    video_file = st.file_uploader(
        "Upload a video file", type=["mp4", "mov", "avi"], help="Upload a video for AI analysis", key="video_file_uploader"
    )
    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(video_file.read())
            video_path = temp_video.name

        st.video(video_path, format="video/mp4", start_time=0)

        if st.button("🔍 Analyze Video File", key="analyze_video_button"):
            if not user_query.strip():
                st.warning("Please enter a question or ensure a predefined prompt is selected.")
            else:
                try:
                    with st.spinner("Processing video and gathering insights..."):
                        processed_video = upload_file(video_path)
                        while processed_video.state.name == "PROCESSING":
                            time.sleep(2) 
                            processed_video = get_file(processed_video.name)
                        if processed_video.state.name == "FAILED":
                            st.error(f"Video processing failed. Error: {processed_video.error.message if processed_video.error else 'Unknown error'}")
                            st.stop()

                        analysis_prompt = (
                            f"""
                            Analyze the uploaded video for content and context.
                            Respond to the following query: {user_query}

                            Provide a detailed, user-friendly, and actionable response.
                            """
                        )
                        response = summarizer_agent.run(analysis_prompt, videos=[processed_video])
                    st.subheader("Video File Analysis Result")
                    st.markdown(response.content)
                except Exception as error:
                    st.error(f"An error occurred during video file analysis: {error}")
                finally:
                    if 'video_path' in locals() and Path(video_path).exists():
                        Path(video_path).unlink(missing_ok=True)
    else:
        st.info("Upload a video file to begin analysis.")

elif input_type == "YouTube Video URL":
    youtube_url = st.text_input("Enter YouTube Video URL:", key="youtube_url_input")
    if youtube_url:
        st.video(youtube_url)
        if st.button("🔍 Summarize YouTube Video", key="analyze_youtube_button"):
            if not user_query.strip():
                st.warning("Please enter a summarization request or ensure a predefined prompt is selected.")
            else:
                video_id = get_youtube_video_id(youtube_url)
                if not video_id:
                    st.error("Invalid YouTube URL. Please enter a valid URL.")
                else:
                    with st.spinner("Fetching Information and summarizing..."):
                        transcript = fetch_youtube_transcript(video_id)
                        if "Could not retrieve transcript" in transcript or "Transcripts are disabled" in transcript or "No transcript found" in transcript:
                            st.error(transcript)
                        else:
                            analysis_prompt = (
                                f"""
                                You are provided with a transcript from a YouTube video.
                                Based on this transcript, respond to the following query: {user_query}

                                Transcript:
                                {transcript}

                                Provide a detailed, user-friendly, and actionable summary or response.
                                """
                            )
                            try:
                                response = summarizer_agent.run(analysis_prompt)
                                st.subheader("YouTube Video Summary")
                                st.markdown(response.content)
                            except Exception as e:
                                st.error(f"Error during summarization: {str(e)}")
    else:
        st.info("Enter a YouTube video URL to summarize.")

elif input_type == "Website URL":
    website_url = st.text_input("Enter Website URL:", key="website_url_input")
    if website_url:
        if st.button("🔍 Summarize Website Content", key="analyze_website_button"):
            if not user_query.strip():
                st.warning("Please enter a summarization request or ensure a predefined prompt is selected.")
            else:
                with st.spinner("Fetching website content and summarizing..."):
                    content = fetch_website_text(website_url)
                    if "Error fetching website content" in content or "Error parsing website content" in content:
                        st.error(content)
                    elif not content.strip():
                        st.warning("Could not extract meaningful content from the website. It might be a very dynamic page or access is restricted.")
                    else:
                        max_chars = 25000 # Increased slightly, but still mindful of context limits
                        if len(content) > max_chars:
                            content = content[:max_chars] + "... [content truncated]"
                        
                        analysis_prompt = (
                            f"""
                            You are provided with text content extracted from a website.
                            Based on this content, respond to the following query: {user_query}

                            Website Content:
                            {content}

                            Provide a detailed, user-friendly, and actionable summary or response.
                            """
                        )
                        try:
                            response = summarizer_agent.run(analysis_prompt)
                            st.subheader("Website Content Summary")
                            st.markdown(response.content)
                        except Exception as e:
                            st.error(f"Error during summarization: {str(e)}")
    else:
        st.info("Enter a website URL to summarize.")

# Customize text area height
st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True
)