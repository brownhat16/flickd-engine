import assemblyai as aai
import moviepy.editor as mp
from pathlib import Path
from typing import Optional
import logging

class AudioTranscriber:
    def __init__(self, api_key: str):
        """
        Initialize audio transcriber with AssemblyAI
        
        Args:
            api_key: AssemblyAI API key
        """
        aai.settings.api_key = api_key
        self.transcriber = aai.Transcriber()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def extract_audio_from_video(self, video_path: str) -> Path:
        """
        Extract audio from video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Path to extracted audio file
        """
        try:
            video_path = Path(video_path)
            audio_path = video_path.with_suffix('.wav')
            
            # Extract audio using moviepy
            video = mp.VideoFileClip(str(video_path))
            
            if video.audio is None:
                self.logger.warning(f"No audio track found in {video_path}")
                video.close()
                return None
            
            video.audio.write_audiofile(
                str(audio_path), 
                verbose=False, 
                logger=None,
                temp_audiofile='temp-audio.m4a'
            )
            video.close()
            
            self.logger.info(f"Audio extracted to {audio_path}")
            return audio_path
            
        except Exception as e:
            self.logger.error(f"Error extracting audio from {video_path}: {e}")
            return None
    
    def transcribe_audio(self, audio_path: str) -> Optional[str]:
        """
        Transcribe audio to text using AssemblyAI
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text or None if failed
        """
        try:
            if not audio_path or not Path(audio_path).exists():
                self.logger.warning(f"Audio file not found: {audio_path}")
                return None
            
            self.logger.info(f"Starting transcription of {audio_path}")
            
            # Configure transcription settings
            config = aai.TranscriptionConfig(
                speech_model=aai.SpeechModel.nano,  # Faster model for short videos
                language_detection=True,
                punctuate=True,
                format_text=True
            )
            
            transcript = self.transcriber.transcribe(str(audio_path), config=config)
            
            if transcript.status == aai.TranscriptStatus.error:
                self.logger.error(f"Transcription failed: {transcript.error}")
                return None
            
            self.logger.info("Transcription completed successfully")
            return transcript.text
            
        except Exception as e:
            self.logger.error(f"Error transcribing audio: {e}")
            return None
    
    def transcribe_video(self, video_path: str, cleanup_audio: bool = True) -> Optional[str]:
        """
        Extract audio from video and transcribe it
        
        Args:
            video_path: Path to video file
            cleanup_audio: Whether to delete extracted audio file after transcription
            
        Returns:
            Transcribed text or None if failed
        """
        try:
            # Extract audio
            audio_path = self.extract_audio_from_video(video_path)
            
            if not audio_path:
                return None
            
            # Transcribe audio
            transcript = self.transcribe_audio(str(audio_path))
            
            # Cleanup audio file if requested
            if cleanup_audio and audio_path and audio_path.exists():
                audio_path.unlink()
                self.logger.info(f"Cleaned up audio file: {audio_path}")
            
            return transcript
            
        except Exception as e:
            self.logger.error(f"Error in video transcription: {e}")
            return None
