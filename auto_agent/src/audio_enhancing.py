import librosa
import numpy as np
import requests
import subprocess
import tempfile
import json
import os 
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class AudioEnhancer:
    def __init__(self, editing_spec, original_audio_path):
        """
        Initialize audio enhancer with editing specification and original audio
        :param editing_spec: Complete editing specification
        :param original_audio_path: Path to original audio file
        """
        self.editing_spec = editing_spec
        self.original_audio_path = original_audio_path
        self.enhanced_audio_path = None

    def enhance_audio(self):
        """Apply audio enhancements to the original audio track"""
        if not self.original_audio_path:
            return self.original_audio_path
        
        try:
            # Create temporary output file
            temp_dir = tempfile.gettempdir()
            enhanced_path = os.path.join(temp_dir, "enhanced_audio.wav")

            # Build FFmpeg command for audio enhancement
            cmd = [
                'ffmpeg',
                '-i', self.original_audio_path,
                '-af', self.get_audio_filters(),
                '-y', enhanced_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✅ Applied audio enhancements")
                return enhanced_path
            else:
                logger.error(f"Audio enhancement error: {result.stderr}")
                return self.original_audio_path
        except Exception as e:
            logger.error(f"Audio enhancement failed: {str(e)}")
            return self.original_audio_path
        
    def get_audio_filters(self):
        """Generate FFmpeg audio filters based on edit specifications"""
        filters = []
        
        # Always apply basic cleanup
        filters.append("highpass=f=80")  # Remove low-frequency noise
        filters.append("lowpass=f=15000")  # Remove high-frequency hiss
        
        # Dynamic range compression for clearer dialogue
        filters.append("compand=attacks=0:decays=0.3:points=-80/-80|-30/-12|0/-3")
        
        # Loudness normalization (EBU R128 standard)
        filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")

        # Special processing based on content type
        if "sports" in self.editing_spec.get("scene_types", []):
            filters.append("compand=attacks=0.1:decays=0.2:points=-90/-90|-70/-70|-30/-15|0/-3")
            filters.append("aecho=0.8:0.9:1000:0.3")  # Stadium reverb effect

        if "emotional" in self.editing_spec.get("scene_types", []):
            filters.append("asoftclip")  # Gentle clipping for warmth
            filters.append("bass=g=3")  # Boost low frequencies
            
        return ",".join(filters) 
    
    def integrate(self):
        """Main method to enhance and integrate original audio"""
        self.enhanced_audio_path = self.enhance_audio()
        
        # Update editing spec with enhanced audio
        self.editing_spec["audio_track"] = self.enhanced_audio_path
        self.editing_spec["audio_enhancements"] = True
        return self.editing_spec
            
    