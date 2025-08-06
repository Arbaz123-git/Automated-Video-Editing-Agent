import numpy as np
import json
from videodb import connect
import os
from dotenv import load_dotenv
import cv2
import librosa
import requests
import tempfile
from urllib.parse import urlparse
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
class VideoDBSceneScorer:
    def __init__(self, video_metadata, edit_spec):
        """
        Initialize with video metadata and editing specifications
        :param video_metadata: Metadata from Step 1 (contains asset_id, scenes, etc.)
        :param edit_spec: Parsed specifications from Step 2
        """
        # Connect to VideoDB
        api_key = os.getenv("VIDEO_DB_API_KEY")
        if not api_key:
            raise ValueError("VIDEO_DB_API_KEY environment variable not set")
        
        self.videodb = connect(api_key=api_key)
        self.asset_id = video_metadata['asset_id']
        self.scenes = video_metadata['scenes']
        self.edit_spec = edit_spec
        self.stream_url = video_metadata.get('video_path')

        # Cache for downloaded video file
        self._video_file_path = None
        self._audio_file_path = None

        # Define weights for different scene types (tag-based)
        self.tag_weights = {
            "high_motion": 0.6,
            "stunts": 0.7,
            "crowd_reaction": 0.4,
            "action": 0.5,
            "emotional": 0.5,
            "romantic": 0.5,
            "sports": 0.5,
            "default": 0.3
        }

    def _download_video_file(self):
        """Download video file for local analysis"""
        if self._video_file_path and os.path.exists(self._video_file_path):
            return self._video_file_path
            
        try:
            # Create temporary file
            temp_dir = tempfile.mkdtemp()
            self._video_file_path = os.path.join(temp_dir, f"{self.asset_id}.mp4")

            # Download video from stream URL
            if self.stream_url:
                logger.info(f"📥 Downloading video from stream URL...")

                # Use ffmpeg to download HLS stream
                cmd = [
                    'ffmpeg', '-i', self.stream_url, 
                    '-c', 'copy', '-y', self._video_file_path
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error(f"FFmpeg error: {result.stderr}")
                    raise RuntimeError(f"Failed to download video: {result.stderr}")
                    
                logger.info(f"✅ Video downloaded to: {self._video_file_path}")
                return self._video_file_path
            else:
                raise ValueError("No stream URL available for download")
                
            
        except Exception as e:
            logger.error(f"Failed to download video: {str(e)}")
            raise

    def _extract_audio_file(self):
        """Extract audio from video file for audio analysis"""
        if self._audio_file_path and os.path.exists(self._audio_file_path):
            return self._audio_file_path
            
        try:
            video_path = self._download_video_file()
            temp_dir = os.path.dirname(video_path)
            self._audio_file_path = os.path.join(temp_dir, f"{self.asset_id}.wav")

            logger.info("🎵 Extracting audio from video...")

            # Extract audio using ffmpeg
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le', '-ar', '22050', '-ac', '1',
                '-y', self._audio_file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"FFmpeg audio extraction error: {result.stderr}")
                raise RuntimeError(f"Failed to extract audio: {result.stderr}")
                    
            logger.info(f"✅ Audio extracted to: {self._audio_file_path}")
            return self._audio_file_path
                
        except Exception as e:
            logger.error(f"Failed to extract audio: {str(e)}")
            raise

    def analyze_scenes(self):
        """Extract features for all scenes using hybrid approach"""
        logger.info("🔍 Analyzing scenes with hybrid approach...")
        features = []
        
        # Download video and audio files once
        video_path = self._download_video_file()
        audio_path = self._extract_audio_file()

        for scene in self.scenes:
            logger.info(f"Analyzing scene {scene['id']} ({scene['start']:.2f}s - {scene['end']:.2f}s)")
            
            # Calculate basic scene metrics
            duration = scene['end'] - scene['start']

            # Real motion analysis using OpenCV
            motion_score = self._analyze_motion_opencv(video_path, scene['start'], scene['end'])

            # Real audio analysis using librosa
            audio_energy = self._analyze_audio_librosa(audio_path, scene['start'], scene['end'])

            # Object/scene type estimation based on motion and audio characteristics
            objects = self._estimate_scene_objects_advanced(scene, motion_score, audio_energy)

            features.append({
                "scene_id": scene['id'],
                "motion": motion_score,
                "audio": audio_energy,
                "objects": objects,
                "duration": duration,
                "start": scene['start'],
                "end": scene['end']
            })

        return features
    
    def _analyze_motion_opencv(self, video_path, start_time, end_time):
        """Real motion analysis using OpenCV frame differencing and optical flow"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Calculate frame numbers
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)

            # Set to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            motion_values = []
            prev_frame = None

            frame_count = 0
            total_frames = end_frame - start_frame

            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)

                if prev_frame is not None:
                    # Calculate frame difference
                    diff = cv2.absdiff(prev_frame, gray)

                    # Threshold the difference
                    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

                    # Calculate motion as percentage of changed pixels
                    motion_pixels = cv2.countNonZero(thresh)
                    total_pixels = gray.shape[0] * gray.shape[1]
                    motion_ratio = motion_pixels / total_pixels

                    motion_values.append(motion_ratio)

                prev_frame = gray.copy()
                frame_count += 1

            cap.release()

            # Calculate average motion score
            if motion_values:
                avg_motion = np.mean(motion_values)
                # Normalize to 0-1 range (typical motion ratios are 0-0.3)
                normalized_motion = min(1.0, avg_motion * 3.0)
                return normalized_motion
            else:
                return 0.1  # Default low motion

        except Exception as e:
            logger.error(f"Motion analysis failed: {str(e)}")
            # Fallback to duration-based estimation
            duration = end_time - start_time
            return max(0.1, 1.0 - (duration / 10.0))
        
    def _analyze_audio_librosa(self, audio_path, start_time, end_time):
        """Real audio analysis using librosa for RMS energy and spectral features"""
        try:
            # Load audio segment
            y, sr = librosa.load(audio_path, offset=start_time, 
                               duration=end_time-start_time, sr=22050)
            
            if len(y) == 0:
                return 0.1
            
            # Calculate RMS energy
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            avg_rms = np.mean(rms)

            # Calculate spectral centroid (brightness)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            avg_centroid = np.mean(spectral_centroids)

            # Calculate zero crossing rate (roughness/noisiness)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            avg_zcr = np.mean(zcr)

            # Combine features into energy score
            # Normalize RMS (typical range 0-0.3)
            normalized_rms = min(1.0, avg_rms * 10.0)

            # Normalize centroid (typical range 0-8000 Hz)
            normalized_centroid = min(1.0, avg_centroid / 4000.0)

            # Normalize ZCR (typical range 0-0.5)
            normalized_zcr = min(1.0, avg_zcr * 2.0)

            # Weighted combination
            energy_score = (0.6 * normalized_rms + 
                          0.2 * normalized_centroid + 
                          0.2 * normalized_zcr)
            
            return min(1.0, energy_score)
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {str(e)}")
            # Fallback to random value
            return np.random.uniform(0.2, 0.6)

    def _estimate_scene_objects_advanced(self, scene, motion_score, audio_energy):
        """Advanced scene object estimation based on motion and audio analysis"""
        potential_objects = []

        # High motion indicators
        if motion_score > 0.6:
            potential_objects.extend(['high_motion', 'action'])
            if motion_score > 0.8:
                potential_objects.append('stunts')

        # Audio energy indicators
        if audio_energy > 0.7:
            potential_objects.extend(['crowd_reaction', 'loud_audio'])
        elif audio_energy < 0.3:
            potential_objects.extend(['quiet_scene', 'emotional'])

        # Duration-based indicators
        duration = scene['end'] - scene['start']
        if duration < 2.0:
            potential_objects.append('quick_cut')
        elif duration > 8.0:
            potential_objects.append('long_take')

        # Combine motion and audio for scene type detection
        combined_score = (motion_score + audio_energy) / 2
        if combined_score > 0.7:
            potential_objects.append('intense_scene')
        elif combined_score < 0.3:
            potential_objects.append('calm_scene')

        # Add requested scene types with higher probability
        for scene_type in self.edit_spec.get('scene_types', []):
            # Higher chance if motion/audio characteristics match
            if scene_type == 'action' and motion_score > 0.5:
                potential_objects.append(scene_type)
            elif scene_type == 'high_motion' and motion_score > 0.6:
                potential_objects.append(scene_type)
            elif scene_type in ['emotional', 'romantic'] and audio_energy < 0.5:
                potential_objects.append(scene_type)
            elif np.random.random() > 0.3:  # 70% chance for other types
                potential_objects.append(scene_type)

        # Add some generic objects
        generic_objects = ['person', 'movement', 'background']
        potential_objects.extend(np.random.choice(generic_objects, 
                                                size=np.random.randint(1, 3), 
                                                replace=False))

        return list(set(potential_objects))  # Remove duplicates

    
    def calculate_scene_score(self, scene_features):
        """Calculate score for a scene based on edit specifications"""
        score = 0

        # Apply object tag weights
        for tag in self.edit_spec.get('scene_types', []):
            if tag in scene_features['objects']:
                score += self.tag_weights.get(tag, self.tag_weights['default'])

        # Add motion and audio components with higher weights for action content
        motion_weight = 0.3 if 'action' in self.edit_spec.get('scene_types', []) else 0.2
        audio_weight = 0.3 if 'intense' in self.edit_spec.get('music_mood', '') else 0.2

        score += motion_weight * scene_features['motion']
        score += audio_weight * scene_features['audio']

        # Bonus for optimal duration (not too short, not too long)
        duration = scene_features['duration']
        if 2.0 <= duration <= 5.0:
            score += 0.15
        elif duration < 1.0:
            score -= 0.1
        elif duration > 8.0:
            score -= 0.05

        # Bonus for high-intensity scenes if requested
        if 'intense' in self.edit_spec.get('music_mood', ''):
            intensity = (scene_features['motion'] + scene_features['audio']) / 2
            if intensity > 0.7:
                score += 0.2

        return max(0, score)  # Ensure non-negative score
    
    def rank_scenes(self):
        """Rank scenes by their relevance score using hybrid analysis"""
        # Extract features using hybrid approach
        features = self.analyze_scenes()

        # Score all scenes
        scored_scenes = []
        for scene_feat in features:
            score = self.calculate_scene_score(scene_feat)
            scored_scenes.append({
                "id": scene_feat['scene_id'],
                "start": scene_feat['start'],
                "end": scene_feat['end'],
                "duration": scene_feat['duration'],
                "motion": scene_feat['motion'],
                "audio": scene_feat['audio'],
                "objects": scene_feat['objects'],
                "score": score
            })

        # Sort by score descending
        scored_scenes.sort(key=lambda x: x['score'], reverse=True)

        logger.info(f"🏆 Top 3 scenes: ")
        for i, scene in enumerate(scored_scenes[:3]):
            logger.info(f"  {i+1}. Scene {scene['id']}: Score={scene['score']:.3f}, "
                       f"Motion={scene['motion']:.3f}, Audio={scene['audio']:.3f}")
            
        return scored_scenes
    
    def select_scenes(self):
        """Select scenes to fit the desired duration using greedy algorithm"""
        ranked_scenes = self.rank_scenes()
        selected = []
        total_duration = 0
        target_duration = self.edit_spec.get('duration', 10)  # Default to 10 seconds

        logger.info(f"🎯 Target duration: {target_duration}s")
        logger.info(f"📊 Available scenes: {len(ranked_scenes)}")

        # Use greedy selection for simplicity
        for scene in ranked_scenes:
            if total_duration + scene['duration'] <= target_duration:
                selected.append(scene)
                total_duration += scene['duration']
                logger.info(f"✅ Selected scene {scene['id']}: {scene['duration']:.2f}s "
                          f"(Score: {scene['score']:.3f}, Motion: {scene['motion']:.3f}, "
                          f"Audio: {scene['audio']:.3f})")

            if total_duration >= target_duration * 0.9:  # Stop when we're close to target
                break

        # If we're still short, add partial scenes or smaller scenes
        if total_duration < target_duration and len(selected) < len(ranked_scenes):
            remaining = target_duration - total_duration
            for scene in ranked_scenes:
                if scene not in selected:
                    if scene['duration'] <= remaining * 1.5:  # Allow slightly over
                        # Adjust scene duration to fit
                        adjusted_scene = scene.copy()
                        adjusted_scene['duration'] = min(scene['duration'], remaining)
                        adjusted_scene['end'] = adjusted_scene['start'] + adjusted_scene['duration']
                        selected.append(adjusted_scene)
                        total_duration += adjusted_scene['duration']
                        logger.info(f"✅ Added adjusted scene {scene['id']}: {adjusted_scene['duration']:.2f}s")
                        break

        return {
            'selected_scenes': selected,
            'total_duration': total_duration,
            'target_duration': target_duration,
            'scene_count': len(selected),
            'used_scene_types': self.edit_spec.get('scene_types', []),
            'music_mood': self.edit_spec.get('music_mood', 'intense')
        }
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            if self._video_file_path and os.path.exists(self._video_file_path):
                os.remove(self._video_file_path)
                logger.info("🧹 Cleaned up video file")
            if self._audio_file_path and os.path.exists(self._audio_file_path):
                os.remove(self._audio_file_path)
                logger.info("🧹 Cleaned up audio file")
        except Exception as e:
            logger.warning(f"Cleanup warning: {str(e)}")

'''
def main_workflow(video_source, user_prompt):
    """Complete video processing workflow with hybrid analysis"""
    scorer = None
    try:
        # Step 1: Video ingestion and metadata extraction
        logger.info("🚀 Starting Step 1: Video Ingestion & Metadata Extraction")
        video_metadata = process_upload(video_source)

        # Step 2: Natural language prompt parsing
        logger.info("🔠 Starting Step 2: Prompt Parsing")
        parser = PromptParser()
        edit_spec = parser.parse_prompt(user_prompt)

        # Step 3: Scene selection and scoring with hybrid analysis
        logger.info("🎬 Starting Step 3: Hybrid Scene Analysis & Selection")
        scorer = VideoDBSceneScorer(video_metadata, edit_spec)

        # This will now use real OpenCV motion analysis and librosa audio analysis
        scene_selection = scorer.select_scenes()

        # Save final results
        output_file = "hybrid_analysis_result.json"
        with open(output_file, "w") as f:
            json.dump({
                "video_metadata": video_metadata,
                "edit_spec": edit_spec,
                "scene_selection": scene_selection,
                "analysis_method": "hybrid_opencv_librosa"
            }, f, indent=2, cls=NumpyEncoder)

        logger.info("🎉 Hybrid analysis completed successfully!")
        logger.info(f"💾 Results saved to: {output_file}")
        
        return scene_selection
    except Exception as e:
        logger.error(f"❌ Workflow failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        # Clean up temporary files
        if scorer:
            scorer.cleanup()

# User inputs
YOUTUBE_URL = 'https://www.youtube.com/watch?v=HluANRwPyNo'  # Short 15s video
USER_PROMPT = "Create a 10-second action-packed highlight reel with intense music"

try:

    # Run complete workflow with hybrid analysis
    print("\n🔥 Starting End-to-End Workflow 🔥")
    result = main_workflow(YOUTUBE_URL, USER_PROMPT)

    print("\n📊 Final Result:")
    print(json.dumps(result, indent=2, cls=NumpyEncoder))


except Exception as e:
    print(f"\n❌ Workflow failed: {str(e)}")  

# Print detailed results
print_detailed_results(result)
'''