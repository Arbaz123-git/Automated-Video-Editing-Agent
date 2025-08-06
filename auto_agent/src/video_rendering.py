import os
import logging
from videodb import connect
from videodb.asset import VideoAsset, AudioAsset
from videodb.timeline import Timeline
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class VideoDBRenderer:
    def __init__(self, editing_spec, video_metadata):
        """
        Initialize with editing specification and video metadata
        :param editing_spec: Complete editing specification
        :param video_metadata: Video metadata from Step 1
        """
        self.editing_spec = editing_spec
        self.video_metadata = video_metadata

        # Connect to VideoDB
        api_key = os.getenv("VIDEO_DB_API_KEY")
        if not api_key:
            raise ValueError("VIDEO_DB_API_KEY environment variable not set")
        
        self.videodb = connect(api_key=api_key)
        self.timeline = Timeline(self.videodb)

    def build_timeline(self):
        """Convert editing specification to VideoDB Timeline with Assets"""
        try:
            asset_id = self.video_metadata["asset_id"]
            logger.info(f"🎬 Building timeline for asset: {asset_id}")

            # Process timeline items
            video_clips = []
            overlay_items = []

            for item in self.editing_spec["timeline"]:
                if item["type"] == "clip":
                    # Create VideoAsset for each clip
                    video_asset = VideoAsset(
                        asset_id=asset_id,
                        start=item["start"],
                        end=item["end"]
                    )
                    video_clips.append(video_asset)
                    logger.info(f"   📹 Added video clip: {item['start']:.2f}s - {item['end']:.2f}s")
                elif item["type"] == "transition":
                    # VideoDB handles transitions automatically between clips
                    # We can add fade effects or other transition logic here if needed
                    logger.info(f"   🔄 Transition: {item.get('style', 'default')}")

            # Add video clips sequentially to timeline
            for video_asset in video_clips:
                self.timeline.add_inline(video_asset)

             # Handle enhanced audio if available
            if self.editing_spec.get("audio_enhancements") and "enhanced_audio_path" in self.editing_spec:
                try:
                    # Upload enhanced audio to VideoDB first
                    enhanced_audio = self.videodb.upload(
                        file_path=self.editing_spec["enhanced_audio_path"]
                    )

                    # Create AudioAsset for the enhanced audio
                    audio_asset = AudioAsset(
                        asset_id=enhanced_audio.id,
                        start=0,
                        end=self.editing_spec["metadata"]["actual_duration"],
                        disable_other_tracks=True,  # Replace original audio
                        fade_in_duration=0.1,
                        fade_out_duration=0.1
                    )

                    # Add as overlay at the beginning
                    self.timeline.add_overlay(0, audio_asset)
                    logger.info("   🎵 Added enhanced audio track")

                except Exception as e:
                    logger.warning(f"Failed to add enhanced audio: {str(e)}")

            # Add audio overlays for music or sound effects if specified
            if "audio_overlays" in self.editing_spec:
                for overlay in self.editing_spec["audio_overlays"]:
                    try:
                        audio_asset = AudioAsset(
                            asset_id=overlay["asset_id"],
                            start=overlay.get("start", 0),
                            end=overlay.get("end", overlay.get("duration", 10)),
                            disable_other_tracks=overlay.get("replace_audio", False),
                            fade_in_duration=overlay.get("fade_in", 0.2),
                            fade_out_duration=overlay.get("fade_out", 0.2)
                        )
                        
                        self.timeline.add_overlay(overlay["timeline_position"], audio_asset)
                        logger.info(f"   🎶 Added audio overlay at {overlay['timeline_position']}s")

                    except Exception as e:
                        logger.warning(f"Failed to add audio overlay: {str(e)}")
            
            logger.info(f"✅ Timeline built with {len(video_clips)} video clips")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build timeline: {str(e)}")
            raise

    def render(self):
        """Generate video stream using VideoDB Timeline"""
        try:
            logger.info("🎬 Starting video compilation...")
            
            # Build the timeline
            self.build_timeline()

            # Generate stream URL
            logger.info("🔄 Generating stream URL...")
            stream_url = self.timeline.generate_stream()

            logger.info(f"✅ Stream generated successfully!")

            return {
                "status": "success",
                "output_url": stream_url,
                "stream_url": stream_url,
                "message": "Video compilation completed successfully",
                "timeline_items": len(self.editing_spec["timeline"]),
                "total_duration": self.editing_spec["metadata"]["actual_duration"]
            }
        
        except Exception as e:
            logger.error(f"Rendering failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Render exception: {str(e)}",
                "error_type": type(e).__name__
            }
        
    
    def create_downloadable_version(self):
        """
        Create a downloadable MP4 version of the compiled video
        Note: This might require additional VideoDB features or external processing
        """
        try: 
            # First generate the stream
            result = self.render()

            if result["status"] != "success":
                return result
            
            stream_url = result["stream_url"]

            # For now, return the stream URL
            # In a production environment, you might want to:
            # 1. Download the stream using ffmpeg
            # 2. Upload to a file storage service
            # 3. Return a permanent download link

            logger.info("📦 Stream URL can be used for playback or download")

            return {
                "status": "success",
                "download_url": stream_url,  # This is actually a stream URL
                "stream_url": stream_url,
                "message": "Use stream URL for playback. For permanent download, additional processing needed.",
                "format": "HLS Stream",
                "note": "This is a streaming URL, not a direct MP4 download"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Download creation failed: {str(e)}"
            }

# Enhanced version with download capability
class VideoDBRendererWithDownload(VideoDBRenderer):
    """Extended renderer that can create downloadable MP4 files"""
    def __init__(self, editing_spec, video_metadata, temp_dir=None):
        super().__init__(editing_spec, video_metadata)
        self.temp_dir = temp_dir or os.path.join(os.getcwd(), "temp_renders")
        os.makedirs(self.temp_dir, exist_ok=True)

    def download_stream_as_mp4(self, stream_url, output_path):
        """Download HLS stream as MP4 using ffmpeg"""
        import subprocess

        try:
            cmd = [
                'ffmpeg', '-i', stream_url,
                '-c', 'copy', '-y', output_path
            ]

            logger.info(f"🔄 Converting stream to MP4: {output_path}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"✅ MP4 created successfully: {output_path}")
                return True
            else:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False
            
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg timeout - video too long or slow connection")
            return False
        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            return False
        
    def render_with_download(self):
        """Render video and create downloadable MP4"""
        try:
            # First create the stream
            stream_result = self.render()
            
            if stream_result["status"] != "success":
                return stream_result
            
            # Generate output filename
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"compiled_video_{timestamp}.mp4"
            output_path = os.path.join(self.temp_dir, output_filename)

            # Download stream as MP4
            if self.download_stream_as_mp4(stream_result["stream_url"], output_path):
                return {
                    "status": "success",
                    "output_url": stream_result["stream_url"],
                    "download_path": output_path,
                    "download_filename": output_filename,
                    "stream_url": stream_result["stream_url"],
                    "message": "Video compiled and MP4 file created",
                    "file_size": os.path.getsize(output_path) if os.path.exists(output_path) else 0
                }
            else:
                # Return stream result even if download failed
                stream_result["message"] += " (MP4 download failed, but stream available)"
                stream_result["download_error"] = "Failed to create MP4 file"
                return stream_result
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Render with download failed: {str(e)}"
            }

