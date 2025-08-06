import os
from dotenv import load_dotenv
from videodb import connect
import yt_dlp 
import time 
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def download_youtube_video(url):
    """Download YouTube videos using yt_dlp"""
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
        'outtmpl': 'temp/%(id)s.%(ext)s',
        'quiet': False,
        'noplaylist': True,
        'ignoreerrors': True,
        'no_warnings': False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # Handle cases where download fails and info is None
        if info:
            return ydl.prepare_filename(info)
        else:
            raise Exception("Failed to extract video info from yt_dlp.")

    
def wait_for_processing(coll, asset_id, timeout=300, interval=5):
    """Poll asset status until processing is complete by fetching the video object"""
    logger.info(f"⏳ Checking if asset {asset_id} is ready...")
    
    try:
        # Get the video object            
        asset = coll.get_video(asset_id)
        
        # Check if the video has the essential attributes
        if hasattr(asset, 'stream_url') and asset.stream_url and hasattr(asset, 'length'):
            logger.info("✅ Asset is ready!")
            logger.info(f"Stream URL: {asset.stream_url}")
            logger.info(f"Duration: {asset.length} seconds")
            return asset
        else:
            logger.warning("⚠️ Asset exists but may not be fully processed yet")
            return asset
            
    except Exception as e:
        logger.error(f"Error checking asset: {str(e)}")
        raise
            
def process_upload(upload_source):
    """Process upload with VideoDB integration"""
    # Get API key from environment
    api_key = os.getenv("VIDEO_DB_API_KEY")
    if not api_key:
        raise ValueError("VIDEO_DB_API_KEY environment variable not set")
    
    # Connect to VideoDB and get default collection
    videodb = connect(api_key=api_key)
    coll = videodb.get_collection()  # Get the default collection
    asset = None

    try:
        # Handle YouTube URLs separately
        if "youtube.com" in upload_source or "youtu.be" in upload_source:
            try:
                logger.info(f"Attempting to download YouTube video: {upload_source}")
                local_path = download_youtube_video(upload_source)
                logger.info(f"YouTube video downloaded to: {local_path}")
                asset = coll.upload(file_path=local_path) # Use collection for upload
                os.remove(local_path)  # Clean up temp file
            except Exception as e:
                logger.warning(f"⚠️ YouTube download failed: {str(e)}")
                logger.info("🔄 Trying direct VideoDB YouTube processing...")
                asset = coll.upload(url=upload_source)
        elif upload_source.startswith("s3://"):
            asset = coll.upload(url=upload_source) # Use collection for upload
        elif upload_source.startswith(("http://", "https://")):
            if ".m3u8" in upload_source:  # HLS stream
                asset = coll.ingest_stream(stream_url=upload_source)
            else:  # Direct URL
                asset = coll.upload(url=upload_source)
        else:  # Local file
            asset = coll.upload(file_path=upload_source)

        # Store asset ID immediately
        asset_id = asset.id
        logger.info(f"📦 Asset created: {asset_id}")

        # Wait for processing to complete - FIXED: Pass collection object, not videodb connection
        asset = wait_for_processing(coll, asset_id)

        # Trigger scene detection
        logger.info("🔍 Triggering scene detection...")
        scene_index_id = asset.index_scenes()
        logger.info(f"Scene indexing started with ID: {scene_index_id}")

        # Wait for scene detection to complete by polling the scene index
        logger.info("⏳ Waiting for scene detection to complete...")
        #indexed_scenes = asset.get_scene_index(scene_index_id) # Pass the scene_index_id
        # Add proper polling mechanism here
        max_wait_time = 600  # 10 minutes max (increased for longer videos)
        check_interval = 15  # Check every 10 seconds
        start_time = time.time()

        while True:
            try:
                # Try to get the scene index
                indexed_scenes = asset.get_scene_index(scene_index_id)
                logger.info("✅ Scene detection completed!")
                break
            except Exception as e:
                if "Index records does not exists" in str(e):
                    # Still processing, wait and retry
                    elapsed_time = time.time() - start_time
                    if elapsed_time > max_wait_time:
                        raise TimeoutError(f"Scene detection timed out after {max_wait_time} seconds")
                    
                    logger.info(f"⏳ Still processing... ({elapsed_time:.0f}s elapsed)")
                    time.sleep(check_interval)
                else:
                    # Different error, re-raise
                    raise

        # Get scene information
        scenes_data = []
        if indexed_scenes:
             for i, scene in enumerate(indexed_scenes):
                scenes_data.append({
                    "id": i,
                    "start": scene['start'],
                    "end": scene['end'],
                })
             logger.info(f"✅ Detected {len(scenes_data)} scenes.")
        else:
            # If no scenes detected, create a single scene for the entire video
            logger.warning("⚠️ No scenes were detected - using entire video as single scene.")
            scenes_data.append({
                "id": 0,
                "start": 0.0,
                "end": float(asset.length),
            })
            scene_index_id = None  # No scene index available

        # Note: VideoDB Video objects don't have update_metadata method
        # Scene data is available in the returned result
        logger.info("✅ Processing completed successfully!")
    
        return {
            "asset_id": asset.id,
            "duration": asset.length,
            "fps": getattr(asset, 'fps', None),
            "scenes": scenes_data,
            "video_path": asset.stream_url,
            "scene_index_id": scene_index_id
        }
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {str(e)}")
        if asset and hasattr(asset, 'id'):
            logger.info(f"Check asset status at: https://app.videodb.io/asset/{asset.id}")
        raise

# Test with a reliable YouTube video

# Test with a reliable YouTube video
#if __name__ == "__main__":
#    try:
#        # Use a short video for faster processing
#        result = process_upload('https://www.youtube.com/watch?v=HluANRwPyNo')  # Short 15s video
        
#        print("\n🎉 Processing successful!")
#        print(f"Asset ID: {result['asset_id']}")
#        print(f"Video URL: {result['video_path']}")
#        print(f"Duration: {result['duration']} seconds")
#        print(f"Scenes detected: {len(result['scenes'])}")
#        if result['scenes']:
#            print("First scene:", result['scenes'][0])
        
#    except Exception as e:
#        print(f"\n❌ Processing failed: {str(e)}")