from datetime import datetime  # Correct import for datetime.now()
import logging 
import json
import traceback 

from src.videoprocessing import process_upload
from src.promptparsing import PromptParser
from src.scene_scorer import VideoDBSceneScorer, NumpyEncoder
from src.transition_planning import TransitionPlanner
from src.audio_enhancing import AudioEnhancer
from src.video_rendering import VideoDBRenderer, VideoDBRendererWithDownload

def main_workflow(youtube_url: str, user_prompt: str, create_download: bool = True) -> dict:
    """
    End-to-end video editing workflow from YouTube URL to timeline specification
    Args:
        youtube_url: YouTube video URL to process
        user_prompt: Natural language editing instructions
        create_download: Whether to create downloadable MP4 (requires ffmpeg)
    Returns:
        Dictionary with processing results including editing specification
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"video_edit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    result = {
        "status": "success",
        "video_metadata": None,
        "edit_spec": None,
        "scene_selection": None,
        "editing_spec": None,
        "render_result": None,
        "processing_time": {
            "step1": None,
            "step2": None,
            "step3": None,
            "step4": None,
            "step5": None,
            "step6": None,
            "total": None
        },
        "error": None
    }
    start_time = datetime.now()
    scorer = None

    try:
        # Step 1: Video ingestion and metadata extraction
        logger.info("\n" + "="*50)
        logger.info("🚀 Step 1: Video Ingestion & Metadata Extraction")
        step1_start = datetime.now()

        # Step 1: Video ingestion and metadata extraction
        logger.info("\n" + "="*50)
        logger.info("🚀 Step 1: Video Ingestion & Metadata Extraction")
        step1_start = datetime.now()
        video_metadata = process_upload(youtube_url)
        result["video_metadata"] = {
            "asset_id": video_metadata['asset_id'],
            "duration": video_metadata['duration'],
            "scene_count": len(video_metadata.get('scenes', [])),
            "stream_url": video_metadata.get('stream_url')
        }
        result["processing_time"]["step1"] = (datetime.now() - step1_start).total_seconds()
        logger.info(f"✅ Video processed in {result['processing_time']['step1']:.1f}s! "
                  f"{result['video_metadata']['scene_count']} scenes detected")
        
        # Step 2: Natural language prompt parsing
        logger.info("\n" + "="*50)
        logger.info("💬 Step 2: Natural-Language Prompt Parsing")
        step2_start = datetime.now()

        parser = PromptParser()
        edit_spec = parser.parse_prompt(user_prompt)
        result["edit_spec"] = edit_spec
        result["processing_time"]["step2"] = (datetime.now() - step2_start).total_seconds()
        logger.info(f"✅ Prompt parsed in {result['processing_time']['step2']:.1f}s! "
                  f"Target: {edit_spec.get('duration', 'N/A')}s "
                  f"Scene types: {edit_spec.get('scene_types', [])}")
        
        # Step 3: Scene selection with hybrid analysis
        logger.info("\n" + "="*50)
        logger.info("🎯 Step 3: AI-Powered Scene Selection")
        step3_start = datetime.now()
        scorer = VideoDBSceneScorer(video_metadata, edit_spec)
        selection_result = scorer.select_scenes()
        result["scene_selection"] = {
            "scenes": [{"id": s['id'], "start": s['start'], "end": s['end']} 
                      for s in selection_result['selected_scenes']],
            "total_duration": selection_result['total_duration'],
            "target_duration": selection_result['target_duration'],
            "scene_count": len(selection_result['selected_scenes']),
            "used_scene_types": selection_result.get('used_scene_types', []),
            "music_mood": selection_result.get('music_mood', '')
        }
        result["processing_time"]["step3"] = (datetime.now() - step3_start).total_seconds()
        logger.info(f"✅ Selected {result['scene_selection']['scene_count']} scenes in "
                  f"{result['processing_time']['step3']:.1f}s "
                  f"({result['scene_selection']['total_duration']:.1f}s total)")
        
        # Step 4: Transition planning & timeline assembly
        logger.info("\n" + "="*50)
        logger.info("🎬 Step 4: Transition Planning & Timeline Assembly")
        step4_start = datetime.now()
        transition_planner = TransitionPlanner(selection_result, edit_spec)
        editing_spec = transition_planner.generate_editing_spec()

        result["editing_spec"] = editing_spec
        result["processing_time"]["step4"] = (datetime.now() - step4_start).total_seconds()
        
        # Log timeline summary
        clip_count = sum(1 for item in editing_spec["timeline"] if item["type"] == "clip")
        transition_count = sum(1 for item in editing_spec["timeline"] if item["type"] == "transition")
        logger.info(f"✅ Timeline created in {result['processing_time']['step4']:.1f}s with:")
        logger.info(f"   - {clip_count} video clips")
        logger.info(f"   - {transition_count} transitions")
        logger.info(f"   - Total runtime: {editing_spec['metadata']['actual_duration']:.2f}s")
        logger.info(f"   - Transition style: {editing_spec['transition_style']}")

        # Step 5: Audio Enhancement (Simplified)
        logger.info("\n" + "="*50)
        logger.info("🔊 Step 5: Original Audio Enhancement")
        step5_start = datetime.now()

        try:
            # Get audio path from scene scorer
            audio_path = scorer._audio_file_path if scorer else None

            # Initialize and run audio enhancement
            audio_enhancer = AudioEnhancer(result["editing_spec"], audio_path)
            editing_spec = audio_enhancer.integrate()
            result["editing_spec"] = editing_spec
            result["processing_time"]["step5"] = (datetime.now() - step5_start).total_seconds()

            logger.info(f"✅ Audio enhanced in {result['processing_time']['step5']:.1f}s")
            logger.info(f"   - Original audio preserved and enhanced")
        except Exception as e:
            logger.error(f"Audio enhancement failed: {str(e)}")
            # Continue with original audio if enhancement fails
            result["editing_spec"]["audio_track"] = audio_path
            result["editing_spec"]["audio_enhancements"] = False
            result["processing_time"]["step5"] = (datetime.now() - step5_start).total_seconds()

        # Step 6: Video Rendering with Fixed VideoDB API
        logger.info("\n" + "="*50)
        logger.info("🎞️ Step 6: Video Rendering")
        step6_start = datetime.now()

        try:
            # Choose renderer based on download requirement
            if create_download:
                renderer = VideoDBRendererWithDownload(result["editing_spec"], result["video_metadata"])
                render_result = renderer.render_with_download()
            else:
                renderer = VideoDBRenderer(result["editing_spec"], result["video_metadata"])
                render_result = renderer.render()

            result["render_result"] = render_result
            result["processing_time"]["step6"] = (datetime.now() - step6_start).total_seconds()

            if render_result["status"] == "success":
                logger.info(f"✅ Video rendered in {result['processing_time']['step6']:.1f}s!")
                logger.info(f"   - Stream URL: {render_result['output_url']}")
                
                if "download_path" in render_result:
                    logger.info(f"   - Download Path: {render_result['download_path']}")
                    logger.info(f"   - File Size: {render_result.get('file_size', 0) / 1024 / 1024:.1f} MB")
                else:
                    logger.info("   - Use stream URL for playback")

            else:
                logger.error(f"❌ Rendering failed: {render_result['message']}")
                result["status"] = "error"
                result["error"] = render_result["message"]

        except Exception as e:
            logger.error(f"Rendering failed: {str(e)}")
            result["status"] = "error"
            result["error"] = str(e)
            result["processing_time"]["step6"] = (datetime.now() - step6_start).total_seconds()

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error(f"\n❌ Workflow failed: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        if scorer:
                scorer.cleanup()
        total_time = (datetime.now() - start_time).total_seconds()
        result["processing_time"]["total"] = total_time
        logger.info(f"\n🏁 Total processing time: {total_time:.1f} seconds")
        logger.info("="*50)
    return result

''' 
YOUTUBE_URL = "https://www.youtube.com/watch?v=6SGRn9OHtFY"  # Test video
USER_PROMPT = "Make a 30-second emotional highlight reel"

print("\n🔥 Starting End-to-End AI Video Editing Workflow 🔥")

def print_result(final_result):
    """Print workflow results in a user-friendly format"""
    print("\n📊 Final Result Summary:")
    print(f"Status: {final_result['status']}")

    if final_result['status'] == "success":
        print(f"\n🔹 Source Video: {YOUTUBE_URL}")
        print(f"🔹 Processed Asset: {final_result['video_metadata']['asset_id']}")
        print(f"🔹 Selected Scenes: {final_result['scene_selection']['scene_count']}")
        print(f"🔹 Timeline Duration: {final_result['editing_spec']['metadata']['actual_duration']:.1f}s")
        print(f"🔹 Audio Enhancement: {'✅ Applied' if final_result['editing_spec'].get('audio_enhancements') else '❌ Not applied'}")

        # Print rendering results
        render_result = final_result.get("render_result")
        if render_result and render_result["status"] == "success":
            print("\n🎬 RENDERED VIDEO:")
            print(f"   - Stream URL: {render_result['output_url']}")
            
            if "download_path" in render_result:
                print(f"   - Download Path: {render_result['download_path']}")
                print(f"   - File Size: {render_result.get('file_size', 0) / 1024 / 1024:.1f} MB")
            
            print(f"   - Job ID: {render_result.get('job_id', 'N/A')}")

        # Save full spec to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"editing_spec_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(final_result, f, indent=2, cls=NumpyEncoder)
        print(f"\n💾 Full specification saved to: {filename}")
    else:
        print(f"\n❌ Error: {final_result['error']}")
        if "render_result" in final_result:
            print(f"   - Render Job ID: {final_result['render_result'].get('job_id', 'N/A')}")
            print(f"   - Render Error: {final_result['render_result'].get('message', 'Unknown error')}")

# Option 1: Create downloadable video (requires ffmpeg)
print("\n=== Rendering with Download ===")
final_result_download = main_workflow(YOUTUBE_URL, USER_PROMPT, create_download=True)
print_result(final_result_download)

# Option 2: Streaming only (faster)
print("\n=== Rendering Stream Only ===")
final_result_stream = main_workflow(YOUTUBE_URL, USER_PROMPT, create_download=False)
print_result(final_result_stream)


'''   
