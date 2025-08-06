import streamlit as st
import sys
import logging
import time

from full.workflow import main_workflow 

# Configure Streamlit page
st.set_page_config(
    page_title="AI Video Editor",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .header {
        color: #FF4B4B;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 20px;
    }
    .subheader {
        color: #1F51FF;
        border-bottom: 2px solid #1F51FF;
        padding-bottom: 5px;
        margin-top: 20px;
    }
    .success { color: #00C853; }
    .error { color: #FF5252; }
    .log-container {
        background-color: #0E1117;
        border-radius: 5px;
        padding: 15px;
        max-height: 400px;
        overflow-y: auto;
        margin-top: 10px;
    }
    .download-btn {
        background-color: #1F51FF !important;
        color: white !important;
        font-weight: bold !important;
        padding: 10px 20px !important;
        border-radius: 5px !important;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitLogger:
    """Custom logger to display logs in Streamlit UI"""
    def __init__(self, container):
        self.container = container
        self.logs = []
        self.lock = False

    def write(self, message):
        if message.strip():
            self.logs.append(message)
            # Update UI with new logs
            self.container.markdown(
                f'<div class="log-container">{"".join(self.logs[-50:])}</div>', 
                unsafe_allow_html=True
            )
    
    def flush(self):
        pass

    
def display_results(result):
    """Display processing results in Streamlit UI"""
    st.markdown("### 📊 Processing Results")

    if result['status'] == 'success':
        st.markdown(f"**Status:** <span class='success'>✅ {result['status']}</span>", unsafe_allow_html=True)
        st.markdown(f"**Source Video:** {result['video_metadata']['asset_id']}")
        st.markdown(f"**Selected Scenes:** {result['scene_selection']['scene_count']}")
        st.markdown(f"**Final Duration:** {result['editing_spec']['metadata']['actual_duration']:.1f}s")

        # Show download button if path exists
        if result['render_result'].get('download_path'):
            with open(result['render_result']['download_path'], 'rb') as f:
                st.download_button(
                    label='📥 Download Edited Video',
                    data=f,
                    file_name='edited_video.mp4',
                    mime='video/mp4',
                    use_container_width=True,
                    key='download_btn',
                    help='Click to download the edited video'
                )
        else:
            st.error("No video file generated for download")

    else:
        st.markdown(f"**Status:** <span class='error'>❌ {result['status']}</span>", unsafe_allow_html=True)
        st.error(f"Error: {result.get('error', 'Unknown error')}")


def main():
    st.markdown('<h1 class="header">🎬 AI Video Editor</h1>', unsafe_allow_html=True)
    st.markdown("Create professional video edits using natural language instructions")

    # Initialize session state
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False

    # Input section
    with st.form(key='input_form'):
        youtube_url = st.text_input(
            "YouTube URL", 
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste the URL of the YouTube video you want to edit"
        )

        user_prompt = st.text_area(
            "Editing Instructions",
            height=150,
            placeholder="Example: 'Create a 60-second action-packed trailer with fast transitions and epic music'",
            help="Describe what kind of edit you want to create"
        )
        
        submit_button = st.form_submit_button("✨ Start Editing", use_container_width=True)

        if submit_button:
            if not youtube_url or not user_prompt:
                st.error("Please provide both a YouTube URL and editing instructions")
            else:
                st.session_state.submitted = True
                st.session_state.youtube_url = youtube_url
                st.session_state.user_prompt = user_prompt
                st.session_state.processing = True
                st.session_state.result = None


    # Processing logic
    if st.session_state.submitted and st.session_state.processing:
                
        # Create log container
        log_container = st.empty()
        logger = StreamlitLogger(log_container)

        # Redirect stdout/stderr to capture logs
        sys.stdout = logger
        sys.stderr = logger

        # Start processing
        with st.spinner("🚀 Processing video - this may take several minutes..."):
            try:
                result = main_workflow(
                    youtube_url=st.session_state.youtube_url,
                    user_prompt=st.session_state.user_prompt,
                    create_download=True
                )
                st.session_state.result = result
            except Exception as e:
                st.error(f"Workflow failed: {str(e)}")
            finally:
                # Restore stdout/stderr
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
                st.session_state.processing = False

    # Show processing status
    if st.session_state.processing:
        st.info("⏳ Video processing in progress. Please wait...")

    # Show results when available
    if st.session_state.result and not st.session_state.processing:
        display_results(st.session_state.result)

        # Show processing times
        times = st.session_state.result.get('processing_time', {})
        if times:
            st.markdown("### ⏱️ Processing Times")
            cols = st.columns(5)
            with cols[0]:
                st.metric("Total Time", f"{times.get('total', 0):.1f}s")
            with cols[1]:
                st.metric("Video Ingestion", f"{times.get('step1', 0):.1f}s")
            with cols[2]:
                st.metric("Scene Selection", f"{times.get('step3', 0):.1f}s")
            with cols[3]:
                st.metric("Timeline Creation", f"{times.get('step4', 0):.1f}s")
            with cols[4]:
                st.metric("Rendering", f"{times.get('step6', 0):.1f}s")

if __name__ == "__main__":
    main()