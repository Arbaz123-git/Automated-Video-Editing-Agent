class TransitionPlanner:
    def __init__(self, scene_selection, edit_spec):
        """
        Initialize with scene selection and editing specifications
        :param scene_selection: Output from SceneScorer
        :param edit_spec: Parsed specifications from Step 2
        """
        self.scene_selection = scene_selection
        self.edit_spec = edit_spec

        # SDK transition mapping with VideoDB-compatible presets
        self.transition_presets = {
            "quick_fade": {"sdk_preset": "FADE_CROSS", "duration": 0.3},
            "hard_cut": {"sdk_preset": "CUT_IMMEDIATE", "duration": 0.0},
            "cinematic": {"sdk_preset": "FADE_DIP_TO_BLACK", "duration": 0.5},
            "dynamic": {"sdk_preset": "SWIPE_RIGHT", "duration": 0.4}
        }

    def calculate_cuts(self, scenes, target_duration):
        """Adjust scene durations proportionally to fit target duration"""
        total_raw = sum(s["duration"] for s in scenes)
        if total_raw == 0:
            return scenes
        ratio = min(1, target_duration / total_raw)

        adjusted = []
        for scene in scenes:
            adj_duration = scene["duration"] * ratio
            adjusted.append({
                **scene,
                "duration": adj_duration,
                "end": scene["start"] + adj_duration  # Adjust end time
            })

        return adjusted
    
    def create_timeline(self):
        """Create a timeline with scenes and transitions"""
        selected_scenes = self.scene_selection["selected_scenes"]
        target_duration = self.scene_selection["target_duration"]

        # Apply proportional duration adjustment
        adjusted_scenes = self.calculate_cuts(selected_scenes, target_duration)

        # Get transition settings
        transition_style = self.edit_spec.get("transition_style", "hard_cut")

        transition_cfg = self.transition_presets.get(
            transition_style, 
            self.transition_presets["hard_cut"]
        )

        timeline = []
        total_duration = 0

        # Add first scene
        if adjusted_scenes:
            first_scene = adjusted_scenes[0]
            timeline.append({
                "type": "clip",
                "scene_id": first_scene["id"],
                "start": first_scene["start"],
                "end": first_scene["end"],
                "duration": first_scene["duration"],
                "sdk_params": {
                    "type": "VIDEO_SEGMENT",
                    "start_sec": first_scene["start"],
                    "end_sec": first_scene["end"]
                }
            })
            total_duration += first_scene["duration"]

        # Add transitions and subsequent scenes
        for i in range(1, len(adjusted_scenes)):
            prev_scene = adjusted_scenes[i-1]
            curr_scene = adjusted_scenes[i]

            # Add transition
            if transition_cfg["duration"] > 0:
                timeline.append({
                    "type": "transition",
                    "effect": transition_cfg["sdk_preset"],
                    "duration": transition_cfg["duration"],
                    "sdk_params": {
                        "type": "TRANSITION",
                        "preset": transition_cfg["sdk_preset"],
                        "duration_ms": int(transition_cfg["duration"] * 1000)
                    }
                })
                total_duration += transition_cfg["duration"]

            # Add scene
            timeline.append({
                "type": "clip",
                "scene_id": curr_scene["id"],
                "start": curr_scene["start"],
                "end": curr_scene["end"],
                "duration": curr_scene["duration"],
                "sdk_params": {
                    "type": "VIDEO_SEGMENT",
                    "start_sec": curr_scene["start"],
                    "end_sec": curr_scene["end"]
                }
            })
            total_duration += curr_scene["duration"]

        return timeline, total_duration
    
    def generate_editing_spec(self):
        """Generate complete editing specification for VideoDB SDK"""
        timeline, total_duration = self.create_timeline()

        # Get transition style from edit_spec or use default
        transition_style = self.edit_spec.get("transition_style", "hard_cut")

        return {
            "metadata": {
                "source_asset": self.scene_selection.get("asset_id", ""),
                "target_duration": self.scene_selection["target_duration"],
                "actual_duration": total_duration,
                "scene_count": len(self.scene_selection["selected_scenes"])
                },
            "timeline": timeline,
            "output_config": {
                "resolution": "1080p",
                "frame_rate": 30,
                "codec": "h264",
                "audio_mix": {
                    "background_music": self.edit_spec.get("music_mood", "intense"),
                    "original_audio_level": 0.7
                }
            },
            "enhancements": self.edit_spec.get("tuning", {}),
            "transition_style": transition_style  # Add this key
        }
