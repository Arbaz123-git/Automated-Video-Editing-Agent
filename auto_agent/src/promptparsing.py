import re
import json
import os
from groq import Groq
from dotenv import load_dotenv 

# Load environment variables from .env file
load_dotenv()

class PromptParser:
    def __init__(self, model_name="llama3-70b-8192"):
        # Verify API key is loaded
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables. "
                             "Please check your .env file")
        
        self.client = Groq(api_key=api_key)  # Use the loaded API key
        self.model = model_name
        self.defaults = {
            "duration": 30,
            "scene_types": [],
            "transition_style": "hard_cut",
            "music_mood": "neutral",
            "tuning": {}
        }

    def extract_duration_fallback(self, prompt):
        """Fallback duration extraction using regex"""
        pattern = r"(\d+)\s*(?:sec|second|s\b|min|minute|m\b)?"
        matches = re.findall(pattern, prompt)
        # Convert all found numbers to seconds
        seconds = 0
        for val in matches:
            num = int(val)
            # If value is less than 10, assume minutes (e.g., "2m")
            if num < 10 and "min" in prompt.lower():
                seconds += num * 60
            else:
                seconds += num
                
        return seconds if seconds > 0 else self.defaults["duration"]
    
    def parse_prompt(self, user_prompt):
        """Parse natural language prompt using GROQ LLM"""
        system_prompt = """
        You are a video editing specification generator. Extract:
        1. Duration in seconds (default: 30)
        2. Primary scene types (comma-separated)
        3. Transition style (default: hard_cut)
        4. Music mood (default: neutral)
        5. Special instructions

        Return JSON format only:
        {
          "duration": 30,
          "scene_types": ["action"],
          "transition_style": "quick_fade",
          "music_mood": "intense",
          "tuning": {}
        }
        """
        try:
            # Call GROQ API
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=256
            )

            # Extract and validate JSON
            json_str = response.choices[0].message.content
            spec = json.loads(json_str)

            # Validate required fields
            for key in self.defaults:
                if key not in spec:
                    spec[key] = self.defaults[key]

            # Ensure scene_types is a list
            if isinstance(spec["scene_types"], str):
                spec["scene_types"] = [s.strip() for s in spec["scene_types"].split(",")]
                
            return spec
        
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"⚠️ LLM parsing failed: {str(e)} - Using fallback extraction")
            return self.fallback_parsing(user_prompt)
        except Exception as e:
            print(f"🚨 GROQ API error: {str(e)}")
            return self.fallback_parsing(user_prompt)
        
    def fallback_parsing(self, prompt):
        """Fallback parsing when LLM fails"""
        spec = self.defaults.copy()
        spec["duration"] = self.extract_duration_fallback(prompt)

        # Scene type detection
        scene_keywords = {
            "action": ["action", "fight", "explosion", "chase"],
            "romantic": ["romantic", "love", "couple", "kiss"],
            "sports": ["sports", "goal", "match", "game", "soccer"],
            "landscape": ["landscape", "nature", "scenic", "view"],
            "comedy": ["funny", "comedy", "laugh", "joke"]
        }
        
        for scene_type, keywords in scene_keywords.items():
            if any(kw in prompt.lower() for kw in keywords):
                spec["scene_types"].append(scene_type)

        # Transition style detection
        if "soft" in prompt.lower() or "fade" in prompt.lower():
            spec["transition_style"] = "soft_fade"
        elif "quick" in prompt.lower() or "fast" in prompt.lower():
            spec["transition_style"] = "quick_cut"

        # Music mood detection
        mood_keywords = {
            "epic": ["epic", "grand", "heroic"],
            "emotional": ["emotional", "sentimental", "romantic", "dramatic"],
            "energetic": ["energetic", "intense", "pumping", "upbeat"]
        }
        for mood, keywords in mood_keywords.items():
            if any(kw in prompt.lower() for kw in keywords):
                spec["music_mood"] = mood
                break
       
        return spec

# Test function with error handling
#def test_parser():
#    parser = PromptParser()

#    test_prompts = [
#        "Make a 45-second highlight reel of the soccer match with intense moments",
#        "Create romantic montage about 25 seconds with soft transitions",
#        "Quick 15s action sequence compilation",
#        "Show me the best parts in a minute"
#    ]
#    for prompt in test_prompts:
#        print(f"\n{'='*50}")
#        print(f"🔹 Prompt: '{prompt}'")
#        try:
#            spec = parser.parse_prompt(prompt)
#            print("✅ Parsed Specification:")
#            print(json.dumps(spec, indent=2))
#        except Exception as e:
#            print(f"❌ Error processing prompt: {str(e)}")
    
#test_parser()

