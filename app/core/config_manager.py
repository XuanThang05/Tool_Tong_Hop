import json
import os

CONFIG_FILE = "data/settings.json"

class ConfigManager:
    def __init__(self):
        self.config_path = os.path.join(os.getcwd(), CONFIG_FILE)
        self.default_config = {
            "pipeline": {
                "scraper": True,
                "plagiarism_check": True,
                "gemini_clean": True,
                "gemini_meta": True,
                "tts": True,
                "waveform": True,
                "video_assemble": True
            },
            "telegram": {
                "bot_token": "",
                "chat_id": ""
            },
            "plagiarism": {
                "similarity_threshold": 80
            },
            "gemini": {
                "api_keys": [],
                "prompt_cleanup": "Clean up this story, correct spelling, and improve flow:\n\n",
                "prompt_metadata": "Generate a title and short description for this story:\n\n"
            }
        }
        self.config = self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_path):
            self.save_config(self.default_config)
            return self.default_config
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return self.default_config

    def save_config(self, new_config=None):
        if new_config:
            self.config = new_config
        
        # Ensure data dir exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving config: {e}")
            
    def get(self, section, key=None):
        if key:
            return self.config.get(section, {}).get(key)
        return self.config.get(section)

    def set(self, section, key, value):
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self.save_config()
