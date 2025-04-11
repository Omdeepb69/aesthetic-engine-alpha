# Aesthetic Engine Alpha
# An agentic AI bot that autonomously generates and posts "hard images" to Twitter

import os
import random
import time
import logging
from datetime import datetime
import argparse
import json
import re
from typing import Dict, List, Tuple, Optional, Union

import tweepy
import torch
from PIL import Image
import requests
from dotenv import load_dotenv
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("aesthetic_engine.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("aesthetic_engine")

# Load environment variables
load_dotenv()

class AestheticConfig:
    """Configuration for the Aesthetic Engine"""
    
    def __init__(self, config_path: str = None):
        # Default configuration
        self.default_config = {
            "generation": {
                "frequency_hours": 3,
                "mode": "local",  # "local" or "api"
                "api_url": "https://api.stability.ai/v1/generation/stable-diffusion-v1-6/text-to-image",
                "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                "image_width": 768,
                "image_height": 768,
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "negative_prompt": "ugly, blurry, low quality, deformed, disfigured, watermark, signature, text"
            },
            "themes": [
                "cyberpunk",
                "vaporwave",
                "synthwave",
                "dystopian",
                "surreal",
                "cosmic",
                "retro",
                "futuristic",
                "hyper-realistic",
                "abstract"
            ],
            "subjects": [
                "animals doing human things",
                "robots in unexpected places",
                "bizarre architecture",
                "impossible landscapes",
                "retro-futuristic gadgets",
                "cosmic entities",
                "strange creatures",
                "dreamlike scenarios",
                "absurd combinations",
                "otherworldly scenes"
            ],
            "quality_keywords": [
                "8K resolution",
                "hyper detailed",
                "cinematic lighting",
                "HDR",
                "photorealistic",
                "award winning",
                "masterpiece",
                "sharp focus",
                "intricate details",
                "volumetric lighting"
            ],
            "twitter": {
                "post_caption": True,
                "include_hashtags": True,
                "max_hashtags": 5
            }
        }
        
        # Load custom config if provided
        self.config = self.default_config.copy()
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                # Update default config with custom values
                self._update_nested_dict(self.config, custom_config)
                logger.info(f"Loaded custom configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading custom config: {e}")
    
    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """Update nested dictionary with another dictionary's values"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = self._update_nested_dict(d[k].copy(), v)
            else:
                d[k] = v
        return d
    
    def get(self, key: str, default=None):
        """Get a config value using dot notation path"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


class PromptEngineering:
    """Generate optimized prompts for image generation"""
    
    def __init__(self, config: AestheticConfig):
        self.config = config
        
        # Load custom prompt elements if available
        self.themes = config.get("themes", [])
        self.subjects = config.get("subjects", [])
        self.quality_keywords = config.get("quality_keywords", [])
        
        # Gen Z aesthetic style modifiers
        self.aesthetic_styles = [
            "y2k aesthetic", "cottage core", "dark academia", "light academia",
            "cyber y2k", "grunge aesthetic", "kidcore", "dreamcore", 
            "weirdcore", "glitchcore", "liminal space", "mall goth",
            "indie sleaze", "goblincore", "royalcore", "fairycore",
            "nostalgiacore", "lofi aesthetic", "kawaii", "vaporwave"
        ]
        
        # Photography/art style modifiers
        self.art_styles = [
            "film photography", "analog photography", "polaroid", "vintage camera",
            "digital art", "concept art", "matte painting", "octane render",
            "unreal engine", "studio ghibli style", "trending on artstation",
            "greg rutkowski", "wes anderson symmetry", "blade runner palette"
        ]
        
        # Specific hard image elements
        self.hard_image_elements = [
            "chrome metal", "neon lights", "cosmic energy", "holographic",
            "hyperrealistic textures", "lens flare", "atmospheric haze",
            "dramatic shadows", "ray tracing", "iridescent", "glossy",
            "glow effects", "cybernetic enhancements", "retro-futuristic"
        ]
        
        # Unusual subjects that make for good "hard images"
        self.unusual_subjects = [
            "capybara in spacesuit", "frog with cybernetic implants", 
            "raccoon hacker", "sloth astronaut", "cat CEO", "robot picnic",
            "dinosaur using smartphone", "cosmic jellyfish", "cyborg plants",
            "sentient crystals", "floating islands", "mechanical butterflies",
            "digital wilderness", "time-traveling tourists", "urban dragon"
        ]
    
    def generate_prompt(self) -> str:
        """Generate a complete prompt for image generation"""
        # Choose basic components
        theme = random.choice(self.themes)
        subject = random.choice(self.subjects + self.unusual_subjects)
        aesthetic = random.choice(self.aesthetic_styles)
        art_style = random.choice(self.art_styles)
        
        # Add 2-4 quality keywords
        quality_terms = random.sample(self.quality_keywords, k=random.randint(2, 4))
        
        # Add 1-2 hard image elements
        hard_elements = random.sample(self.hard_image_elements, k=random.randint(1, 2))
        
        # Combine everything with a 70% chance of including each optional component
        components = [subject]
        
        if random.random() < 0.7:
            components.append(f"with {theme} theme")
        
        if random.random() < 0.7:
            components.append(aesthetic)
            
        if random.random() < 0.7:
            components.append(art_style)
        
        # Always add quality and hard elements
        components.extend(quality_terms)
        components.extend(hard_elements)
        
        # Construct the final prompt
        prompt = ", ".join(components)
        
        # 30% chance to add a dramatic descriptor at the beginning
        dramatic_starters = [
            "Epic", "Mind-blowing", "Breathtaking", "Stunning", 
            "Incredible", "Magnificent", "Ultra-detailed"
        ]
        
        if random.random() < 0.3:
            prompt = f"{random.choice(dramatic_starters)} {prompt}"
            
        logger.info(f"Generated prompt: {prompt}")
        return prompt
    
    def generate_simple_prompt(self) -> str:
        """Generate a simplified prompt focusing on a clear subject"""
        subject = random.choice(self.unusual_subjects)
        style = random.choice(self.art_styles)
        quality = random.choice(self.quality_keywords)
        
        prompt = f"{subject}, {style}, {quality}"
        logger.info(f"Generated simple prompt: {prompt}")
        return prompt


class CaptionGenerator:
    """Generate Gen Z style captions for images"""
    
    def __init__(self):
        # Gen Z slang and expressions
        self.slang_phrases = [
            "vibe check: passed", "absolutely slaps", "this hits different",
            "living rent free in my mind", "no cap", "certified fresh",
            "main character energy", "that's the tea", "big mood",
            "it's giving...", "not me obsessing over", "rent free",
            "understood the assignment", "iykyk", "cooked", "based",
            "unhinged (in a good way)", "core memory", "ate and left no crumbs",
            "zero notes", "stays true to itself", "extremely my vibe",
            "a whole aesthetic", "period.", "we're so back", "real",
            "energy is immaculate", "just dropped", "goes hard"
        ]
        
        # Caption starters
        self.starters = [
            "when the", "that moment when", "pov:", "me when", 
            "literally just", "imagine", "woke up and chose", 
            "not a want but a need", "the way", "this is so",
            "", "", "", ""  # Empty strings to sometimes have no starter
        ]
        
        # Emoji sets
        self.emoji_sets = [
            "💯🔥", "✨👁️👄👁️✨", "😤🙌", "🥶🥵", "👑💅", 
            "🤌✨", "😳👉👈", "💀🖤", "🌈🌊", "🤯🚀",
            "🧠💭", "👽🌌", "🐸☕", "🤠🤙", "⚡🔋",
            "🌟🔮", "🖤⛓️", "🎭🎪", "🌱🍄", "🕳️🐇"
        ]
        
        # Hashtags
        self.hashtags = [
            "aesthetic", "vibe", "fyp", "genalpha", "xyzbca", 
            "hardimage", "aiart", "digitalart", "aiesthetic", "corecore",
            "nofilter", "nocontext", "liminalspaces", "surrealism", "dreamscape",
            "vibecheck", "instadaily", "cursedimages", "blessedtimeline", "timelineshift",
            "maincharacter", "altreality", "dreamcore", "weirdcore", "dimensionhopping"
        ]
        
        # Opinion phrases to insert occasionally
        self.opinions = [
            "can't believe this exists", "need this framed on my wall",
            "would die for this", "obsessed with this energy",
            "this aesthetic is everything", "new personality just dropped",
            "switching my entire personality to this", "make it my lockscreen immediately",
            "might get this tattooed", "this is so real", "completely unserious",
            "actually my brain 24/7"
        ]

    def extract_subjects_from_prompt(self, prompt: str) -> List[str]:
        """Extract potential subjects from the prompt to mention in caption"""
        # Simple extraction by looking for specific patterns or nouns
        subjects = []
        
        # Look for objects followed by actions/descriptions
        patterns = [
            r'(\w+\s\w+)\sin\s\w+',  # "cat in spacesuit"
            r'(\w+)\s(\w+ing)',      # "frog swimming"
            r'(\w+)\swith\s(\w+)',    # "raccoon with laptop"
            r'(\w+)\s(\w+)'          # "robot picnic"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, prompt.lower())
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        subjects.extend([m for m in match if len(m) > 3])
                    else:
                        subjects.append(match)
        
        # If nothing found, split by commas and take first chunk
        if not subjects and ',' in prompt:
            subjects = [prompt.split(',')[0].strip()]
            
        return subjects

    def generate_caption(self, prompt: str) -> Tuple[str, List[str]]:
        """Generate a caption with optional hashtags based on the prompt"""
        # Extract potential subjects from the prompt
        subjects = self.extract_subjects_from_prompt(prompt)
        subject_text = random.choice(subjects) if subjects else "this"
        
        # Construct caption
        starter = random.choice(self.starters)
        slang = random.choice(self.slang_phrases)
        emoji = random.choice(self.emoji_sets)
        
        # 30% chance to include an opinion
        opinion = ""
        if random.random() < 0.3:
            opinion = f" {random.choice(self.opinions)}"
        
        # Build the caption
        if starter:
            caption = f"{starter} {subject_text} {slang}{opinion} {emoji}"
        else:
            caption = f"{subject_text} {slang}{opinion} {emoji}"
            
        # Clean up any double spaces
        caption = re.sub(r'\s{2,}', ' ', caption).strip()
        
        # Generate hashtags (3-5)
        num_hashtags = random.randint(3, 5)
        
        # Start with theme-related hashtags from the prompt
        prompt_words = set(re.findall(r'\b\w{4,}\b', prompt.lower()))
        theme_hashtags = [word for word in prompt_words 
                         if word not in ['with', 'using', 'from', 'like', 'style', 'very', 'hyper', 'ultra'] 
                         and len(word) > 3]
        
        # Combine with general hashtags and select final set
        all_hashtags = theme_hashtags + self.hashtags
        selected_hashtags = random.sample(all_hashtags, min(num_hashtags, len(all_hashtags)))
        
        logger.info(f"Generated caption: {caption}")
        logger.info(f"Selected hashtags: {selected_hashtags}")
        
        return caption, selected_hashtags


class ImageGenerator:
    """Generate images using either local models or API"""
    
    def __init__(self, config: AestheticConfig):
        self.config = config
        self.mode = config.get("generation.mode", "local")
        self.api_url = config.get("generation.api_url", "")
        self.model_id = config.get("generation.model_id", "stabilityai/stable-diffusion-xl-base-1.0")
        self.width = config.get("generation.image_width", 768)
        self.height = config.get("generation.image_height", 768)
        self.steps = config.get("generation.num_inference_steps", 30)
        self.guidance_scale = config.get("generation.guidance_scale", 7.5)
        self.negative_prompt = config.get("generation.negative_prompt", "")
        
        # Initialize the model if using local mode
        if self.mode == "local":
            logger.info(f"Loading local model: {self.model_id}")
            self._init_local_model()
        else:
            logger.info(f"Using API mode with endpoint: {self.api_url}")
            # Verify API key existence
            if not os.getenv("STABILITY_API_KEY") and "stability.ai" in self.api_url:
                logger.warning("STABILITY_API_KEY not found in environment variables")
    
    def _init_local_model(self):
        """Initialize the local Stable Diffusion model"""
        try:
            # Use DPM-Solver++ for faster inference
            scheduler = DPMSolverMultistepScheduler.from_pretrained(
                self.model_id, 
                subfolder="scheduler"
            )
            
            # Initialize the pipeline
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                scheduler=scheduler,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.pipe = self.pipe.to("cuda")
                logger.info("Using GPU for image generation")
            else:
                logger.info("No GPU available, using CPU (this will be slow)")
                
            # Enable memory efficient attention if using PyTorch 2.0+
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                self.pipe.enable_xformers_memory_efficient_attention()
                
        except Exception as e:
            logger.error(f"Failed to initialize local model: {e}")
            raise
    
    def generate_image(self, prompt: str) -> Optional[Image.Image]:
        """Generate an image based on the prompt"""
        if self.mode == "local":
            return self._generate_local(prompt)
        else:
            return self._generate_api(prompt)
    
    def _generate_local(self, prompt: str) -> Optional[Image.Image]:
        """Generate image using local model"""
        try:
            # Generate the image
            logger.info(f"Generating image with local model...")
            
            result = self.pipe(
                prompt=prompt,
                negative_prompt=self.negative_prompt,
                width=self.width,
                height=self.height,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance_scale
            )
            
            # Check for safety issues (just in case)
            if hasattr(result, "nsfw_content_detected") and result.nsfw_content_detected and result.nsfw_content_detected[0]:
                logger.warning("NSFW content detected, image generation failed")
                return None
                
            # Get the image
            image = result.images[0]
            logger.info("Local image generation successful")
            return image
            
        except Exception as e:
            logger.error(f"Error generating image locally: {e}")
            return None
    
    def _generate_api(self, prompt: str) -> Optional[Image.Image]:
        """Generate image using Stability AI API"""
        try:
            api_key = os.getenv("STABILITY_API_KEY")
            if not api_key:
                logger.error("STABILITY_API_KEY not found in environment variables")
                return None
                
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            payload = {
                "text_prompts": [
                    {
                        "text": prompt,
                        "weight": 1.0
                    },
                    {
                        "text": self.negative_prompt,
                        "weight": -1.0
                    }
                ],
                "cfg_scale": self.guidance_scale,
                "height": self.height,
                "width": self.width,
                "samples": 1,
                "steps": self.steps
            }
            
            logger.info(f"Requesting image generation from API...")
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                return None
                
            data = response.json()
            
            # Extract image data from response
            if "artifacts" in data and len(data["artifacts"]) > 0:
                image_data = data["artifacts"][0]["base64"]
                import base64
                from io import BytesIO
                
                image = Image.open(BytesIO(base64.b64decode(image_data)))
                logger.info("API image generation successful")
                return image
            else:
                logger.error("No image data in API response")
                return None
                
        except Exception as e:
            logger.error(f"Error generating image via API: {e}")
            return None
    
    def save_image(self, image: Image.Image, output_dir: str = "output") -> Optional[str]:
        """Save the generated image to disk"""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename based on timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_dir}/aesthetic_engine_{timestamp}.png"
            
            # Save the image
            image.save(filename)
            logger.info(f"Image saved to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return None


class TwitterPoster:
    """Post images and captions to Twitter"""
    
    def __init__(self):
        # Check for environment variables
        required_vars = [
            "TWITTER_API_KEY", 
            "TWITTER_API_SECRET",
            "TWITTER_ACCESS_TOKEN", 
            "TWITTER_ACCESS_SECRET"
        ]
        
        for var in required_vars:
            if not os.getenv(var):
                logger.warning(f"Missing environment variable: {var}")
        
        # Initialize the Twitter API client
        try:
            self.client = tweepy.Client(
                consumer_key=os.getenv("TWITTER_API_KEY"),
                consumer_secret=os.getenv("TWITTER_API_SECRET"),
                access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
                access_token_secret=os.getenv("TWITTER_ACCESS_SECRET")
            )
            
            # Initialize API v1.1 for media upload
            auth = tweepy.OAuth1UserHandler(
                os.getenv("TWITTER_API_KEY"),
                os.getenv("TWITTER_API_SECRET"),
                os.getenv("TWITTER_ACCESS_TOKEN"),
                os.getenv("TWITTER_ACCESS_SECRET")
            )
            self.api = tweepy.API(auth)
            
            logger.info("Twitter client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Twitter client: {e}")
            self.client = None
            self.api = None
    
    def post_image(self, image_path: str, caption: str, hashtags: List[str] = None) -> bool:
        """Post an image to Twitter with caption and hashtags"""
        if not self.client or not self.api:
            logger.error("Twitter client not initialized")
            return False
            
        try:
            # Format the tweet text
            tweet_text = caption
            
            # Add hashtags if provided
            if hashtags:
                hashtag_text = " ".join([f"#{tag}" for tag in hashtags])
                tweet_text = f"{tweet_text}\n\n{hashtag_text}"
                
            # Upload the media
            media = self.api.media_upload(image_path)
            media_id = media.media_id
            
            # Post the tweet
            self.client.create_tweet(
                text=tweet_text,
                media_ids=[media_id]
            )
            
            logger.info(f"Tweet posted successfully with image: {image_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error posting to Twitter: {e}")
            return False


class AestheticEngine:
    """Main class that coordinates the entire process"""
    
    def __init__(self, config_path: str = None):
        self.config = AestheticConfig(config_path)
        self.prompt_engineer = PromptEngineering(self.config)
        self.caption_generator = CaptionGenerator()
        self.image_generator = ImageGenerator(self.config)
        self.twitter_poster = TwitterPoster()
        
        # Create output directory
        os.makedirs("output", exist_ok=True)
        
    def run_once(self) -> bool:
        """Run a single generation and posting cycle"""
        try:
            # 1. Generate the prompt
            prompt = self.prompt_engineer.generate_prompt()
            
            # 2. Generate the image
            image = self.image_generator.generate_image(prompt)
            if not image:
                logger.error("Failed to generate image")
                return False
                
            # 3. Save the image
            image_path = self.image_generator.save_image(image)
            if not image_path:
                logger.error("Failed to save image")
                return False
                
            # 4. Generate caption and hashtags
            caption, hashtags = self.caption_generator.generate_caption(prompt)
            
            # 5. Post to Twitter if enabled
            if self.config.get("twitter.post_caption", True):
                success = self.twitter_poster.post_image(image_path, caption, hashtags)
                if not success:
                    logger.warning("Failed to post to Twitter, but image was generated")
            else:
                logger.info("Twitter posting disabled in config")
                
            # Save metadata for reference
            self._save_metadata(prompt, caption, hashtags, image_path)
            
            logger.info("Run completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in run cycle: {e}")
            return False
    
    def _save_metadata(self, prompt: str, caption: str, hashtags: List[str], image_path: str):
        """Save metadata about the generation for reference"""
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "caption": caption,
            "hashtags": hashtags,
            "image_path": image_path
        }
        
        # Get base filename from image path
        base_name = os.path.splitext(image_path)[0]
        metadata_path = f"{base_name}_metadata.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
    
    def run_continuously(self):
        """Run the engine continuously based on configured frequency"""
        frequency_hours = self.config.get("generation.frequency_hours", 3)
        sleep_seconds = frequency_hours * 60 * 60
        
        logger.info(f"Starting continuous operation, generating every {frequency_hours} hours")
        
        while True:
            success = self.run_once()
            
            if success:
                logger.info(f"Sleeping for {frequency_hours} hours until next generation")
            else:
                # If failed, try again in 30 minutes
                logger.info("Generation failed, will retry in 30 minutes")
                sleep_seconds = 30 * 60
                
            time.sleep(sleep_seconds)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Aesthetic Engine Alpha - AI Image Generator Bot")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--continuous", action="store_true", help="Run continuously based on frequency in config")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save generated images")
    
    args = parser.parse_args()
    
    # Initialize and run the engine
    engine = AestheticEngine(args.config)
    
    if args.continuous:
        engine.run_continuously()
    else:
        engine.run_once()


if __name__ == "__main__":
    main()
