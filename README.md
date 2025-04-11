# Aesthetic Engine Alpha

## Description
An agentic AI bot that autonomously generates unique, culturally resonant 'hard images' using optimized prompts and posts them to Twitter with Gen Z-infused captions and relevant hashtags. It's coded to understand the assignment: creating visuals that slap.

## Features
- Automated text-to-image generation using Stable Diffusion (via diffusers) or an API (e.g., Stability AI) focused on creating surreal, cool, or absurdly epic visuals ('hard images').
- Dynamic prompt engineering incorporating keywords for high quality (e.g., 'cinematic lighting', 'hyperrealistic', '8k') and specific themes (e.g., 'cat piloting fighter jet', 'capybara DJ', 'grizzly bear on Harley in space').
- Culturally-aware caption generation using rule-based templates or a simple LLM, incorporating Gen Z slang and subtle meme references (e.g., 'vibe check', 'standing on business', 'cooked aesthetic').
- Automated tweeting of the generated image and caption using the Twitter API v2.
- Configuration for controlling generation frequency and basic content themes.

## Learning Benefits
Gain hands-on experience with agentic AI concepts, advanced prompt engineering for text-to-image models, integrating multiple APIs (Twitter, AI models), social media automation, working with generative AI locally or via cloud, and translating cultural trends into AI instructions.

## Technologies Used
- tweepy (for Twitter API v2 interaction)
- diffusers (Hugging Face library for Stable Diffusion)
- transformers (Hugging Face library, needed by diffusers)
- torch (PyTorch, deep learning framework)
- python-dotenv (for managing API keys securely)
- Pillow (for image manipulation if needed)
- requests (if using external image generation APIs instead of local diffusers)

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/Omdeepb69/aesthetic-engine-alpha.git
cd aesthetic-engine-alpha

# Install dependencies
pip install -r requirements.txt
```

## Usage
[Instructions on how to use the project]

## Project Structure
[Brief explanation of the project structure]

## License
MIT
