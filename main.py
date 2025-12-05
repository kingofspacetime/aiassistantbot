import os
from flask import Flask, request, jsonify
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from typing import Final
import asyncio
import threading
import random
import schedule
import time
import aiohttp
import json
import logging
from PIL import Image
import io
import base64
import imageio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
BOT_TOKEN: Final = os.getenv('BOT_TOKEN')
GROQ_API_KEY: Final = os.getenv('GROQ_API_KEY')
GEMINI_API_KEY: Final = os.getenv('GEMINI_API_KEY')  # Add this to your environment variables
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
GROQ_MODEL = "llama-3.1-8b-instant"
PORT: Final = int(os.getenv('PORT', '8080'))

# Create Flask app
app = Flask(__name__)
bot_app = None

# Load character data from JSON file
def load_characters():
    """Load character data from JSON file"""
    try:
        with open('characters.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("characters.json file not found!")
        return {
            "anime_characters": [],
            "manhwa_characters": [],
            "movie_characters": [],
            "tv_series_characters": [],
            "real_life_people": []
        }
    except json.JSONDecodeError:
        logger.error("Invalid JSON in characters.json file!")
        return {
            "anime_characters": [],
            "manhwa_characters": [],
            "movie_characters": [],
            "tv_series_characters": [],
            "real_life_people": []
        }

# Load character data
character_data = load_characters()
anime_characters = character_data.get("anime_characters", [])
manhwa_characters = character_data.get("manhwa_characters", [])
movie_characters = character_data.get("movie_characters", [])
tv_series_characters = character_data.get("tv_series_characters", [])
real_life_people = character_data.get("real_life_people", [])

# Combined list of all characters/people for random selection
all_characters = anime_characters + manhwa_characters + movie_characters + tv_series_characters + real_life_people

# Print character counts for verification
logger.info(f"Loaded characters:")
logger.info(f"  Anime: {len(anime_characters)}")
logger.info(f"  Manhwa: {len(manhwa_characters)}")
logger.info(f"  Movies: {len(movie_characters)}")
logger.info(f"  TV Series: {len(tv_series_characters)}")
logger.info(f"  Real Life: {len(real_life_people)}")
logger.info(f"  Total: {len(all_characters)}")

# Store group chats for scheduled messages - Use a thread-safe set
group_chats = set()
group_chats_lock = threading.Lock()

def get_random_character():
    """Get a random character from all lists"""
    if not all_characters:
        return "Anonymous Sage"
    return random.choice(all_characters)

def get_character_category(character_name):
    """Get the category of a character"""
    if character_name in anime_characters:
        return "Anime"
    elif character_name in manhwa_characters:
        return "Manhwa"
    elif character_name in movie_characters:
        return "Movie"
    elif character_name in tv_series_characters:
        return "TV Series"
    elif character_name in real_life_people:
        return "Real Life"
    else:
        return "Unknown"


async def extract_frame_from_webm(webm_bytes):
    """Extract first frame from WebM video using imageio"""
    try:
        import tempfile
        import os
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
            temp_file.write(webm_bytes)
            temp_path = temp_file.name
        
        try:
            # Read video with imageio
            reader = imageio.get_reader(temp_path, 'ffmpeg')
            
            # Get first frame
            first_frame = reader.get_data(0)  # Get frame at index 0
            
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(first_frame)
            
            # Save as PNG bytes
            png_buffer = io.BytesIO()
            pil_image.save(png_buffer, format='PNG', optimize=True)
            png_bytes = png_buffer.getvalue()
            
            # Cleanup
            reader.close()
            
            logger.info(f"ImageIO: Extracted frame from WebM ({len(webm_bytes)} bytes) to PNG ({len(png_bytes)} bytes)")
            return png_bytes
            
        finally:
            # Always cleanup temp file
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Could not delete temp file: {e}")
                
    except Exception as e:
        logger.error(f"ImageIO frame extraction failed: {e}")
        raise Exception("Could not extract frame from animated sticker")


async def extract_frame_from_tgs(tgs_bytes):
    """Extract first frame from TGS (Lottie) animation"""
    try:
        import json
        import tempfile
        import os
        import gzip
        from lottie import objects
        from lottie.exporters.cairo import export_png
        
        # Create temporary file for TGS
        with tempfile.NamedTemporaryFile(suffix='.tgs', delete=False) as temp_file:
            temp_file.write(tgs_bytes)
            temp_path = temp_file.name
        
        try:
            # Parse TGS (it's gzipped JSON)
            with gzip.open(temp_path, 'rt', encoding='utf-8') as f:
                lottie_data = json.load(f)
            
            # Create Lottie animation object
            animation = objects.Animation.load(lottie_data)
            
            # Create temporary PNG file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as png_file:
                png_path = png_file.name
            
            # Export first frame as PNG using Cairo exporter
            export_png(animation, png_path, 512, 512, frame=0)
            
            # Read the rendered PNG
            with open(png_path, 'rb') as f:
                png_bytes = f.read()
            
            logger.info(f"Lottie: Extracted frame from TGS ({len(tgs_bytes)} bytes) to PNG ({len(png_bytes)} bytes)")
            return png_bytes
            
        finally:
            # Cleanup temp files
            try:
                os.unlink(temp_path)
                if 'png_path' in locals():
                    os.unlink(png_path)
            except Exception as e:
                logger.warning(f"Could not delete temp files: {e}")
                
    except Exception as e:
        logger.error(f"Lottie frame extraction failed: {e}")
        raise Exception("Could not extract frame from TGS animation")



async def convert_webp_to_png(image_bytes, is_animated=False):
    """Convert WebP image to PNG format, with basic animated support"""
    try:
        # Open the image with PIL
        image = Image.open(io.BytesIO(image_bytes))
        
        # For animated images, try to get the first frame
        if is_animated and hasattr(image, 'seek') and hasattr(image, 'n_frames'):
            try:
                image.seek(0)  # Go to first frame
                logger.info(f"Extracted first frame from animated image with {getattr(image, 'n_frames', 'unknown')} frames")
            except Exception as e:
                logger.warning(f"Could not seek to first frame: {e}")
        
        # Convert to RGB if necessary (WebP can have different modes)
        if image.mode in ('RGBA', 'LA', 'P'):
            # Convert to RGB
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            rgb_image.paste(image, mask=image.split()[-1] if len(image.split()) == 4 else None)
            image = rgb_image
        elif image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        # Save as PNG
        png_buffer = io.BytesIO()
        image.save(png_buffer, format='PNG', optimize=True)
        png_bytes = png_buffer.getvalue()
        
        logger.info(f"Converted {'animated ' if is_animated else ''}WebP ({len(image_bytes)} bytes) to PNG ({len(png_bytes)} bytes)")
        return png_bytes
        
    except Exception as e:
        logger.error(f"Error converting WebP to PNG: {e}")
        return image_bytes  # Return original if conversion fails


async def convert_gif_to_png(image_bytes):
    """Convert GIF to PNG format by extracting first frame"""
    try:
        # Open the GIF with PIL
        image = Image.open(io.BytesIO(image_bytes))
        
        # Get the first frame
        image.seek(0)
        
        # Convert to RGB if necessary
        if image.mode in ('RGBA', 'LA', 'P'):
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            rgb_image.paste(image, mask=image.split()[-1] if len(image.split()) == 4 else None)
            image = rgb_image
        elif image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        # Save as PNG
        png_buffer = io.BytesIO()
        image.save(png_buffer, format='PNG', optimize=True)
        png_bytes = png_buffer.getvalue()
        
        logger.info(f"Converted GIF ({len(image_bytes)} bytes) to PNG ({len(png_bytes)} bytes)")
        return png_bytes
        
    except Exception as e:
        logger.error(f"Error converting GIF to PNG: {e}")
        raise Exception("Could not convert GIF to PNG")

async def extract_text_with_ocr_space(image_bytes, mime_type="image/png"):
    """Extract text using OCR.Space API as fallback"""
    try:
        # OCR.Space API endpoint
        url = "https://api.ocr.space/parse/image"
        
        # Prepare the image data
        files = {
            'file': ('image.png', image_bytes, mime_type)
        }
        
        data = {
            'apikey': 'helloworld',  # Free API key
            'language': 'eng',
            'isOverlayRequired': False,
            'detectOrientation': True,
            'scale': True,
            'OCREngine': 2  # Use engine 2 for better accuracy
        }
        
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Create form data manually
            form_data = aiohttp.FormData()
            form_data.add_field('file', image_bytes, filename='image.png', content_type=mime_type)
            for key, value in data.items():
                form_data.add_field(key, str(value))
                
            async with session.post(url, data=form_data) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get('IsErroredOnProcessing', False):
                        return "❌ OCR processing failed"
                    
                    # Extract text from response
                    text_results = []
                    for parsed_result in result.get('ParsedResults', []):
                        text = parsed_result.get('ParsedText', '').strip()
                        if text:
                            text_results.append(text)
                    
                    if text_results:
                        return f"English: {' '.join(text_results)}"
                    else:
                        return "No text found in image"
                else:
                    return f"❌ OCR service error: {response.status}"
                    
    except Exception as e:
        logger.error(f"OCR.Space API error: {e}")
        return "❌ OCR service temporarily unavailable"




async def extract_text_from_image(image_bytes, mime_type="image/png", is_animated=False):
    """Extract and translate text from image using Google Gemini"""
    if not GEMINI_API_KEY:
        return "❌ Gemini API key not configured. Please set GEMINI_API_KEY environment variable."
    
    try:
        logger.info(f"Processing: mime_type={mime_type}, is_animated={is_animated}, size={len(image_bytes)}")
        
        
        # Handle GIF files
        if mime_type == "image/gif":
            try:
                logger.info("Processing GIF with PIL")
                image_bytes = await convert_gif_to_png(image_bytes)
                mime_type = "image/png"
            except Exception as gif_error:
                logger.error(f"GIF processing failed: {gif_error}")
                return "❌ Could not process GIF format"

                # Handle TGS (Lottie animated stickers)
        elif mime_type == "application/x-tgsticker":
            try:
                logger.info("Processing TGS (Lottie) sticker")
                image_bytes = await extract_frame_from_tgs(image_bytes)
                mime_type = "image/png"
            except Exception as tgs_error:
                logger.error(f"TGS processing failed: {tgs_error}")
                return "❌ Could not process TGS animated sticker. The animation format is not supported yet."
                
        # Handle WebM (animated stickers)     
        if mime_type == "video/webm":
            try:
                logger.info("Processing animated sticker (WebM) with imageio")
                image_bytes = await extract_frame_from_webm(image_bytes)
                mime_type = "image/png"
            except Exception as webm_error:
                logger.error(f"WebM processing failed: {webm_error}")
                return "❌ Could not process animated sticker. Please try a static sticker! 🙂"
        
        # Handle WebP (static stickers) 
        elif mime_type == "image/webp":
            try:
                logger.info("Processing static sticker (WebP)")
                image_bytes = await convert_webp_to_png(image_bytes, is_animated)
                mime_type = "image/png"
            except Exception as webp_error:
                logger.error(f"WebP processing failed: {webp_error}")
                return "❌ Could not process sticker format"
        
        # Validate processed image
        if len(image_bytes) == 0:
            return "❌ Processed image is empty"
        
        # Check file size limit
        if len(image_bytes) > 20 * 1024 * 1024:
            return "❌ Image is too large for processing"
        
        logger.info(f"Sending to Gemini: {len(image_bytes)} bytes as {mime_type}")
        
        # Rest of your Gemini API code stays the same...
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        payload = {
            "contents": [{
                "parts": [
                    {
                        "text": "Please extract ALL text from this image and translate it to English. If the text is already in English, just return the extracted text. If there's no text in the image, respond with 'No text found in image'. Format your response as: 'Original: [original text] | English: [translated text]' or just 'English: [text]' if it's already in English."
                    },
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": image_base64
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 1000
            }
        }
        
        headers = {"Content-Type": "application/json"}
        url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
        timeout = aiohttp.ClientTimeout(total=60)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                logger.info(f"Gemini API Status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    if 'candidates' in data and len(data['candidates']) > 0:
                        candidate = data['candidates'][0]
                        if 'content' in candidate and 'parts' in candidate['content']:
                            text = candidate['content']['parts'][0]['text'].strip()
                            return text
                        else:
                            return "❌ Invalid response from translation service"
                    else:
                        return "❌ Could not extract text from the image"
                else:
                    error_text = await response.text()
                    logger.error(f"Gemini API Error {response.status}: {error_text}")
                    
                    # Try OCR.Space as fallback for any Gemini failure
                    logger.info(f"Gemini failed with status {response.status}, trying OCR.Space fallback")
                    return await extract_text_with_ocr_space(image_bytes, mime_type)
                    
    except Exception as e:
        logger.error(f"Error in extract_text_from_image: {e}")
        return f"❌ Error processing image: {str(e)}"

async def generate_quote_with_groq(character_name):
    """Generate a positive quote using Groq API"""
    logger.info(f"generate_quote_with_groq called for: {character_name}")
    
    try:
        if not GROQ_API_KEY:
            logger.warning("GROQ_API_KEY not found! Using fallback quotes.")
            return get_fallback_quote()
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""Generate a short, positive, and inspirational quote that {character_name} might say or that represents their philosophy. 
        The quote should be:
        - Motivational and uplifting
        - 1-2 sentences maximum
        - In the style or spirit of {character_name}
        - Family-friendly and appropriate for all ages
        - Focused on themes like perseverance, friendship, hope, courage, or personal growth
        
        Format the response as just the quote without any additional text or quotation marks.
        
        Character: {character_name}"""
        
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": GROQ_MODEL,
            "stream": False,
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(GROQ_API_URL, headers=headers, json=payload) as response:
                logger.info(f"GROQ API Status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    quote = data["choices"][0]["message"]["content"].strip()
                    logger.info(f"GROQ Response: {quote}")
                    return quote
                else:
                    error_text = await response.text()
                    logger.error(f"GROQ API Error {response.status}: {error_text}")
                    return get_fallback_quote()
                    
    except Exception as e:
        logger.error(f"GROQ API Exception: {str(e)}")
        return get_fallback_quote()

def get_fallback_quote():
    """Get a fallback quote when API fails"""
    fallback_quotes = [
        "Never give up on your dreams, no matter how hard the journey gets.",
        "The strength to keep going comes from believing in yourself.",
        "Every small step forward is progress worth celebrating.",
        "True courage isn't the absence of fear, but acting despite it.",
        "The bonds we forge with others make us stronger than we ever imagined.",
        "Success is not final, failure is not fatal: it is the courage to continue that counts.",
        "Believe in yourself and all that you are. You are capable of amazing things.",
        "The only way to do great work is to love what you do.",
        "In the middle of difficulty lies opportunity.",
        "Your limitation is only your imagination."
    ]
    return random.choice(fallback_quotes)

async def send_quote_to_chat(chat_id, bot_instance):
    """Send a quote to a specific chat"""
    try:
        character = get_random_character()
        category = get_character_category(character)
        quote = await generate_quote_with_groq(character)
        
        # Add category emoji
        category_emoji = {
            "Anime": "🎌",
            "Manhwa": "🇰🇷", 
            "Movie": "🎬",
            "TV Series": "📺",
            "Real Life": "🌟"
        }.get(category, "✨")
        
        message = f"💫 *Quote of the Moment* 💫\n\n\"{quote}\"\n\n— _{character}_ {category_emoji}\n\n🌟 _Keep shining!_ 🌟"
        
        await bot_instance.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode='Markdown'
        )
        logger.info(f"Scheduled quote sent to chat {chat_id}")
        return True
    except Exception as e:
        logger.error(f"Error sending quote to chat {chat_id}: {e}")
        # Check if it's a chat-related error (removed from group, etc.)
        if "chat not found" in str(e).lower() or "blocked" in str(e).lower() or "kicked" in str(e).lower():
            return False
        return True


async def transcribe_and_translate_voice(voice_bytes):
    """Transcribe voice message and translate to English using Groq API"""
    if not GROQ_API_KEY:
        return "❌ Groq API key not configured. Please set GROQ_API_KEY environment variable."
    
    try:
        # Groq Whisper API endpoint
        whisper_url = "https://api.groq.com/openai/v1/audio/transcriptions"
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        
        # Prepare the audio file for upload
        files = {
            'file': ('voice.ogg', voice_bytes, 'audio/ogg'),
            'model': (None, 'whisper-large-v3'),
            'response_format': (None, 'json'),
            'language': (None, 'auto')  # Auto-detect language
        }
        
        timeout = aiohttp.ClientTimeout(total=60)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Create form data
            form_data = aiohttp.FormData()
            form_data.add_field('file', voice_bytes, filename='voice.ogg', content_type='audio/ogg')
            form_data.add_field('model', 'whisper-large-v3')
            form_data.add_field('response_format', 'json')
            
            async with session.post(whisper_url, headers=headers, data=form_data) as response:
                logger.info(f"Groq Whisper API Status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    transcribed_text = data.get('text', '').strip()
                    
                    if not transcribed_text:
                        return "❌ No speech detected in the voice message"
                    
                    logger.info(f"Transcribed text: {transcribed_text}")
                    
                    # Now translate to English using the same Groq API
                    if await is_english(transcribed_text):
                        return f"🎤 *Voice Transcription* 🎤\n\nEnglish: {transcribed_text}"
                    else:
                        translated_text = await translate_to_english(transcribed_text)
                        return f"🎤 *Voice Translation* 🎤\nEnglish Translation: {translated_text}"
                        
                else:
                    error_text = await response.text()
                    logger.error(f"Groq Whisper API Error {response.status}: {error_text}")
                    return f"❌ Voice transcription failed: {response.status}"
                    
    except Exception as e:
        logger.error(f"Error in transcribe_and_translate_voice: {e}")
        return f"❌ Error processing voice message: {str(e)}"

async def is_english(text):
    """Simple check if text is likely English"""
    # Basic check - if most words are common English words
    english_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their'}
    words = text.lower().split()
    if len(words) == 0:
        return True
    english_count = sum(1 for word in words if word in english_words)
    return (english_count / len(words)) > 0.3

async def translate_to_english(text):
    """Translate text to English using Groq API"""
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""Translate the following text to English. If it's already in English, just return the original text.

Text: {text}

Translation:"""
        
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": GROQ_MODEL,
            "stream": False,
            "temperature": 0.3,
            "max_tokens": 500
        }
        
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(GROQ_API_URL, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    translation = data["choices"][0]["message"]["content"].strip()
                    return translation
                else:
                    logger.error(f"Translation API Error {response.status}")
                    return text  # Return original if translation fails
                    
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text

async def translate_voice_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Translate voice messages using /tv command"""
    try:
        # Check if this is a reply to a voice message
        if not update.message.reply_to_message or not update.message.reply_to_message.voice:
            await update.message.reply_text(
                "🎤 Please reply to a voice message with /tv to translate it!\n\n"
                "Usage: Reply to any voice message and type /tv"
            )
            return
        
        # Send typing action
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
        
        # Get the voice message
        voice = update.message.reply_to_message.voice
        
        # Send processing message
        processing_msg = await update.message.reply_text(
            "🎤 Transcribing voice message...", 
            reply_to_message_id=update.message.reply_to_message.message_id
        )
        
        try:
            # Get the file
            file = await context.bot.get_file(voice.file_id)
            
            # Download the file
            voice_bytes = await file.download_as_bytearray()
            
            logger.info(f"Processing voice message: {len(voice_bytes)} bytes, duration: {voice.duration}s")
            
            # Check file size (Groq has limits)
            if len(voice_bytes) > 25 * 1024 * 1024:  # 25MB limit
                await processing_msg.edit_text("❌ Voice message is too large. Please send a shorter message.")
                return
            
            # Transcribe and translate
            result = await transcribe_and_translate_voice(voice_bytes)
            
            # Edit the processing message with the result
            try:
                await processing_msg.edit_text(result, parse_mode='Markdown')
                logger.info(f"Voice translation sent to user {update.message.from_user.id}")
            except Exception as edit_error:
                logger.warning(f"Could not edit processing message: {edit_error}")
                await update.message.reply_text(
                    result, 
                    parse_mode='Markdown',
                    reply_to_message_id=update.message.reply_to_message.message_id
                )
                try:
                    await processing_msg.delete()
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Error processing voice message: {e}")
            error_response = "❌ Sorry, I couldn't process this voice message. Please try again or send a clearer recording."
            try:
                await processing_msg.edit_text(error_response)
            except:
                await update.message.reply_text(error_response)
            
    except Exception as e:
        logger.error(f"Error in translate_voice_command: {e}")
        await update.message.reply_text(
            "❌ Sorry, something went wrong while processing the voice message. Please try again later!"
        )



async def scheduled_quotes_job():
    """Send quotes to all registered group chats"""
    if not bot_app or not bot_app.bot:
        logger.error("Bot app not initialized for scheduled quotes")
        return
        
    with group_chats_lock:
        current_chats = group_chats.copy()
    
    if not current_chats:
        logger.info("No group chats registered for scheduled quotes")
        return
        
    logger.info(f"Sending scheduled quotes to {len(current_chats)} chats")
    
    failed_chats = []
    success_count = 0
    
    for chat_id in current_chats:
        try:
            success = await send_quote_to_chat(chat_id, bot_app.bot)
            if success:
                success_count += 1
            else:
                failed_chats.append(chat_id)
        except Exception as e:
            logger.error(f"Failed to send quote to {chat_id}: {e}")
            failed_chats.append(chat_id)
    
    # Remove failed chats from the set
    if failed_chats:
        with group_chats_lock:
            for chat_id in failed_chats:
                group_chats.discard(chat_id)
                logger.info(f"Removed inactive chat {chat_id} from group list")
    
    logger.info(f"Scheduled quotes: {success_count} sent, {len(failed_chats)} failed")

# Commands
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    chat_type = update.message.chat.type
    chat_id = update.message.chat.id
    
    if chat_type == 'private':
        await update.message.reply_text(
            f'Hi {user.first_name}! Welcome to Quote Palace! 🏰\n\n'
            'Use /quote to get an inspirational quote from your favorite characters!\n\n'
            'Add me to groups to get automatic positive quotes every 6 hours! 📅'
        )
    else:
        # Add group to scheduled quotes
        with group_chats_lock:
            group_chats.add(chat_id)
        
        logger.info(f"Added group chat {chat_id} to scheduled quotes. Total groups: {len(group_chats)}")
        
        await update.message.reply_text(
            f'Hello everyone! 👋\n\n'
            'Quote Palace is now active in this group! I\'ll send positive quotes every 6 hours to keep everyone motivated! 🌟\n\n'
            'Use /quote anytime to get an instant inspirational quote!\n'
            'Use /ts and reply to a sticker to translate any text in it! 🌍'
        )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """
🏰 *Quote Palace Bot Commands* 🏰

/start - Welcome message and bot setup
/help - Display this help menu
/quote - Get an inspirational quote from anime, manhwa, movie, TV, or real-life characters
/ts - Translate text in stickers (reply to a sticker with this command)
/tg - Translate text in GIFs (reply to a GIF with this command)
/tv - Translate voice messages (reply to a voice message with this command)
/ud - Get Urban Dictionary definitions for slang terms
/ask - Ask any question and get a concise 30-word answer
/status - Check bot status and group count (admin only)

✨ *Features:*
• In private chats: Get quotes on demand
• In groups: Automatic quotes every 6 hours + on-demand quotes
• Sticker text translation from any language to English
• Voice message translation from any language to English
• Quotes from 500+ beloved characters and personalities
• Powered by AI for fresh, positive content

🌟 _Spreading positivity, one quote at a time!_ 🌟
    """
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show bot status - for debugging"""
    with group_chats_lock:
        group_count = len(group_chats)
    
    gemini_status = "✅" if GEMINI_API_KEY else "❌"
    
    status_text = f"""
🤖 *Bot Status* 🤖

Active Groups: {group_count}
Scheduler Running: {'✅' if schedule.jobs else '❌'}
Bot Instance: {'✅' if bot_app else '❌'}
Total Characters: {len(all_characters)}
Gemini API: {gemini_status}

_For production: Quotes sent every 6 hours_
    """
    await update.message.reply_text(status_text, parse_mode='Markdown')

async def quote_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate and send a quote"""
    try:
        # Send typing action
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
        
        character = get_random_character()
        category = get_character_category(character)
        quote = await generate_quote_with_groq(character)
        
        # Add category emoji
        category_emoji = {
            "Anime": "🎌",
            "Manhwa": "🇰🇷", 
            "Movie": "🎬",
            "TV Series": "📺",
            "Real Life": "🌟"
        }.get(category, "✨")
        
        message = f"💫 *Quote of the Moment* 💫\n\n\"{quote}\"\n\n— _{character}_ {category_emoji}\n\n🌟 _Keep shining!_ 🌟"
        
        await update.message.reply_text(message, parse_mode='Markdown')
        logger.info(f"Quote sent to user {update.message.from_user.id}")
        
    except Exception as e:
        logger.error(f"Error in quote command: {e}")
        await update.message.reply_text(
            "Sorry, I couldn't generate a quote right now. Please try again later! 😊"
        )


# Add this helper function to clean up the translation output
def clean_translation_output(result):
    """Clean up translation output to show only English text"""
    if "No text found" in result:
        return result
    
    # If result contains "Original:" and "English:", extract only the English part
    if "Original:" in result and "English:" in result:
        # Split by "English:" and take the part after it
        english_part = result.split("English:")[-1].strip()
        return english_part
    
    # If result only contains "English:", remove the prefix
    if result.startswith("English:"):
        return result.replace("English:", "").strip()
    
    # If it's just the translation without any prefixes, return as is
    return result

# Update the translate_sticker_command function - replace the response formatting section:
async def translate_sticker_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Translate text in stickers using /ts command"""
    try:
        # Check if this is a reply to a sticker
        if not update.message.reply_to_message or not update.message.reply_to_message.sticker:
            await update.message.reply_text(
                "🌍 Please reply to a sticker with /ts to translate the text in it!\n\n"
                "Usage: Reply to any sticker and type /ts"
            )
            return
        
        # Send typing action
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
        
        # Get the sticker
        sticker = update.message.reply_to_message.sticker
        
        # Send processing message as reply to the original sticker
        processing_msg = await update.message.reply_text(
            "🔍 Analyzing sticker for text...", 
            reply_to_message_id=update.message.reply_to_message.message_id
        )
        
        try:
            # Get the file
            file = await context.bot.get_file(sticker.file_id)
            
            # Download the file
            file_bytes = await file.download_as_bytearray()

            # Check sticker type and determine mime type
            is_animated = sticker.is_animated
            is_video = sticker.is_video
            
            if is_animated and not is_video:
                # TGS (Lottie) animated sticker
                mime_type = "application/x-tgsticker"
                logger.info(f"Processing TGS animated sticker: {len(file_bytes)} bytes")
            elif is_animated and is_video:
                # WebM video sticker
                mime_type = "video/webm"
                logger.info(f"Processing WebM animated sticker: {len(file_bytes)} bytes")
            else:
                # Static WebP sticker
                if len(file_bytes) >= 12 and file_bytes[8:12] == b'WEBP':
                    mime_type = "image/webp"
                else:
                    mime_type = "video/webm"  # Fallback
                logger.info(f"Processing static sticker: {len(file_bytes)} bytes, mime: {mime_type}")
            
            # Extract and translate text
            result = await extract_text_from_image(file_bytes, mime_type, is_animated)
            
            # Format response - UPDATED PART
            if "No text found" in result:
                response = "🤷‍♂️ No text found in this sticker!"
            else:
                # Clean the result to show only English translation
                clean_result = clean_translation_output(result)
                response = f"🌍 *Sticker Translation* 🌍\n\n{clean_result}"
            
            # Edit the processing message with the result
            try:
                await processing_msg.edit_text(response, parse_mode='Markdown')
                logger.info(f"Sticker translation sent to user {update.message.from_user.id}")
            except Exception as edit_error:
                logger.warning(f"Could not edit processing message: {edit_error}")
                # If editing fails, send new message and try to delete the processing one
                await update.message.reply_text(
                    response, 
                    parse_mode='Markdown',
                    reply_to_message_id=update.message.reply_to_message.message_id
                )
                try:
                    await processing_msg.delete()
                except Exception as delete_error:
                    logger.warning(f"Could not delete processing message: {delete_error}")
            
        except Exception as e:
            logger.error(f"Error processing sticker: {e}")
            
            # Edit processing message with error
            error_response = "❌ Sorry, I couldn't process this sticker. Please try again or try with a static sticker."
            try:
                await processing_msg.edit_text(error_response)
            except Exception as edit_error:
                logger.warning(f"Could not edit processing message with error: {edit_error}")
                # If editing fails, send new error message
                await update.message.reply_text(
                    error_response,
                    reply_to_message_id=update.message.message_id
                )
                try:
                    await processing_msg.delete()
                except Exception as delete_error:
                    logger.warning(f"Could not delete processing message: {delete_error}")
            
    except Exception as e:
        logger.error(f"Error in translate_sticker_command: {e}")
        await update.message.reply_text(
            "❌ Sorry, something went wrong while processing the sticker. Please try again later!"
        )

# Update the translate_gif_command function - replace the response formatting section:
async def translate_gif_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Translate text in GIFs using /tg command"""
    try:
        # Check if this is a reply to an animation or document
        replied_msg = update.message.reply_to_message
        if not replied_msg:
            await update.message.reply_text(
                "🎬 Please reply to a GIF with /tg to translate the text in it!\n\n"
                "Usage: Reply to any GIF and type /tg"
            )
            return
        
        # Check for animation (GIF) or document that might be a GIF
        target_file = None
        if replied_msg.animation:
            target_file = replied_msg.animation
            file_type = "animation"
        elif replied_msg.document and replied_msg.document.mime_type in ['image/gif', 'video/mp4']:
            target_file = replied_msg.document
            file_type = "document"
        else:
            await update.message.reply_text(
                "🤷‍♂️ Please reply to a GIF file. I can process animations and GIF documents."
            )
            return
        
        # Send typing action
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
        
        # Send processing message
        processing_msg = await update.message.reply_text(
            "🔍 Analyzing GIF for text...", 
            reply_to_message_id=update.message.reply_to_message.message_id
        )
        
        try:
            # Get the file
            file = await context.bot.get_file(target_file.file_id)
            
            # Download the file
            file_bytes = await file.download_as_bytearray()
            
            # Determine mime type
            if target_file.mime_type:
                mime_type = target_file.mime_type
            else:
                mime_type = "image/gif"
            
            logger.info(f"Processing GIF: {len(file_bytes)} bytes, mime: {mime_type}")
            
            # Extract and translate text
            result = await extract_text_from_image(file_bytes, mime_type, True)
            
            # Format response - UPDATED PART
            if "No text found" in result:
                response = "🤷‍♂️ No text found in this GIF!"
            else:
                # Clean the result to show only English translation
                clean_result = clean_translation_output(result)
                response = f"🎬 *GIF Translation* 🎬\n\n{clean_result}"
            
            # Edit the processing message with the result
            try:
                await processing_msg.edit_text(response, parse_mode='Markdown')
                logger.info(f"GIF translation sent to user {update.message.from_user.id}")
            except Exception as edit_error:
                logger.warning(f"Could not edit processing message: {edit_error}")
                await update.message.reply_text(
                    response, 
                    parse_mode='Markdown',
                    reply_to_message_id=update.message.reply_to_message.message_id
                )
                try:
                    await processing_msg.delete()
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Error processing GIF: {e}")
            error_response = "❌ Sorry, I couldn't process this GIF. Please try again or try with a smaller file."
            try:
                await processing_msg.edit_text(error_response)
            except:
                await update.message.reply_text(error_response)
            
    except Exception as e:
        logger.error(f"Error in translate_gif_command: {e}")
        await update.message.reply_text(
            "❌ Sorry, something went wrong while processing the GIF. Please try again later!"
        )
        
async def ud_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Urban Dictionary command - get slang definitions"""
    
    # Check if word argument is provided
    if not context.args:
        await update.message.reply_text("📚 Usage: /ud <word>\nExample: /ud rizz")
        return
    
    word = ' '.join(context.args).strip().lower()
    
    if not word:
        await update.message.reply_text("❌ Please provide a word to define!")
        return
    
    # Check if word is inappropriate using GROQ
    if await is_inappropriate_word_groq(word):
        funny_reply = get_funny_inappropriate_reply()
        await update.message.reply_text(funny_reply)  # Removed the header line
        return
    
    # Send typing action
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    
    # Send initial loading message
    loading_message = await update.message.reply_text(f"🔍 Looking up '{word}' in urban dictionary...")
    
    # Get definition from GROQ
    definition = await get_urban_definition(word)
    
    # Format the final response
    response_text = f"📚 **Urban Dictionary: {word}**\n\n{definition}"
    
    # Edit the loading message with the final result
    await loading_message.edit_text(response_text, parse_mode='Markdown')


async def is_inappropriate_word_groq(word: str) -> bool:
    """Check if word is inappropriate/NSFW using GROQ AI"""
    logger.info(f"Checking appropriateness for word: {word}")
    
    try:
        if not GROQ_API_KEY:
            logger.warning("GROQ_API_KEY not found for content moderation!")
            return False  # Default to allowing if API is unavailable
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [
                {
                    "role": "system", 
                    "content": """You are a content moderation assistant. Your job is to determine if a word or phrase is inappropriate for a family-friendly bot.

Consider something inappropriate if it's:
- Profanity or vulgar language
- Sexual content or references
- Offensive slurs or hate speech
- Explicit adult content
- Drug-related content
- Violent or disturbing content

Respond with ONLY "YES" if the word is inappropriate, or "NO" if it's appropriate.
Be strict but reasonable - mild slang is usually OK, but anything NSFW or offensive should be marked as inappropriate."""
                },
                {
                    "role": "user", 
                    "content": f"Is this word inappropriate for a family-friendly bot: {word}"
                }
            ],
            "model": GROQ_MODEL,
            "stream": False,
            "temperature": 0.1,  # Low temperature for consistent moderation
            "max_tokens": 10
        }
        
        logger.info(f"Making GROQ API call for content moderation...")
        
        timeout = aiohttp.ClientTimeout(total=15)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(GROQ_API_URL, headers=headers, json=payload) as response:
                logger.info(f"GROQ Moderation API Status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    result = data["choices"][0]["message"]["content"].strip().upper()
                    
                    logger.info(f"GROQ Moderation Response: {result}")
                    return result == "YES"
                else:
                    error_text = await response.text()
                    logger.error(f"GROQ Moderation API Error {response.status}: {error_text}")
                    return False  # Default to allowing if API fails
                    
    except Exception as e:
        logger.error(f"GROQ Moderation API Exception: {str(e)}")
        return False  # Default to allowing if there's an exception


def get_funny_inappropriate_reply() -> str:
    """Get a random funny reply for inappropriate words"""
    import random
    
    funny_replies = [
        "🙈 You are not ready yet to know this! Come back when you're older and wiser!",
        "🚫 Nah nah nah! You're not ready for that level of vocabulary yet! Try something like 'pizza' instead.",
        "😏 You are not ready yet to know this! Your innocent mind must be protected at all costs!",
        "🛡️ Access denied! You are not ready yet to know this. Try asking about cats or cookies instead!",
        "🤫 You are not ready yet to know this! Let's keep things PG-13 around here, shall we?",
        "🙊 Whoa there, sailor! You are not ready yet to know this. How about we look up 'rainbow' instead?",
        "🎭 You are not ready yet to know this! Your pure soul doesn't need such corruption... yet!",
        "🔒 This definition is locked behind an age gate! You are not ready yet to know this, young padawan.",
        "😇 You are not ready yet to know this! Let's keep your search history family-friendly!",
        "🚨 Warning: Adult content detected! You are not ready yet to know this. Try 'memes' instead!"
    ]
    
    return random.choice(funny_replies)


async def get_urban_definition(word: str) -> str:
    """Get urban dictionary style definition using GROQ API"""
    logger.info(f"get_urban_definition called with: {word}")
    
    try:
        if not GROQ_API_KEY:
            logger.warning("GROQ_API_KEY not found!")
            return "❌ Dictionary service is offline!"
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [
                {
                    "role": "system", 
                    "content": """You are an expert Urban Dictionary assistant specialized in modern slang and internet culture. Provide accurate, clear definitions for slang terms.

Guidelines:
- Give a precise, easy-to-understand definition (1-2 sentences)
- Keep it family-friendly and appropriate 
- Include ONE realistic example of how it's used in conversation
- Use this exact format: "**Definition:** [clear definition]\n**Example:** [realistic usage example]"
- Focus on the most common/popular meaning of the term
- If it's internet slang, explain the context (gaming, social media, etc.)
- Make it sound natural and relatable to young people
- NO additional notes, explanations, or disclaimers
- If unsure about a term, admit it honestly but try to provide context"""
                },
                {
                    "role": "user", 
                    "content": f"Define this slang word/term: {word}"
                }
            ],
            "model": GROQ_MODEL,
            "stream": False,
            "temperature": 0.6,  # Slightly creative but consistent
            "max_tokens": 120
        }
        
        logger.info(f"Making API call to GROQ for urban definition...")
        
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(GROQ_API_URL, headers=headers, json=payload) as response:
                logger.info(f"GROQ API Status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    definition = data["choices"][0]["message"]["content"].strip()
                    
                    # Clean up any unwanted additions
                    if "Note:" in definition:
                        definition = definition.split("Note:")[0].strip()
                    if "Disclaimer:" in definition:
                        definition = definition.split("Disclaimer:")[0].strip()
                    
                    logger.info(f"GROQ Urban Definition Response: {definition}")
                    return definition if definition else "🤷‍♂️ This word is too new for my dictionary!"
                else:
                    error_text = await response.text()
                    logger.error(f"GROQ API Error {response.status}: {error_text}")
                    return "❌ Dictionary service is temporarily unavailable!"
                    
    except Exception as e:
        logger.error(f"GROQ API Exception: {str(e)}")
        return "🤖 Dictionary lookup failed. Please try again later!"

def run_scheduler():
    """Run the scheduler in a separate thread"""
    logger.info("Scheduler thread started")
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)  # Check every second
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            time.sleep(5)  # Wait before retrying

# FIXED: Proper webhook handling with persistent event loop
webhook_loop = None
webhook_thread = None

def setup_webhook_loop():
    """Set up a persistent event loop for webhook processing"""
    global webhook_loop
    webhook_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(webhook_loop)
    logger.info("Webhook event loop started")
    webhook_loop.run_forever()

@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming webhook requests"""
    try:
        json_data = request.get_json(force=True)
        
        # Only process if it's a message (not other updates like join/leave)
        if 'message' not in json_data:
            return jsonify({'status': 'ignored'})
        
        # Only process if it's a command (starts with /)
        message = json_data.get('message', {})
        text = message.get('text', '')
        
        if not text.startswith('/'):
            return jsonify({'status': 'ignored'})
        
        update = Update.de_json(json_data, bot_app.bot)
        
        # Use the persistent event loop to process the update
        future = asyncio.run_coroutine_threadsafe(
            bot_app.process_update(update), 
            webhook_loop
        )
        
        # Wait for completion with timeout
        try:
            future.result(timeout=30)
        except asyncio.TimeoutError:
            logger.error("Webhook processing timed out")
            return jsonify({'status': 'timeout'})
            
        return jsonify({'status': 'ok'})
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'groups': len(group_chats), 'scheduler_jobs': len(schedule.jobs)})

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Quote Palace Bot is running!', 'groups': len(group_chats)})

async def setup_bot():
    """Set up the bot application"""
    global bot_app
    
    try:
        bot_app = Application.builder().token(BOT_TOKEN).build()
        
        # Add handlers
        bot_app.add_handler(CommandHandler('start', start_command))
        bot_app.add_handler(CommandHandler('help', help_command))
        bot_app.add_handler(CommandHandler('quote', quote_command))
        bot_app.add_handler(CommandHandler('status', status_command))
        bot_app.add_handler(CommandHandler('ts', translate_sticker_command))
        bot_app.add_handler(CommandHandler('tg', translate_gif_command))
        bot_app.add_handler(CommandHandler('tv', translate_voice_command))
        bot_app.add_handler(CommandHandler('ud', ud_command))
        bot_app.add_handler(CommandHandler('ask', ask_command))
        
        # Initialize the bot
        await bot_app.initialize()
        await bot_app.start()
        
        # Set webhook
        webhook_url = f"https://gcaiassistantbot-production.up.railway.app/webhook"
        await bot_app.bot.set_webhook(webhook_url)
        
        logger.info("Bot setup completed successfully")
        
    except Exception as e:
        logger.error(f"Error setting up bot: {e}")
        raise

def run_bot_setup():
    """Run bot setup in a separate thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(setup_bot())
        logger.info("Bot setup completed in thread")
    except Exception as e:
        logger.error(f"Bot setup failed: {e}")
    finally:
        loop.close()

async def run_scheduled_job():
    """Wrapper to run scheduled job in async context"""
    try:
        await scheduled_quotes_job()
    except Exception as e:
        logger.error(f"Scheduled job error: {e}")

def schedule_quotes():
    """Schedule quotes every minute for testing"""
    def job():
        logger.info("Scheduled job triggered")
        if bot_app and webhook_loop and not webhook_loop.is_closed():
            try:
                future = asyncio.run_coroutine_threadsafe(
                    run_scheduled_job(), 
                    webhook_loop
                )
                future.result(timeout=60)
                logger.info("Scheduled job completed successfully")
            except Exception as e:
                logger.error(f"Scheduled job failed: {e}")
        else:
            logger.error("Bot app or webhook loop not available for scheduled job")
    
    # Schedule for every 6 hours
    schedule.every(6).hours.do(job)
    
    # Start scheduler thread
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()
    
    logger.info("Quote scheduler started - quotes every 6 hours")

async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Answer questions concisely using GROQ API"""
    
    # Check if question is provided
    if not context.args:
        await update.message.reply_text(
            "💭 Usage: /ask <your question>\n"
            "Example: /ask what is the difference between shorting and limiting in trading"
        )
        return
    
    question = ' '.join(context.args).strip()
    
    if not question:
        await update.message.reply_text("❌ Please provide a question to answer!")
        return
    
    # Send typing action
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    
    # Send initial loading message
    loading_message = await update.message.reply_text("🤔 Thinking...")
    
    try:
        if not GROQ_API_KEY:
            await loading_message.edit_text("❌ AI service is currently unavailable.")
            return
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a concise AI assistant. Answer questions in MAXIMUM 30 words. Be precise, clear, and to the point. No fluff or unnecessary details."
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            "model": GROQ_MODEL,
            "stream": False,
            "temperature": 0.3,
            "max_tokens": 60  # Roughly 30 words
        }
        
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(GROQ_API_URL, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    answer = data["choices"][0]["message"]["content"].strip()
                    
                    # Format response
                    response_text = f"💡 **Answer:**\n\n{answer}"
                    await loading_message.edit_text(response_text, parse_mode='Markdown')
                    logger.info(f"Question answered for user {update.message.from_user.id}")
                else:
                    error_text = await response.text()
                    logger.error(f"GROQ API Error {response.status}: {error_text}")
                    await loading_message.edit_text("❌ Failed to get answer. Please try again!")
                    
    except Exception as e:
        logger.error(f"Error in ask_command: {e}")
        await loading_message.edit_text("❌ Something went wrong. Please try again later!")


if __name__ == '__main__':
    print('Starting Quote Palace Bot..........')
    print(f'Bot Token: {BOT_TOKEN[:10]}...' if BOT_TOKEN else 'Bot Token: NOT SET')
    print(f'GROQ API Key: {GROQ_API_KEY[:10]}...' if GROQ_API_KEY else 'GROQ API Key: NOT SET')
    print(f'Gemini API Key: {GEMINI_API_KEY[:10]}...' if GEMINI_API_KEY else 'Gemini API Key: NOT SET')
    
    if not BOT_TOKEN:
        print("ERROR: BOT_TOKEN environment variable not set!")
        exit(1)
    
    if not GROQ_API_KEY:
        print("WARNING: GROQ_API_KEY environment variable not set! Using fallback quotes only.")
    
    if not GEMINI_API_KEY:
        print("WARNING: GEMINI_API_KEY environment variable not set! Sticker translation won't work.")
    
    # Start webhook event loop thread
    webhook_thread = threading.Thread(target=setup_webhook_loop)
    webhook_thread.daemon = True
    webhook_thread.start()
    
    # Wait for webhook loop to start
    time.sleep(2)
    
    # Set up bot in a separate thread
    bot_thread = threading.Thread(target=run_bot_setup)
    bot_thread.daemon = True
    bot_thread.start()
    
    # Wait for bot setup
    time.sleep(5)
    
    # Start scheduling quotes
    schedule_quotes()
    
    # Run Flask app
    logger.info(f"Starting Flask app on port {PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=False)
