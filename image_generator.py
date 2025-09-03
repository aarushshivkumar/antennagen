import os
import json
import time
from datetime import datetime
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import fnmatch

def log_message(message):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    
    # Also save to log file
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/generation_{datetime.now().strftime('%Y%m%d')}.log", "a") as f:
        f.write(f"[{timestamp}] {message}\n")

def save_progress(key_index, total_images, key_images, exhausted_keys):
    """Save current progress to file"""
    progress = {
        "current_key_index": key_index,
        "total_images_generated": total_images,
        "images_per_key": key_images,
        "exhausted_keys": exhausted_keys,
        "last_updated": datetime.now().isoformat()
    }
    
    with open("progress.json", "w") as f:
        json.dump(progress, f, indent=2)

def load_progress():
    """Load progress from file if exists"""
    if os.path.exists("progress.json"):
        try:
            with open("progress.json", "r") as f:
                return json.load(f)
        except:
            pass
    
    return {
        "current_key_index": 0,
        "total_images_generated": 0,
        "images_per_key": {},
        "exhausted_keys": []
    }

def test_api_key(client):
    """Test if API key is still working"""
    try:
        # Try a simple generation request
        resp = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents="simple test image",
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )
        return True
    except Exception as e:
        error_msg = str(e).lower()
        # Check for quota/rate limit errors
        if any(keyword in error_msg for keyword in ['quota', 'limit', 'rate', 'exceeded', '429']):
            return False
        # For other errors, we might want to try again
        log_message(f"API test error (retrying): {e}")
        return True

def generate_with_key(client, key_name, contents, output_dir, start_index, max_images_per_key=500):
    """Generate images with a specific API key until exhausted"""
    images_generated = 0
    consecutive_errors = 0
    sleep_time = 8
    requests_made = 0
    
    log_message(f"Starting generation with key: {key_name}")
    
    while images_generated < max_images_per_key:
        try:
            # Progressive sleep after every 15 requests
            if requests_made > 0 and requests_made % 15 == 0:
                sleep_time += 7
                log_message(f"Increasing sleep time to {sleep_time}s after {requests_made} requests")
            
            time.sleep(sleep_time)
            
            # Make API request
            resp = client.models.generate_content(
                model="gemini-2.0-flash-exp-image-generation",
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            )
            
            requests_made += 1
            
            # Process response
            image_saved = False
            for part in resp.candidates[0].content.parts:
                if part.text is not None:
                    log_message(f"Generated text: {part.text[:100]}...")
                elif part.inline_data is not None:
                    # Save image
                    total_files = len(fnmatch.filter(os.listdir(output_dir), '*.jpg'))
                    image_path = os.path.join(output_dir, f"gen_{start_index + total_files + 1}.jpg")
                    
                    image = Image.open(BytesIO(part.inline_data.data))
                    image.save(image_path)
                    
                    images_generated += 1
                    image_saved = True
                    log_message(f"Saved image {images_generated} for key {key_name}: {image_path}")
            
            if image_saved:
                consecutive_errors = 0
            else:
                log_message("No image in response")
                
        except Exception as e:
            consecutive_errors += 1
            error_msg = str(e).lower()
            
            log_message(f"Error with key {key_name} (attempt {consecutive_errors}): {e}")
            
            # Check for quota exhaustion
            if any(keyword in error_msg for keyword in ['quota', 'limit', 'rate', 'exceeded', '429']):
                log_message(f"Key {key_name} exhausted after {images_generated} images")
                break
                
            # If too many consecutive errors, assume key is exhausted
            if consecutive_errors >= 5:
                log_message(f"Key {key_name} failed {consecutive_errors} times, assuming exhausted")
                break
                
            # Wait longer after errors
            time.sleep(min(60, sleep_time * 2))
    
    log_message(f"Finished with key {key_name}. Generated {images_generated} images.")
    return images_generated

def main():
    # Get API keys
    gen_keys_env = os.getenv("GEN_KEYS")
    if not gen_keys_env:
        log_message("ERROR: GEN_KEYS environment variable not set")
        return
    
    all_keys = gen_keys_env.strip().split()
    log_message(f"Found {len(all_keys)} API keys")
    
    # Setup
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load previous progress
    progress = load_progress()
    
    # Image generation prompt
    contents = """Camera mounted on top of cellular communication tower looking straight down, view from tower peak showing dense cluster of antennas and equipment directly below, prominent electrical junction box or equipment cabinet visible on tower structure, downward perspective from tower summit revealing complex array of telecommunications equipment, mix of large white panel antennas and smaller directional antennas visible from above, metal wire management box or control cabinet prominently featured among the equipment, some antennas partially obscured by equipment at higher levels and by the wire box, overlapping antenna arrangements with upper equipment and junction box casting shadows on lower antennas, antennas at various orientations showing tops, sides, and angled mounting positions, layered installation with natural occlusion from tower-top viewpoint including wire box obstruction, white and gray colored antenna panels and cylindrical equipment, electrical cabinet or junction box with visible cable connections, complex antenna clustering viewed from directly overhead on tower structure, residential buildings and green vegetation visible far below in background, realistic photography from tower peak position, high detail, equipment arranged in tiers descending down the tower"""
    
    total_images_start = len(fnmatch.filter(os.listdir(output_dir), '*.jpg'))
    log_message(f"Starting with {total_images_start} existing images")
    
    # Process each API key
    for i in range(progress["current_key_index"], len(all_keys)):
        key = all_keys[i]
        key_name = f"key_{i+1}"
        
        # Skip if key is already exhausted
        if key in progress["exhausted_keys"]:
            log_message(f"Skipping already exhausted key: {key_name}")
            continue
        
        try:
            # Create client
            client = genai.Client(api_key=key)
            
            # Test key first
            if not test_api_key(client):
                log_message(f"Key {key_name} appears to be exhausted, skipping")
                progress["exhausted_keys"].append(key)
                continue
            
            # Generate images with this key
            images_generated = generate_with_key(
                client, key_name, contents, output_dir, total_images_start
            )
            
            # Update progress
            progress["current_key_index"] = i
            progress["total_images_generated"] += images_generated
            progress["images_per_key"][key_name] = images_generated
            
            if images_generated < 50:  # If very few images, likely exhausted
                progress["exhausted_keys"].append(key)
            
            save_progress(i, progress["total_images_generated"], 
                         progress["images_per_key"], progress["exhausted_keys"])
            
            # Check if we've generated enough images total
            total_current = len(fnmatch.filter(os.listdir(output_dir), '*.jpg'))
            log_message(f"Total images now: {total_current}")
            
        except Exception as e:
            log_message(f"Fatal error with key {key_name}: {e}")
            progress["exhausted_keys"].append(key)
            continue
    
    # Final summary
    final_count = len(fnmatch.filter(os.listdir(output_dir), '*.jpg'))
    log_message(f"Image generation complete. Final count: {final_count}")
    log_message(f"Images per key: {progress['images_per_key']}")
    log_message(f"Exhausted keys: {len(progress['exhausted_keys'])}/{len(all_keys)}")

if __name__ == "__main__":
    main()
