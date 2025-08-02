from diffusers import StableDiffusionPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline, set_seed
from PIL import Image
import torch
import gradio as gr
import os

# Load Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to("cuda" if torch.cuda.is_available() else "cpu")

# Load BLIP for image captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load GPT2 for mood-based rewriting
caption_generator = pipeline("text-generation", model="gpt2")
set_seed(42)

# Generate caption from image using BLIP
def generate_image_caption(image):
    inputs = blip_processor(image, return_tensors="pt").to(blip_model.device)
    output = blip_model.generate(**inputs)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    return caption

# Rewrite caption with mood using GPT2
def rewrite_caption_with_mood(caption, mood):
    prompt = f"Rewrite the following caption in a {mood} tone: '{caption}'"
    result = caption_generator(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
    return result

# Main function

def full_pipeline(prompt, mood):
    image = pipe(prompt).images[0]
    temp_path = "temp_image.png"
    image.save(temp_path)
    caption = generate_image_caption(Image.open(temp_path).convert("RGB"))
    mood_caption = rewrite_caption_with_mood(caption, mood)
    os.remove(temp_path)
    return image, mood_caption

# Gradio Interface
gr.Interface(
    fn=full_pipeline,
    inputs=[
        gr.Textbox(label="Enter Prompt for Image"),
        gr.Dropdown(["funny", "romantic", "inspirational", "sarcastic"], label="Caption Mood")
    ],
    outputs=[
        gr.Image(label="Generated Image"),
        gr.Textbox(label="Generated Caption")
    ],
    title="SnapScribe AI",
    description="Generate AI Images and Captions Based on Your Prompt & Mood"
).queue().launch()
