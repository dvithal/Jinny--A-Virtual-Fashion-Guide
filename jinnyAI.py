import gradio as gr
from PIL import Image
import torch
import random
import os
from colorthief import ColorThief
from transformers import BlipForConditionalGeneration, BlipProcessor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import io

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BLIP (for outfit image captioning)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

# Load FLAN-T5 (for generating feedback text)
flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)

def extract_style_keywords(message):
    keywords = []
    style_words = ["minimal", "oversized", "vintage", "bold", "casual", "formal", "classic",
                "edgy", "grunge","boho","sporty","streetwear", "elevated basics"]
    for word in style_words:
        if word in message.lower():
            keywords.append(word)
    return keywords

def extract_palette_text(image):
    with io.BytesIO() as f:
        image.save(f, format="PNG")
        f.seek(0)
        ct = ColorThief(f)
        palette = ct.get_palette(color_count=5)
        hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in palette]
        return ", ".join(hex_colors)

def  detect_garment_type(image):
    prompt = "What type of garment is this? (e.g., dress, jacket, pants, shirt, top)"
    inputs = blip_processor(images=image, text=prompt, return_tensors="pt").to(device)
    output_ids = blip_model.generate(**inputs, max_new_tokens=15, num_beams=3)
    garment_type = blip_processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return garment_type

def chat_with_jinny(message, image, gender, preferences):

    gender = gender if gender else "unspecified"
    message = message if message else "They asked for outfit advice."

    # If greeting only
    if message.strip().lower() in ["hi", "hello", "hey", "hii", "yo", "hola"]:
        return ("Hi, I’m Jinny — your virtual fashion guide. I can help you pick the best outfit for any "
                "occasion and rate your looks.  Upload an outfit or tell me what occasion you're dressing for,"
                " and I'll help you style it!")

    if ("thanks" in message.lower()) and image is None:
        return "No problem, it's my pleasure to help you."

    if ("rate" in message.lower() or "help with an outfit" in message.lower()) and (image is None):
        return ("I'd love to help you rate your outfit or suggest improvements! "
                "Please upload an image so I can give you a proper score and detailed advice.")

    # flavor = random.choice(flavors)
    style_keywords = extract_style_keywords(message)
    suggestion_line = ""
    if preferences and preferences.get("style"):
        suggestion_line = f"You mentioned liking a {preferences['style']} style before — would you like something similar today?\n"

    if image is not None:
        inputs = blip_processor(images=image, text=["a photo of something"], return_tensors="pt").to(device)
        output_ids = blip_model.generate(**inputs, max_new_tokens=50, num_beams=3)
        caption = blip_processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Get palette
        palette_text = extract_palette_text(image)

        # Garment type
        garment_type = detect_garment_type(image)

        prompt = (
            f"{suggestion_line}"
            f"A user says: \"{message}\". The user identifies as {gender}.\n"
            f"Outfit caption: \"{caption}\"\n"
            f"type of colors in the outfit: \"{palette_text}\"\n"
            f"Garment type detected: {garment_type}\n"
            "As a professional fashion stylist, analyze this outfit in detail.\n"
            " Mention pros and cons.\n"
            " Suggest at least three improvements or accessories, considering gender.\n"
            " Mention shoe colors, accessories, or hair styling tips.\n"
            "Be very descriptive, detailed, and talkative. Write at least 5-6 sentences, sound friendly and encouraging.\n"
            "Finally, say clearly whether you'd recommend wearing it as is or with changes."
        )
    else:
        prompt = (
            f"A user says: \"{message}\". The user identifies as {gender}.\n"
            "Please give detailed outfit advice based on the text.\n"
            "Include colors, cuts, shoe choices, accessories, and hairstyle.\n"
            "Be encouraging and detailed so they get a clear idea even without a photo."
        )

        if ("thanks" in message.lower()) and image is not None:
            return "No problem, it's my pleasure to help you."

    flan_inputs = flan_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    flan_outputs = flan_model.generate(
        flan_inputs,
        max_new_tokens=500,
        # temperature=0.9,
        # top_p=0.95,
        repetition_penalty=2.0,
        no_repeat_ngram_size=3
    )
    result = flan_tokenizer.decode(flan_outputs[0], skip_special_tokens=True)

    # Save detected style preference
    if style_keywords:
        preferences["style"] = ", ".join(style_keywords)

    return result

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "<h1 style='text-align: center;'> VIRTUAL FASHION GUIDE</h1>"
        "<p style='text-align: center; font-size: 18px;'>"
        "<strong>Hi, I’m Jinny</strong> — your virtual fashion guide. I can help you pick the best outfit for any occasion "
        "and rate your looks. Upload an outfit or just chat with me!"
        "</p>"
    )

    with gr.Row():
        msg = gr.Textbox(label="What's on your mind?", placeholder="Ask Jinny for outfit advice...")
        gender = gr.Radio(choices=["Male", "Female"], label="Select your gender")
        img = gr.Image(type="pil", label="Upload an outfit")

    chatbot = gr.Chatbot(label="Chatbot", type="messages")
    stored_image = gr.State()
    user_preferences = gr.State(value={})

    def respond(message, image, gender, history, stored_image, preferences):
        if preferences is None:
            preferences = {}

        if image is not None:
            stored_image = image
        else:
            image = stored_image

        reply = chat_with_jinny(message,  stored_image, gender,  preferences)

        history = history or []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        return history, "", stored_image, preferences

    msg.submit(respond, [msg, img, gender, chatbot, stored_image, user_preferences], [chatbot, msg, stored_image,user_preferences])

# Run app
demo.launch(share=True)

