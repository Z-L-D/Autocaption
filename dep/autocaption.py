import os
import sys
import requests
import argparse
from PIL import Image

import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoTokenizer

# Arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--llava_dir", type=str, default="")
    args = parser.parse_args()
    return args

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup Environment
    torch_device = "cuda"

    # Load LLM
    text_encoder = LlavaForConditionalGeneration.from_pretrained(args.llava_dir, torch_dtype=torch.float16).to(torch_device)
    img_processor = AutoProcessor.from_pretrained(args.llava_dir)

    # Batch Process Input Directory
    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Check if output text file already exists
            output_filename = os.path.splitext(filename)[0] + '.txt'
            output_path = os.path.join(args.output_dir, output_filename)
            if os.path.exists(output_path):
                print(f"Skipping {filename} as its description file already exists.")
                continue

            image_path = os.path.join(args.input_dir, filename)
            raw_image = Image.open(image_path)

            # Create prompt for image description
            prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\nDescribe the scene in detail, highlighting all the key elements, objects, and actions depicted in the image. Mention any explicit or potentially offensive content clearly with vulgarity if necessary. Use a minimum of 75 words and aim for a descriptive and engaging caption.\nASSISTANT:"
            inputs = img_processor(prompt, raw_image, return_tensors='pt').to(torch_device, torch.float16)

            # Generate caption
            output = text_encoder.generate(**inputs, max_new_tokens=200, do_sample=False)
            decoded_output = img_processor.decode(output[0][2:], skip_special_tokens=True)

            # Extract only the assistant's response
            assistant_response = decoded_output.split('ASSISTANT: ')[-1].strip()

            # Write output to a text file
            with open(output_path, 'w') as file:
                file.write(assistant_response)

if __name__ == "__main__":
    args = parse_args()
    main(args)
