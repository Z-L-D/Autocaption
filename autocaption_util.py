import os
import sys
import argparse
import json
import io
import numpy as np
from PIL import Image
import torch
from transformers import (LlavaForConditionalGeneration, AutoProcessor, LlavaProcessor, TextIteratorStreamer)
import gradio as gr
import time 
from threading import Thread

# Global variables to store the LLM and image processor
text_encoder = None
img_processor = None


# Arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--llava_dir", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--keep_loaded", type=bool, default=True)
    args = parser.parse_args()
    return args


def load_models(llava_dir):
    global text_encoder, img_processor
    if text_encoder is None or img_processor is None:
        config_path = os.path.join(llava_dir, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        architecture = config['architectures'][0]  # Use the first architecture listed

        torch_device = "cuda"

        if architecture == "LlavaForConditionalGeneration":
            text_encoder = LlavaForConditionalGeneration.from_pretrained(llava_dir, torch_dtype=torch.float16).to(torch_device)
        else:
            raise ValueError(f"Unsupported architecture, please use Llava models: {architecture}")

        img_processor = AutoProcessor.from_pretrained(llava_dir, torch_dtype=torch.float16)


def unload_models():
    global text_encoder, img_processor
    text_encoder = None
    img_processor = None
    torch.cuda.empty_cache()  # Clear the GPU cache


def chat(args):
    global text_encoder, img_processor
    yield f'Loading Language Model...'
    load_models(args.llm_dir)

    prompt = args.prompt
    image = args.image

    print("Received prompt: ", prompt)
    print("Received image type: ", type(image))

    # Convert numpy array to file-like object if necessary
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'), 'RGB')
        print("Converted ndarray to PIL Image")
    elif isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
        print("Converted bytes to PIL Image")
    else:
        image = Image.open(image)
        print("Loaded image using PIL")

    # Ensure the image is correctly loaded
    if image is None:
        print("Error: Image is None")
        yield "Error: Image could not be processed."
        return

    prompts = [f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{prompt}\nASSISTANT:"]

    try:
        inputs = img_processor(prompts, image, return_tensors='pt', padding=True).to("cuda")
        print("Inputs processed for model")
        yield f'Processing caption...'
    except Exception as e:
        print("Error during input processing: ", str(e))
        yield f"Error during input processing: {str(e)}"
        return
    
    try:
        output = text_encoder.generate(**inputs, max_new_tokens=200, do_sample=False)
        print("Model output generated")
    except Exception as e:
        print("Error during model generation: ", str(e))
        yield f"Error during model generation: {str(e)}"
        return

    try:
        decoded_output = img_processor.decode(output[0], skip_special_tokens=True)
        print("Decoded output: " + decoded_output)
        response = decoded_output.split('ASSISTANT: ')[-1].strip()
        print("Response: " + response)
        yield response
    except Exception as e:
        print("Error during output decoding: ", str(e))
        yield f"Error during output decoding: {str(e)}"
        return

    # Unload models if not keeping them loaded
    if not args.keep_loaded:
        unload_models()
        print("Models unloaded")


def batch_process_images(image_paths, prompt, img_processor, torch_device):
    images = [Image.open(image_path) for image_path in image_paths]
    prompts = [f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{prompt}\nASSISTANT:" for _ in images]
    inputs = img_processor(prompts, images, return_tensors='pt', padding=True).to(torch_device)
    return inputs


def captioning(args):
    global text_encoder, img_processor
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models if not already loaded
    yield f'Loading Language Model...'
    load_models(args.llm_dir)

    prompt = args.prompt

    # List all image files and count text files
    try:
        all_image_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        total_images = len(all_image_files)
        text_files = [os.path.join(args.output_dir, f) for f in os.listdir(args.output_dir) if f.lower().endswith('.txt')]
        processed_images = len(text_files)
    except Exception as e:
        print("Error during file listing: ", str(e))
        yield f"Error during file listing: {str(e)}"
        return

    # Display initial progress
    initial_progress = (processed_images / total_images) * 100
    yield f'Initial progress: {processed_images}/{total_images} images ({initial_progress:.2f}%)'

    # Batch Process Input Directory
    for i in range(0, total_images, args.batch_size):
        batch_files = []
        for f in all_image_files[i:i+args.batch_size]:
            output_filename = os.path.splitext(os.path.basename(f))[0] + '.txt'
            output_path = os.path.join(args.output_dir, output_filename)
            if not os.path.exists(output_path):
                batch_files.append(f)
        if not batch_files:
            continue

        try:
            inputs = batch_process_images(batch_files, prompt, img_processor, "cuda")
            print("Inputs processed for model")
        except Exception as e:
            print("Error during input processing: ", str(e))
            yield f"Error during input processing: {str(e)}"
            return

        try:
            outputs = text_encoder.generate(**inputs, max_new_tokens=200, do_sample=False)
            print("Model output generated")
        except Exception as e:
            print("Error during model generation: ", str(e))
            yield f"Error during model generation: {str(e)}"
            return

        for idx, output in enumerate(outputs):
            try:
                decoded_output = img_processor.decode(output[2:], skip_special_tokens=True)
                assistant_response = decoded_output.split('ASSISTANT: ')[-1].strip()
                output_filename = os.path.splitext(os.path.basename(batch_files[idx]))[0] + '.txt'
                output_path = os.path.join(args.output_dir, output_filename)
                with open(output_path, 'w') as file:
                    file.write(assistant_response)
                print(f"Processed file: {output_path}")
                processed_images += 1
                progress = (processed_images / total_images) * 100
                yield f'Processed {processed_images}/{total_images} images ({progress:.2f}%)'
            except Exception as e:
                print("Error during output decoding or saving: ", str(e))
                yield f"Error during output decoding or saving: {str(e)}"
                return

    # Unload models if not keeping them loaded
    if not args.keep_loaded:
        unload_models()
        print("Models unloaded")


def load_dataset(folder_path):
    samples = []
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(folder_path, filename)
            image_path = os.path.normpath(image_path)  # Normalize the path
            text_path = os.path.join(folder_path, os.path.splitext(filename)[0] + '.txt')
            if os.path.exists(text_path):
                with open(text_path, 'r') as file:
                    text = file.read()
                samples.append([image_path, text])
    print("Loaded samples:", samples)
    return samples