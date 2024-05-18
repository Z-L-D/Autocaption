import os
import sys
import argparse
from PIL import Image
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor

# Arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--llava_dir", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--custom_prompt", type=str, default="")
    args = parser.parse_args()
    return args

def batch_process_images(image_paths, img_processor, torch_device, custom_prompt):
    images = [Image.open(image_path) for image_path in image_paths]
    prompts = [f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{custom_prompt}\nASSISTANT:" for _ in images]
    inputs = img_processor(prompts, images, return_tensors='pt', padding=True).to(torch_device)
    return inputs

def main(args, progress_callback=None):
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup Environment
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load LLM
    text_encoder = LlavaForConditionalGeneration.from_pretrained(args.llava_dir, torch_dtype=torch.float16).to(torch_device)
    img_processor = AutoProcessor.from_pretrained(args.llava_dir, torch_dtype=torch.float16)

    # List all image files and count text files
    all_image_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    total_images = len(all_image_files)
    text_files = [os.path.join(args.output_dir, f) for f in os.listdir(args.output_dir) if f.lower().endswith('.txt')]
    processed_images = len(text_files)

    # Display initial progress
    initial_progress = (processed_images / total_images) * 100
    if progress_callback:
        progress_callback(f'Initial progress: {processed_images}/{total_images} images ({initial_progress:.2f}%)')

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
            inputs = batch_process_images(batch_files, img_processor, torch_device, args.custom_prompt)
            outputs = text_encoder.generate(**inputs, max_new_tokens=200, do_sample=False)
            for idx, output in enumerate(outputs):
                decoded_output = img_processor.decode(output[2:], skip_special_tokens=True)
                assistant_response = decoded_output.split('ASSISTANT: ')[-1].strip()
                output_filename = os.path.splitext(os.path.basename(batch_files[idx]))[0] + '.txt'
                output_path = os.path.join(args.output_dir, output_filename)
                with open(output_path, 'w') as file:
                    file.write(assistant_response)
                processed_images += 1
                progress = (processed_images / total_images) * 100
                if progress_callback:
                    progress_callback(f'Processed {processed_images}/{total_images} images ({progress:.2f}%)')
        except Exception as e:
            if progress_callback:
                progress_callback(f'Error processing batch: {str(e)}')
            continue

if __name__ == "__main__":
    args = parse_args()
    main(args)
