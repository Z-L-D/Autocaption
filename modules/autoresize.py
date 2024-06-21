import argparse
import os
from PIL import Image

def resize_image(input_path, output_path, max_width, max_height):
    with Image.open(input_path) as img:
        # Calculate the target size while maintaining aspect ratio
        img_ratio = img.width / img.height
        target_ratio = max_width / max_height
        
        if img_ratio >= target_ratio:
            # Image is wider
            scale_factor = max_width / img.width
        else:
            # Image is taller
            scale_factor = max_height / img.height
        
        new_width = int(img.width * scale_factor)
        new_height = int(img.height * scale_factor)

        # Resize the image
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        # Save the resized image
        resized_img.save(output_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Resize images to fit within specified dimensions while maintaining aspect ratio.")
    parser.add_argument("--max_width", type=int, required=True, help="Maximum width of the resized images.")
    parser.add_argument("--max_height", type=int, required=True, help="Maximum height of the resized images.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing images to resize.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save resized images.")
    args = parser.parse_args()
    return args

def main(args):
    args = parse_args()

    # Check if output directory exists, create if not
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Process each image in the input directory
    for filename in os.listdir(args.input_dir):
        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, filename)

        if os.path.isfile(input_path):
            try:
                resize_image(input_path, output_path, args.max_width, args.max_height)
                print(f"Resized {filename} successfully.")
            except Exception as e:
                print(f"Failed to resize {filename}: {e}")

if __name__ == "__main__":
    args = parse_args()
    main(args)