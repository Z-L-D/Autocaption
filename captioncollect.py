import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Pair images with captions from text files and output the pairs to a text file.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing images and their corresponding text files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output text file.")
    args = parser.parse_args()
    return args

def is_image_file(filename):
    # List of common image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    return any(filename.lower().endswith(ext) for ext in image_extensions)

def main(args):
    args = parse_args()

    # Ensure the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Prepare the output text file
    output_file_path = os.path.join(args.output_dir, "image_caption_pairs.txt")
    
    with open(output_file_path, 'w') as output_file:
        # Process each file in the input directory
        for filename in os.listdir(args.input_dir):
            if is_image_file(filename):
                base_name = os.path.splitext(filename)[0]
                text_filename = base_name + '.txt'
                text_file_path = os.path.join(args.input_dir, text_filename)
                
                if os.path.isfile(text_file_path):
                    with open(text_file_path, 'r') as text_file:
                        caption = text_file.read().strip()
                    image_path = os.path.join(args.input_dir, filename)
                    output_file.write(f"{image_path}\t{caption}\n")
                    print(f"Processed {filename} successfully.")
                else:
                    print(f"Caption file for {filename} not found.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
