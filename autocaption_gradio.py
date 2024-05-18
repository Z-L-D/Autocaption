import gradio as gr
import os
from autocaption_util import main, parse_args

def run_script(input_dir, output_dir, llava_dir, batch_size, custom_prompt):
    # Prepare arguments
    args = parse_args()
    args.input_dir = input_dir
    args.output_dir = output_dir
    args.llava_dir = llava_dir
    args.batch_size = int(batch_size)
    args.custom_prompt = custom_prompt
    
    # Run main function from the script and capture the generator output
    progress_updates = []
    
    def progress_callback(update):
        progress_updates.append(update)
    
    main(args, progress_callback=progress_callback)
    
    return progress_updates

# Default prompts
default_prompts = [
    "Describe the scene in detail, highlighting all the key elements, objects, and actions depicted in the image. Mention any explicit or potentially offensive content clearly with vulgarity if necessary. Use a minimum of 75 words and aim for a descriptive and engaging caption.",
    "Provide a detailed description of the image, focusing on the main subjects, their activities, and the background. Include any notable features and describe the overall atmosphere. Ensure the caption is at least 50 words long.",
    "Summarize the image by describing the primary elements, actions, and any notable details. Aim for a concise yet informative caption with a minimum of 30 words."
]

# Disable Gradio Analytics
gr.Blocks(analytics_enabled=False)
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

# Define Gradio interface
interface = gr.Blocks()

with interface:
    gr.Markdown("## Batch Image Captioning with LLaVA")
    
    with gr.Row():
        input_dir = gr.Textbox(label="Input Directory", placeholder="Path to your input directory containing images")
        output_dir = gr.Textbox(label="Output Directory", placeholder="Path to your output directory for text files")
        llava_dir = gr.Textbox(label="LLaVA Model Directory", placeholder="Path to your pretrained LLaVA model directory")
        batch_size = gr.Slider(minimum=1, maximum=10, value=2, step=1, label="Batch Size")
        custom_prompt = gr.Textbox(label="Custom Prompt", placeholder="Enter your custom prompt here")
    
    with gr.Row():
        gr.Examples(
            examples=[[prompt] for prompt in default_prompts],
            inputs=custom_prompt,
            label="Default Prompts",
            examples_per_page=3
        )
    
    progress_updates = gr.Textbox(label="Progress Updates")

    gr.Interface(
        fn=run_script,
        inputs=[input_dir, output_dir, llava_dir, batch_size, custom_prompt],
        outputs=progress_updates,
        title="Batch Image Captioning with LLaVA",
        description='''This tool captions images in batches using a LLaVA model to generate descriptions. 
        It was designed with intent to use the <a href="https://huggingface.co/HuggingFaceH4/vsft-llava-1.5-7b-hf-trl" target="_blank">Vision Supervised Fine Tuning (VSFT) LLaVA 1.5 7b model</a>.
        For more information for this model, please refer to the <a href="https://arxiv.org/abs/2401.10222" target="_blank">paper</a>.
        </br></br>
        Note: This batch processor should be capable of loading any LLaVA model of your choosing, should the above model not suit your needs.'''
    )

if __name__ == "__main__":
    interface.launch(server_name='0.0.0.0', server_port=7860)
