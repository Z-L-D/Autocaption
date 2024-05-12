import gradio as gr
import os
from autocaption_util import main, parse_args

def run_script(input_dir, output_dir, llava_dir, batch_size):
    # Prepare arguments
    args = parse_args()
    args.input_dir = input_dir
    args.output_dir = output_dir
    args.llava_dir = llava_dir
    args.batch_size = int(batch_size)
    
    # Run main function from the script and capture the generator output
    progress_updates = main(args)  # main should be a generator now
    for update in progress_updates:
        yield update

# Disable Gradio Analytics
gr.Blocks(analytics_enabled=False)
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

# Define Gradio interface
interface = gr.Interface(
    fn=run_script,
    inputs=[
        gr.Textbox(label="Input Directory", placeholder="Path to your input directory containing images"),
        gr.Textbox(label="Output Directory", placeholder="Path to your output directory for text files"),
        gr.Textbox(label="LLaMA Model Directory", placeholder="Path to your pretrained LLaVA model directory"),
        gr.Slider(minimum=1, maximum=10, value=2, step=1, label="Batch Size")
    ],
    outputs=gr.Label(),  # Updated to use Label for live data
    title="Batch Image Captioning with LLaVA",
    description='''This tool captions images in batches using a LLaVA model to generate descriptions. 
    It was designed with intent to use the <a href="https://huggingface.co/HuggingFaceH4/vsft-llava-1.5-7b-hf-trl" target="_blank">Vision Supervised Fine Tuning (VSFT) LLaVA 1.5 7b model</a>.
    For more information for this model, please refer to the <a href="https://arxiv.org/abs/2401.10222" target="_blank">paper</a>.
    </br></br>
    Note: This batch processor should be capable of loading any LLaVA model of your choosing, should the above model not suit your needs.'''
)

if __name__ == "__main__":
    interface.launch(server_name='0.0.0.0', server_port=7860)
