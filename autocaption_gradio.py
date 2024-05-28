import gradio as gr
import os
import yaml
from autocaption_util import main, parse_args

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load configuration
config = load_config()

# Prepare choices for dropdown
llm_choices = [(item['label'], item['value']) for item in config['llm_directories']]

# Load default prompts from config
prompts = config.get('prompts', [])

def save_config(config, config_path='config.yaml'):
    with open(config_path, 'w') as file:
        yaml.safe_dump(config, file)

def update_selected_llm(selected_llm):
    label = next((label for label, value in llm_choices if value == selected_llm), selected_llm)
    config['selected_llm'] = {'label': label, 'value': selected_llm}
    save_config(config)

def run_script(input_dir, output_dir, llava_dir, batch_size, custom_prompt, keep_loaded):
    # Prepare arguments
    args = parse_args()
    args.input_dir = input_dir
    args.output_dir = output_dir
    args.llava_dir = llava_dir
    args.batch_size = int(batch_size)
    args.custom_prompt = custom_prompt
    args.keep_loaded = keep_loaded  # Pass the checkbox state to the args
    
    # Run main function from the script and capture the generator output
    progress_updates = main(args)  # main should be a generator now
    for update in progress_updates:
        yield update

# Disable Gradio Analytics
gr.Blocks(analytics_enabled=False)
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

# Define Gradio interface
with gr.Blocks(title="Autocaption") as interface:
    gr.Markdown("## Autocaption")

    with gr.Row():
        llm_dir = gr.Dropdown(
            choices=llm_choices, 
            label="LLM Model Directory", 
            value=config['selected_llm']['value'],
            interactive=True
        )
    with gr.Row():
        progress_updates = gr.Label(label="Captioning Progress")
    with gr.Row():
        with gr.Column():
            input_dir = gr.Textbox(label="Image Input Directory", placeholder="Path to your input directory containing images")
            output_dir = gr.Textbox(label="Caption Output Directory", placeholder="Path to your output directory for text files")
            with gr.Row():
                batch_size = gr.Slider(minimum=1, maximum=10, value=2, step=1, label="Batch Size")
                keep_loaded = gr.Checkbox(label="Keep LLM Loaded After Captioning", value=True)  # Add the checkbox
        with gr.Column():
            with gr.Group():
                custom_prompt = gr.Textbox(label="Description Prompt", placeholder="Enter your custom prompt here")
                examples = gr.Examples(
                    examples=[[prompt] for prompt in prompts],
                    inputs=custom_prompt,
                    label="Library Prompt",
                    examples_per_page=6
                )  
    with gr.Row():
        submit = gr.Button("Caption", elem_id="caption", variant="primary")
    submit.click(
        fn=run_script,
        inputs=[input_dir, output_dir, llm_dir, batch_size, custom_prompt, keep_loaded],  # Include the checkbox in inputs
        outputs=progress_updates,
    ).success()

    llm_dir.change(fn=update_selected_llm, inputs=llm_dir, outputs=None)

if __name__ == "__main__":
    interface.launch(server_name='0.0.0.0', server_port=7860)
