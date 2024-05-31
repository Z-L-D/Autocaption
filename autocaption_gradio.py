import gradio as gr
import os
import yaml
from autocaption_util import chat, captioning, load_dataset, parse_args

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

def run_chat_script(llm_dir, prompt, image, keep_loaded_chat):
    # Prepare arguments
    args = parse_args()
    args.llm_dir = llm_dir
    args.image = image
    args.prompt = prompt
    args.keep_loaded = keep_loaded_chat  # Pass the checkbox state to the args

    chat_updates = chat(args)
    for update in chat_updates:
        yield update

def run_captioning_script(input_dir, output_dir, llm_dir, batch_size, custom_prompt, keep_loaded_captioning):
    # Prepare arguments
    args = parse_args()
    args.input_dir = input_dir
    args.output_dir = output_dir
    args.llm_dir = llm_dir
    args.batch_size = int(batch_size)
    args.prompt = custom_prompt
    args.keep_loaded = keep_loaded_captioning  # Pass the checkbox state to the args
    
    # Run captioning function from the script and capture the generator output
    progress_updates = captioning(args) 
    for update in progress_updates:
        yield update

def load_samples(folder_path):
    return load_dataset(folder_path)

def display_sample(sample):
    image_path, text = sample
    print("Displaying image:", image_path)
    return image_path, text

# Disable Gradio Analytics
gr.Blocks(analytics_enabled=False)
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

# Define Gradio interface
with gr.Blocks(title="Autocaption") as interface:
    gr.Markdown("# Autocaption")
    gr.Markdown("### v1.0")

    with gr.Row():
        llm_dir = gr.Dropdown(
            choices=llm_choices, 
            label="LLM Model Directory", 
            value=config['selected_llm']['value'],
            interactive=True
        )
    with gr.Row(): 
        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column():                    
                    with gr.Group():
                        custom_prompt_chat = gr.Textbox(label="Your description prompt", placeholder="Type your description prompt here")
                        examples = gr.Examples(
                                examples=[[prompt] for prompt in prompts],
                                inputs=custom_prompt_chat,
                                label="Prompt Library",
                                examples_per_page=6
                            ) 
                    keep_loaded_chat = gr.Checkbox(label="Keep LLM Loaded After Captioning", value=True)  # Add the checkbox
                    submit_button = gr.Button("Submit")
                    image = gr.Image(label="Upload Image", interactive=True)

                with gr.Column():
                    # chatbox = gr.Chatbot()
                    chatbox = gr.Label(label="Image Caption")

            submit_button.click(
                fn=run_chat_script, 
                inputs=[llm_dir, custom_prompt_chat, image, keep_loaded_chat], 
                outputs=chatbox,
            ).success()
            
        with gr.Tab("Batch Captioning"):
            with gr.Row():
                progress_updates = gr.Label(label="Captioning Progress")
            with gr.Row():
                with gr.Column():
                    input_dir = gr.Textbox(label="Image Input Directory", placeholder="Path to your input directory containing images")
                    output_dir = gr.Textbox(label="Caption Output Directory", placeholder="Path to your output directory for text files")
                    with gr.Row():
                        batch_size = gr.Slider(minimum=1, maximum=10, value=2, step=1, label="Batch Size")
                        keep_loaded_captioning = gr.Checkbox(label="Keep LLM Loaded After Captioning", value=True)  # Add the checkbox
                with gr.Column():
                    with gr.Group():
                        custom_prompt_captioning = gr.Textbox(label="Description Prompt", placeholder="Enter your custom prompt here")
                        examples = gr.Examples(
                            examples=[[prompt] for prompt in prompts],
                            inputs=custom_prompt_captioning,
                            label="Prompt Library",
                            examples_per_page=6
                        )  
            with gr.Row():
                submit = gr.Button("Caption", elem_id="caption", variant="primary")
            submit.click(
                fn=run_captioning_script,
                inputs=[input_dir, output_dir, llm_dir, batch_size, custom_prompt_captioning, keep_loaded_captioning],  # Include the checkbox in inputs
                outputs=progress_updates,
            ).success()
        
        with gr.Tab("Dataset"):
            folder_path_input = gr.Textbox(label="Dataset Folder Path", placeholder="Enter the path to the dataset folder")

            load_button = gr.Button("Load Dataset")
            
            dataset = gr.Dataset(
                components=[gr.Image(), gr.Textbox()],
                samples=[],
                headers=["Image", "Text"],
                label="Dataset"
            )
        
            image_display = gr.Image(label="Selected Image")
            text_display = gr.Textbox(label="Selected Text", lines=10)
        
            load_button.click(fn=load_samples, inputs=folder_path_input, outputs=dataset, queue=True)
            dataset.select(fn=display_sample, inputs=dataset, outputs=[image_display, text_display], queue=True)

    llm_dir.change(fn=update_selected_llm, inputs=llm_dir, outputs=None)

if __name__ == "__main__":
    interface.launch(server_name='0.0.0.0', server_port=7860)
