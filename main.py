import torch
from diffusers import DiffusionPipeline

# Load the base and refiner models
def load_model(model_name: str) -> DiffusionPipeline:
    """
    Loads a diffusion model pipeline.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        DiffusionPipeline: The loaded model pipeline.
    """
    return DiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")

# Initialize the base and refiner models
base_model_name = "stabilityai/stable-diffusion-xl-base-1.0"
refiner_model_name = "stabilityai/stable-diffusion-xl-refiner-1.0"

base = load_model(base_model_name)
refiner = load_model(refiner_model_name)
refiner.text_encoder_2 = base.text_encoder_2
refiner.vae = base.vae

# Define inference settings
n_steps = 40
high_noise_frac = 0.8
prompt = "A majestic lion jumping from a big stone at night"

# Run both models to generate the final image
latent_image = base(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent"
).images

final_image = refiner(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=latent_image
).images[0]
