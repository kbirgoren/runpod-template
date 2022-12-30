import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from utils import timestamp_as_string
import os
import argparse


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Simple example of a generation script.")
    parser.add_argument(
        "--total_images",
        type=str,
        default=None,
        required=True,
        help="Total image that you want to generate.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        required=True,
        help="Prompt to generate image",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        required=False,
        help="Negative prompt to generate image",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default='runwayml/stable-diffusion-v1-5',
        required=False,
        help="Model id to be used in generation",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main(args):
    total_image = args.total_image
    prompt = args.prompt
    model_id = args.model_id
    negative_prompt = args.prompt

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Cuda is available")

    print("Setting pipeline")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16)

    print("Setting scheduler")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    print("Running pipeline")
    pipe = pipe.to("cuda")

    generated_images = []

    for i in range(int(total_image)):

        current_str = str(i+1)
        total_str = str(total_image)

        print("Generating image " + current_str + "/" + total_str + "")

        now = timestamp_as_string()
        filename = now + "-" + str(i) + ".png"

        image = pipe(prompt,
                     negative_prompt,
                     height=512,
                     width=512,
                     guidance_scale=7.5,
                     num_inference_steps=20).images[0]
        image.save("output/" + filename)
        generated_images.append(filename)


if __name__ == "__main__":
    args = parse_args()
    main(args)
