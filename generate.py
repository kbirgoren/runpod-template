import torch
import os
from diffusers import StableDiffusionPipeline, DDIMScheduler
from re import sub
import time


def camel_case(s):
    s = sub(r"(_|-)+", " ", s).title().replace(" ",
                                               "").replace(",", "").replace("(", "").replace(")", "")
    return ''.join([s[0].lower(), s[1:]])


if torch.cuda.is_available():
    print("Cuda is available")

print("Setting cuda alloc_conf")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

model_id = "./trained/1100"


print("Setting pipeline")
pipeline = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16)

print("Setting scheduler")
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

print("Running pipeline")
pipeline = pipeline.to("cuda")


prompt = "formal portrait of kurtulusbirgoren as doc brown from back to the future. digital art by eugene de blaas, ross tran, and nasreddine dinet, vibrant color scheme, intricately detailed, in the style of romanticism, cinematic, artstation, greg rutkowski "

for x in range(50):
    print("Generating image")
    now = str(int(time.time()))
    image = pipeline(prompt,
                     height=512,
                     width=512,
                     guidance_scale=7.5,
                     num_inference_steps=20).images[0]

    filename = "kurtulusbirgoren"
    image.save("output/" + filename + "-" + now + "-"+str(x)+".png")
