import torch
from diffusers import AutoPipelineForText2Image

print("Cargando el modelo...")

modelo = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float32
            )

modelo = modelo.to("cpu")

prompt = input("Escribe el prompt de la imagen que quieres crear: ")
negative_prompt = "blurry, low quality, distorted, bad anatomy, extra fingers, poorly drawn hands, watermark, signature, text, error, cropped, jpeg artifacts, ugly"

print("Generando imagen...")

imagen = modelo(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=2,
        guidance_scale=1.5,
        height=1024,
        width=1024
        ).images[0]

imagen.save("imagen.png")

print("Imagen guardada.")
