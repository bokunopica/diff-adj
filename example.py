from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

pipe = pipe.to("cuda")

prompt = "the side view of a patient on the bed in a simple style"
image = pipe(prompt).images[0]  
    
image.save("test.png")
