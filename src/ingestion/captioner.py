from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from dotenv import load_dotenv
import os

load_dotenv()


class ImageCaptioner:

    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        hf_token = os.getenv("HF_TOKEN")

        print(f"🔄 Loading BLIP model on {self.device}...")

        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)

        self.model.eval()

        print("✅ Captioning model loaded.")

    @torch.no_grad()
    def get_caption(self, image_path: str):

        try:

            with Image.open(image_path) as img:
                image = img.convert("RGB")
                image.thumbnail((1024, 1024))

            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            output = self.model.generate(**inputs, max_new_tokens=50)

            caption = self.processor.decode(output[0], skip_special_tokens=True)

            return caption

        except Exception as e:

            print(f"❌ Captioning failed for {image_path}: {e}")
            return None

# if __name__ == "__main__":

#     captioner = ImageCaptioner()

#     test_image = "test_image.jpg"

#     caption = captioner.get_caption(test_image)

#     print("Caption:", caption)