from PIL import Image


class CroppedImage:
    def __init__(self, parent_path: str, box: tuple, cls: str):
        self.parent_path = parent_path
        self.box = box
        self.cls = cls

    def get_cropped_image(self) -> Image.Image:
        try:
            image = Image.open(self.parent_path).convert("RGB")
            return image.crop(self.box)
        except Exception as e:
            print(f"Error cropping image at {self.parent_path}: {e}")
            raise
