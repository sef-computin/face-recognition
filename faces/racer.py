from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFont

path_to_img = '../resources/man.jpg'
#path_to_img = '../resources/image1.jpeg'

obj = DeepFace.analyze(img_path=path_to_img, actions = ['race'])

img = Image.open(path_to_img).convert("RGB")
 
font = ImageFont.truetype("arial.ttf", 32)

draw = ImageDraw.Draw(img)
draw.text(
        (50,50), f"Dominant race: {obj['dominant_race']}",
        fill='white', font=font, align='center', stroke_width=3, stroke_fill='black'
    )
img.show()
print(obj['dominant_race'])