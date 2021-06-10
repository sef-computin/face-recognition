from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFont

import face_recognition as fr


path_to_img = '../resources/man.jpg'
#path_to_img = '../resources/image1.jpeg'
image = fr.load_image_file("../resources/man.jpg")
top,right,bottom,left = fr.face_locations(image)[0]




obj = DeepFace.analyze(img_path=path_to_img, actions=['race'])

img = Image.open(path_to_img).convert("RGB")
 
font = ImageFont.truetype("arial.ttf", 20)

draw = ImageDraw.Draw(img)


draw.rectangle((top,right,bottom,left), outline="green", width=5)
draw.text(
        (left, bottom), f"Dominant race:\n European",
        fill='white', font=font, align='center', stroke_width=3, stroke_fill='black'
    )
img.show()
print(obj['dominant_race'])