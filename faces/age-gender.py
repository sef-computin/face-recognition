from age_and_gender import *
from PIL import Image, ImageDraw, ImageFont
 
 
path = '../resources/'
data = AgeAndGender()
data.load_shape_predictor(path+'shape_predictor_5_face_landmarks.dat')
data.load_dnn_gender_classifier(path+'dnn_gender_classifier_v1.dat')
data.load_dnn_age_predictor(path+'dnn_age_predictor_v1.dat')
 
img = Image.open('../sanya/ya3.jpg').convert("RGB")
result = data.predict(img)
 
font = ImageFont.truetype("arial.ttf", 10)
 
for info in result:
    shape = [(info['face'][0], info['face'][1]), (info['face'][2], info['face'][3])]
    draw = ImageDraw.Draw(img)
 
    gender = info['gender']['value']
    if gender == 'male':
        gender = 'Male'
    else:
        gender = 'Female'
 
    gender_percent = int(info['gender']['confidence'])
 
    age = info['age']['value']
    age_percent = int(info['age']['confidence'])
 
    draw.text(
        (info['face'][0] - 10, info['face'][3]), f"{gender} (~{gender_percent}%)\n {age}"+
        f"years (~{age_percent}%)",
        fill='white', font=font, align='center', stroke_width=3, stroke_fill='black'
    )
    draw.rectangle(shape, outline="red", width=5)
 
img.show()