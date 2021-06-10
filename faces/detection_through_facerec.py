import PIL.Image
import PIL.ImageDraw
import face_recognition as fr
import matplotlib.pyplot as plt
 
#image1 =fr.load_image_file("../resources/image1.jpeg")
image2 =fr.load_image_file("../resources/image2.jpeg")

 
print(image2)
plt.imshow(image2)


pil_image = PIL.Image.fromarray(image2)
face_loc = fr.face_locations(image2)
for face_location in face_loc:
    top,right,bottom,left = face_location
    draw_shape = PIL.ImageDraw.Draw(pil_image)
    draw_shape.rectangle([left, top, right, bottom], outline="blue", width = 5)
pil_image.save("../resources/output.jpg")
