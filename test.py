# -*- coding: utf-8 -*-
"""
Программа выводит изображение с веб-камеры на экран. Для закрытия программы
нажмите q.
"""

import cv2


#
# Если к ноутбуку кроме встроенной веб-камеры подключена ещё одна через USB,
# то для вывода изображения с последней следует изменить аргумент VideoCapture
# на 1
cap = cv2.VideoCapture(0)

if cap.isOpened() is False:
    print("Ошибка при попытке подключения к веб-камере")

while cap.isOpened():
    ret, frame = cap.read()
    if ret is True:
        cv2.imshow('ONTI2019', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()