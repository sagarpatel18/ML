import cv2
car_cascade = cv2.CascadeClassifier("haarcascade_car.xml")

cap = cv2.VideoCapture('Traffic.mp4')

while True:
    respose, color_img = cap.read()

    if respose == False:
        break
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    faces = car_cascade.detectMultiScale(gray_img, 1.1, 1)

    for (x, y, w, h) in faces:
        cv2.rectangle(color_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.imshow('img', color_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
