from keras.models import load_model
import cv2
import cv2 as cv

import numpy as np
model=load_model('model_1.h5')
class_names={0:'angry',1:'happy',2:'neutral'}
cap=cv2.VideoCapture(0)

Image_size=100
channels=3
while True:
    ret,img=cap.read()
    resized_Image=cv2.resize(img,(Image_size,Image_size))
    Normalized_Image=resized_Image/255.0
    Image=np.expand_dims(Normalized_Image,0)
    print(Image.shape)

    result=model.predict(Image)
    pred=np.argmax(result)
    label = np.argmax(result, axis=1)[0]

    cv2.putText(img,class_names[label],(95,59),cv2.FONT_HERSHEY_PLAIN,
                5,(255,0,255),5)






    cv2.imshow("Image",img)

    if cv2.waitKey(20)==40:
        break


cap.release()
cv2.destroyAllWindows()






#
#
#
#
#
#
#
#
#


























