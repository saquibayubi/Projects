# Importing libraries
import streamlit as st
import numpy as np
import cv2
from keras.models import load_model

#loading model
model=load_model("C:\\Users\\Hi\\Desktop\\Udemy_120_Projects\\Plant_Disease_Streamlit\\Plant_Disease_Streamlit_by_Saquib\\Plant_Disease.h5")

# Names of Classes
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

# Setting Title of App
st.title("Plant Disease Detection App made by Mohammed Saquib Ayubi")
st.markdown("Upload an Image of a Plant Leaf in .jpg format only")

#Uploading the Plant Image 

plant_image= st.file_uploader("Choose an image...", type='jpg')
submit = st.button("Predict")

# When user clicks On Predict button

if submit:

    if plant_image is not None:

        # Convert file to an Open CV image
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)

        # We specify the dtype argument as np.uint8 to ensure that the NumPy array has a datatype of unsigned 8-bit integers, which is the expected datatype for image data in OpenCV.
                        
        opencv_image = cv2.imdecode(file_bytes,1)
                        
        #  The cv2.imdecode() function takes two arguments: the file_bytes variable containing the image data as a NumPy array, and a flag that specifies the color channel order. In this case, we set the flag to 1, which specifies that the image data is in the blue-green-red (BGR) color channel order. This is the default color channel order used by OpenCV.
                        
        # Displaying the image
        st.image(opencv_image, channels='BGR') 
        st.write(opencv_image.shape)
        # Resizing the Image
        opencv_image = cv2.resize(opencv_image,(256,256))
        # Convert image to 4 dimension
        opencv_image.shape = (1,256,256,3)
        y_pred = model.predict(opencv_image)
        result= CLASS_NAMES[np.argmax(y_pred)]
        st.title(str("This is " +result.split('-')[0]+ " Leaf with " +result.split('-')[1]))
                        
#jnius==1.1.0
#xmlrpclib==1.0.1