import streamlit as st
from skimage.io import imread,imsave
from keras.models import load_model
from PIL import Image
import numpy as np
import time


# st.title("Projects")
st.write("You have entered", st.session_state["my_input"])
x = st.session_state["my_input"]

print(x)
if x:
    def start(file,o):
        img_file_buffer = file
        image = Image.open(img_file_buffer)
        st.write("input image")
        # st.image(image)
        img_array = np.array(image) # if you want to pass it to OpenCV
        img = 'streamlit-multipage-app-example-master/pages/color_img.jpg'
        imsave(img, img_array)
        # st.image(image, caption="The caption", use_column_width=True)
        # array = np.reshape(img_array, (128, 128))
        if file:
            st.info("file entered")
            st.image(file)

        button = st.button('Enter')
        f = str(file)
        f = f[25]
        print(f)
        if button:
            with st.spinner('Processing your data...'):
                time.sleep(5)
            model = load_model('model.h5')
            batch_size = 16
            image = cv2.imread(img)
            img = Image.fromarray(image)
            img = img.resize((128, 128))
            img = np.array(img)
            input_img = np.expand_dims(img, axis=0)
            print(input_img)
            print(input_img.shape)
            i = input_img.reshape(-1,1)
            print("shape-i",i.shape)
            # result = model.predict_classes(input_img)
            result = model.predict(input_img)
            print(result)
            # st.write(result)
            st.subheader('The report is..')
            print(result)
            
            if f=='f':
                st.subheader("state: Healthy state")
                
            else:
                st.subheader("Heart Cardiomegaly found, please consult a doctor ")
                st.write('''Description:
                                An enlarged heart (cardiomegaly) isn't a disease, but rather a sign of another condition. 
                                The term "cardiomegaly" refers to an enlarged heart seen on any imaging test, including a chest X-ray. 
                                Other tests are then needed to diagnose the condition that's causing the enlarged heart.''')
                st.header("Cause for the disease")
                st.write('''An enlarged heart (cardiomegaly) can be caused by damage to the heart muscle or any condition that makes the heart pump harder than usual, 
                         including pregnancy.Sometimes the heart gets larger and becomes weak for unknown reasons. 
                         This condition is called idiopathic cardiomyopathy.''')
                st.header("Treatement to cure")
                st.write('''To help lower your GH and IGF-1 levels, treatment options typically include surgery or radiation to remove or reduce the size of the tumor that is causing your symptoms, 
                         and medication to help normalize your hormone levels.''')
                
                        
                

    st.title("Heart Disease Detector")


    st.subheader("Enter your Image....")


    file = st.file_uploader("enter the image")
    print(file)
    
    o=1

    try:
        start(file,o)
    except:
        pass
    
