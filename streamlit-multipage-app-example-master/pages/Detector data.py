import streamlit as st
import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
import joblib,time

# st.title("Projects")
st.write("You have entered", st.session_state["my_input"])
x = st.session_state["my_input"]

print(x)
if x:
    def start(o):
                
        st.title("Heart Disease Detection")

        st.subheader("Enter the Deteails correctly")


        name = st.text_input(
                "Enter Your Name:")

        age = st.text_input(
                "Enter Your Age:")

        sex = st.text_input(
                "Enter Your Sex:")

        if sex.lower()== 'male':
            sex = 1
        elif sex.lower() == 'female':
            sex=0
        else:
            sex=3

        cp = st.text_input(
                "Enter Your cp value:")

        trestbps = st.text_input(
                "Enter Your TRESTBPS value:")

        chol = st.text_input(
                "Enter Your CHOL Value:")

        fbs = st.text_input(
                "Enter Your FBS VALUE:")

        restecg = st.text_input(
                "Enter Your RESTECG value:")

        thalach = st.text_input(
                "Enter Your THALACH value:")

        exang = st.text_input(
                "Enter Your EXANG value:")

        oldpeak = st.text_input(
                "Enter Your OLDPEAK value:")

        slope = st.text_input(
                "Enter Your Slope value:")

        ca = st.text_input(
                "Enter Your CA value:")

        thal = st.text_input(
                "Enter Your THAL Value:")

        x = st.button("submit")
        if x:
            # x=np.array((age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal))
            # x = np.array((1,1,1,1,1,1,1,1,1,1,1,1,1))
            if x:
                model3 = joblib.load('D:/Py/heart_disease_prediction/model.pkl')
                p = model3.predict(np.array((int(age),int(sex),int(cp),int(trestbps),int(chol),int(fbs),int(restecg),int(thalach)
                                            ,int(exang),int(oldpeak),int(slope),int(ca),int(thal))).reshape(1,-1))
                print("streamlit predictions",p[0])
                
            with st.spinner('Processing your data...'):
                time.sleep(5)
            st.success('Done!')
                
            if p[0] == 0:
                st.success("you are perfectly alright....")
            else:                
                st.error("Your Data results Disease confirmed, Please consult the Doctor")
                                
                

    o = 1
    print(o)
    # st.write(o)
        # st.title("Crop Disease Detection ")


    try:
        start(o)
    except:
        pass
    