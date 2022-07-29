
import numpy as np
import cv2
import streamlit as st
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# load model


classifier =load_model('models/model.h5')

# load weights into new model
classifier.load_weights("models/model.h5")

emotion_dict = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]
#load face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)



            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
                label_position = (x, y)
                cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
        return img

def main():
    # Face Analysis Application #
    st.title("Visual Sentiment Detection Application👧")
    activities = ["Home", "Face emotion recognition", "About Creator"]
    st.sidebar.image("image/1657772826245-modified (1).png",use_column_width=True)
    st.sidebar.text('-------------Saugata Deb--------------')
       
    choice = st.sidebar.selectbox("Dashboard", activities)
    
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#afbaa7;padding:10px">
                                            <h4 style="color:black;text-align:center;">
                                            Welcome guys🖐 , I am here to guide you.😇</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        html_temp_home1 = """<div style="background-color:#baa7af;padding:10px">
                                            <h4 style="color:black;text-align:center;"> 
                                            You can choose options from the dashboard to go further with the application 👈👆 </h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        html_temp_home1 = """<div style="background-color:#a9a7ba;padding:10px">
                                            <h4 style="color:black;text-align:center;"> 
                                            Let's see your lovely face😏😟😃😡😲😧</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        


    elif choice == "Face emotion recognition":
        st.header("Live Feed:camera:")
        st.write("Press on Start Button to configure your camera. I can't wait further to see your lovely reactions.😃 ")
        webrtc_streamer(key="example", video_processor_factory=VideoTransformer)

    elif choice == "About Creator":
        st.subheader("About Creator 👨‍🔧")
        html_temp_about1= """<div style="background-color:#9fbecd;padding:10px">
                                    <h4 style="color:balck;text-align:center;">
                                    This application has been developed by Saugata Deb</h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)
        html_temp_about1= """<div style="background-color:#9a9bc0;padding:10px">
                                    <h4 style="color:black;text-align:center;">
                                    To know more about the creator. 
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)
        st.write("Log onto: https://www.linkedin.com/in/saugata-deb")

        
        html_temp_about1= """<div style="background-color:#afbaa7;padding:10px">
                                    <h4 style="color:black;text-align:center;">
                                    Workplace of the creator.</h4> 
                                    </div>
                                    </br>"""
            
                                    
        st.markdown(html_temp_about1, unsafe_allow_html=True)
        st.write("Log onto: https://github.com/SaugataDeb")
        
    else:
        pass


if __name__ == "__main__":
    main()
