import pandas as pd
import streamlit as st
import tensorflow as tf
import time
from PIL import Image, ImageOps
import numpy as np
import webbrowser

st.set_page_config(layout="wide")

options= st.sidebar.radio('PNEUMPREDICT MENU',options=['🏠Home','🏥About Pneumonia','🤖Application','⚠️Disclaimer','🔖Resources', '👨🏻‍💻About me'  ])

def Ho():
    st.title(":red[_Pneumpredict_]")
    st.subheader(":grey[Web App for PNEUMonia PREDICTion using X-ray image classifications]")

    home_img = "https://th.bing.com/th/id/OIP.P_SRM8TgPRk1jWMYSkeQxQHaFR?pid=ImgDet&rs=1"
    st.image(home_img, width=800)

def Ab():
    st.header(':red[What is Pneumonia?]')
    video = "https://upload.wikimedia.org/wikipedia/commons/d/d5/En.Wikipedia-VideoWiki-Pneumonia.webm"
    st.video(video, format="video/mp4", start_time=0)
    st.write("Source and further reading available at https://en.wikipedia.org/wiki/Pneumonia")
    

def Ap():
    
    @st.cache(allow_output_mutation=True)
    def load_model():
        model=tf.keras.models.load_model("./xray_model_80-20.h5")
        return model

    with st.spinner('Please wait, while the model is being loaded..'):
      model=load_model()

    def main():
      st.header(":red[Pneumonia prediction using _Pneumpredict_]")
    
    if __name__ == '__main__':
      main()

    file = st.file_uploader(" ", accept_multiple_files=False, help="Only one file at a time. The image should be of high quality")

    if file is None:
      st.subheader("Please upload an X-ray image using the browse button :point_up:")
      st.write("Sample images can be found [here](https://github.com/sabahuddinahmad/Pneumpredict/tree/main/sample_images) !")
      image1 = Image.open('./web_img/compared.JPG')
      st.image(image1, use_column_width=True)
    
    else:
      st.subheader("Thank you for uploading the image. Below you see image which you have just uploaded!")
      st.subheader("Scroll down to see the prediction results...")  
      with st.spinner('Processing your image now.......'):

        path = file

        img = tf.keras.utils.load_img(
        path, target_size=(180, 180)
        )

        st.image(img, use_column_width=True)

        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.sigmoid(predictions)

        time.sleep(2)
        st.success('Prediction complete!')
        st.subheader(
        f"This X-ray image most likely belongs to {'Infected lungs' if np.max(score) > 0.5 else 'Normal lungs'}!"
        )

def Di():
    image2 = Image.open('./web_img/disclaimer.JPG')
    st.image(image2, use_column_width=True)
    st.header('This App does not substitute for medical advice that you get from a healthcare professional!')
    st.header('') 
    st.subheader('1. Accuracy of prediction depends on the datasets which were used for training the model within this App, and also depends on the quality of image provided.')
    st.subheader('2. Do not use prediction results from this App to diagnose or treat any medical or health condition.')
    st.subheader('3. App cannot classify underlying medical reasons that corresponds to the infections, for example: bacterial, viral, smoking, etc.')
    st.subheader('4. Healthcare professional will do blood tests and other physical examinations to identify root cause of the infections.')
    
def Ci():
    st.header(':red[Dataset availibility & recommended resources:]') 
    st.subheader('')
    st.subheader("1. Dataset used for this project is available as [Chest X-ray Images at Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).")
    st.subheader("2. Above dataset is part of a [publication](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5), _Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning_.")
    st.subheader("3. Inspiration for TensorFlow implementation in image classification on above dataset was from a [Notebook on Kaggle by Amy Jang](https://www.kaggle.com/code/amyjang/tensorflow-pneumonia-classification-on-x-rays).")
    st.subheader("4. To implement TensorFlow in image classification, there is an amazing [tutorial](https://www.tensorflow.org/tutorials/images/classification).")

def Me():
    st.header(':red[About myself:]') 
    st.subheader('')
    st.subheader('I am called Dr. Sabahuddin Ahmad, but you can also call me _Sabah_. I developed this application as a part of final project for Data Science Bootcamp at WBS Coding School, Berlin, Germany.')
    st.subheader('Before starting this bootcamp, I studied Bioinformatics in India and have recently completed my doctoral degree in Computational Biochemistry from University of Duesseldorf, Germany. I have working experience of nearly eight years in computer aided drug design, identifying novel and therapeutically potential protein-drug targets using molecular modeling and molecular dynamics simulation studies. As an outcome, I contributed to over ten [publications](https://scholar.google.com/citations?user=fvF831wAAAAJ&hl=en) in high impact journals. I also have experience of nearly three years in high-performance computing (JUWELS).')
    st.subheader('I will be happy to connect with you on [LinkedIn](https://www.linkedin.com/in/sabahuddinahmad), [Twitter](https://twitter.com/sabahahmad_IN) or on [Instagram](https://www.instagram.com/dr.sabahuddinahmad/).')   
          

if options == '🏠Home':
    Ho()
elif options == '🏥About Pneumonia':
    Ab()
elif options == '🤖Application':
    Ap()
elif options == '⚠️Disclaimer':
    Di()
elif options == '🔖Resources':
    Ci()
elif options == '👨🏻‍💻About me':
    Me()

      
