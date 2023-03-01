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
    st.write(":grey[Web App for PNEUMonia PREDICTion using X-ray image classifications]")

    home_img = Image.open('./web_img/home.jpg')
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

    file = st.file_uploader(" ", accept_multiple_files=False, help="Only one file at a time. The image should be of good quality")

    if file is None:
      st.subheader("Please upload an X-ray image using the browse button :point_up:")
      st.write("Sample images can be found [here](https://github.com/sabahuddinahmad/Pneumpredict/tree/main/sample_images) !")
      image1 = Image.open('./web_img/compared.JPG')
      st.image(image1, use_column_width=True)
    
    else:
      st.subheader("Thank you for uploading X-ray image!") 
      with st.spinner('Pneumpredict is now processing your image.......'):

        path = file

        img = tf.keras.utils.load_img(
        path, target_size=(180, 180)
        )

        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.sigmoid(predictions)

        time.sleep(2)
        st.success('Prediction is complete!')
        st.subheader(
        f"Uploaded X-ray image looks like this :point_down: and most likely belongs to {'Infected lungs' if np.max(score) > 0.5 else 'Normal lungs'}!"
        )
        st.image(img, width=400)
        st.subheader("Thank you for using Pneumpredict")

def Di():
    image2 = Image.open('./web_img/disclaimer.JPG')
    st.image(image2, use_column_width=True)
    st.subheader('This App does not substitute a healthcare professional!')
    st.header('') 
    st.write('1. Accuracy of prediction depends on the datasets which were used for training the model within this App, and also depends on the quality of image provided.')
    st.write('2. Do not use prediction results from this App to diagnose or treat any medical or health condition.')
    st.write('3. App cannot classify underlying medical reasons that corresponds to the infections, for example: bacterial, viral, smoking, etc.')
    st.write('4. Healthcare professional will do blood tests and other physical examinations to identify root cause of the infections.')
    
def Ci():
    st.header(':red[Dataset availibility & recommended resources:]') 
    st.subheader('')
    st.write("1. Dataset used for this project is available as [Chest X-ray Images at Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).")
    st.write("2. Above dataset is part of a [publication](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5), _Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning_.")
    st.write("3. Inspiration for TensorFlow implementation in image classification on above dataset was from a [Notebook on Kaggle by Amy Jang](https://www.kaggle.com/code/amyjang/tensorflow-pneumonia-classification-on-x-rays).")
    st.write("4. To implement TensorFlow in image classification, there is an amazing [tutorial](https://www.tensorflow.org/tutorials/images/classification).")

def Me():
    st.header(':red[About myself:]') 
    st.subheader('')
    st.write('You can call me _Sabah_. I developed this application as a part of final project for Data Science Bootcamp at WBS Coding School, Berlin, Germany.')
    st.write('Before starting this bootcamp, I studied Bioinformatics in India and have recently completed my doctoral degree in Computational Biochemistry from University of Duesseldorf, Germany. I have working experience of nearly eight years in computer aided drug design, identifying novel and therapeutically potential protein-drug targets using molecular modeling and molecular dynamics simulation studies. As an outcome, I contributed to over ten [publications](https://scholar.google.com/citations?user=fvF831wAAAAJ&hl=en) in high impact journals. I also have experience of nearly three years in high-performance computing (JUWELS).')
    st.write('I will be happy to connect with you on [LinkedIn](https://www.linkedin.com/in/sabahuddinahmad), [Twitter](https://twitter.com/sabahahmad_IN) or on [Instagram](https://www.instagram.com/dr.sabahuddinahmad/).')   
          

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

      
