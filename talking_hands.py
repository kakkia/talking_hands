import streamlit as st
from PIL import Image
import tensorflow as tf 
from tensorflow import keras
import numpy as np
from tensorflow.keras import preprocessing
from tensorflow.keras import models
import matplotlib.pyplot as plt

st.title("Talking Hands")
st.header("The fine art of Italian gestures")

st.write("""
When speaking Italian it is mandatory that you express your emotions with your hands. \n
#TalkingHands is the first image classification app that will help you to express your emotions when you just can’t find the words. \n
For my final project in Data Science at the WBS Coding School, I trained a convolutional neural network to automatically classify three different hand gestures commonly used in Italy. You will be finally able to understand Italians by looking at their hands!\n
So far the model has only been trained with:""")

hp = Image.open('Desktop/WBS_bootcamp/talking_hands/images/gestures.jpg')
st.image(hp, width=None)

st.write("Do you want to give us a hand? You can upload a picture of your gesture and you will find how what it means and how to use it.")


uploaded_file = st.file_uploader(label='Upload your picture')
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.write("Classifying your picture...")
    #st.image(image, caption = "Your picture")
    #image_data = uploaded_file.getvalue() # this is so wrong
    #st.write(image_data) # this is so wrong
    # ---- i should write in here what format I want my image to be?
pred_gesture = st.button('Predict your gesture')
#st.write('Not just yet. I will upload my model later')
# ---- if i click on predict your gesture i want to activate the model and get predictions # 
class_names = ["what", "shoo", "perfect"] 

def predict_class(image):
    model = tf.keras.models.load_model('Desktop/WBS_bootcamp/talking_hands/model_th')
    #img = tf.keras.utils.load_img(image, target_size=(180, 180)) ### first problem
    img = image.resize((150,150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    #img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    label = class_names[np.argmax(score)]
    result = f"This image most likely belongs to {class_names[np.argmax(score)]}" #" with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
    return label, result
           
def class_description(label):
    
    if label == "what":
        text = st.markdown("""<h4>What the hell are you saying?</h4>
        <li>The tips of the fingers of one hand are brought sharply together to form an upward-pointing cone.</li>
        <li>The hand can either be held motionless or be shaken more or less violently up and down.</li>
        <li>How fast you move it, depends on the degree of impatience expressed.</li>
        <li>Don't be afraid of using it when someone tells you something upsetting.</li>""", True)
    elif label == "shoo":
        text = st.markdown("""<h4>Get lost!</h4>
         <li>The arm is stuck out in front and the palm rotated upward, while the face takes on a look of righteous indignation. </li>
         <li>You can slowly move the arm as to follow the person from afar until you can't see them anymore.</li>
         <li>This is used to criticize and ridicule someone's actions, words or appearance.</li>
         <li>You are politely asking someone to leave before you can actually say something about them.</li>""",True)
    else:
        text = st.markdown("""<h4>That's just perfect</h4>
        <li>This gesture express both approval and hearty satifaction.</li>
        <li>It is typical of the good-natured and contented gourmet.</li>
        <li>Use it anythime you find something delightful - and remember it's not only about food.</li>""",True)
    return text

if pred_gesture:  
    figure = plt.figure()
    plt.imshow(image)
    plt.axis('off')
    label, result = predict_class(image)
    st.subheader(result)
    
    col_img, col_text = st.columns(2)
    
    with col_img:
        st.pyplot(figure) 
    with col_text:
        class_description(label)
else:
    st.write('')
