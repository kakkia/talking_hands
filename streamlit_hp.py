import streamlit as st

def load_image():
    uploaded_file = st.file_uploader(label='Upload your picture')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)

def main():
    st.title("Talking Hands")
    st.header("The fine art of Italian gestures")
    from PIL import Image
    image = Image.open('images/gestures.jpg')
    st.image(image, width=None)
    st.write("""
When speaking Italian it is mandatory that you express your emotions with your hands. \n
#TalkingHands is the first image classification app that will help you to express your emotions when you just can’t find the words. \n
For my final project in Data Science at the WBS Coding School, I trained a convolutional neural network to automatically classify three different hand gestures commonly used in Italy. You will be finally able to understand Italians by looking at their hands!\n
Do you want to give us a hand? You can upload a picture of your gesture and you will find how what it means and how to use it.""")
    load_image()
    
if __name__ == '__main__':
    main()

if st.button('Predict the gesture'):
     st.write('Not just yet. I am still training my model')

