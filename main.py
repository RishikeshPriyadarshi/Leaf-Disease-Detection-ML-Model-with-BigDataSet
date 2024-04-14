import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    # image_path = "home_page.jpeg"
    image_path = "2024-04-13-01-33-26.png"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our goal is to assist in promptly identifying diseases in plants. Share an image of your plant, and our system will analyze it to detect any signs of disease. Let's work together to protect our greenery and ensure thriving plant life!

    ### How It Works
    1. **Upload Image:** Navigate to the **Disease Recognition** page and submit an image of the plant showing suspected diseases.
    2. **Analysis:** : Employing advanced algorithms, our system will meticulously process the uploaded image to detect potential diseases.
    3. **Results:** Access the outcome of the analysis, accompanied by refined recommendations for subsequent steps.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset encompasses around 87,000 RGB images capturing the spectrum of healthy and diseased crop leaves, meticulously classified into 38 unique categories. It undergoes partitioning into training and validation sets, adhering to an 80/20 distribution, while upholding the integrity of the directory structure. Furthermore, a distinct directory is subsequently curated to house 33 test images, serving the purpose of predictive analysis.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        # class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        #             'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
        #             'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
        #             'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
        #             'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
        #             'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
        #             'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
        #             'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
        #             'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
        #             'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
        #             'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
        #             'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
        #             'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
        #               'Tomato___healthy']
        class_name = ['scab', 'rot', 'rust', 'healthy',
                    'healthy', 'Powdery_mildew', 
                    'healthy', 'leaf_spot Gray_leaf_spot', 
                    'Common_rust_', 'Northern_Leaf_Blight', 'healthy', 
                    'Black_rot', 'Esca_(Black_Measles)', 'Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'healthy', 'Haunglongbing_(Citrus_greening)', 'Bacterial_spot',
                    'healthy', 'Bacterial_spot', 'healthy', 
                    'Early_blight', 'Late_blight', 'healthy', 
                    'healthy', 'healthy', 'Powdery_mildew', 
                    'Leaf_scorch', 'healthy', 'Bacterial_spot', 
                    'Early_blight', 'Late_blight', 'Leaf_Mold', 
                    'Septoria_leaf_spot', 'Spider_mites Two-spotted_spider_mite', 
                    'Target_Spot', 'Yellow_Leaf_Curl_Virus', 'mosaic_virus',
                      'healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
