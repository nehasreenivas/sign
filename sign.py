import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
import os
import speech_recognition as sr

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

DEMO_VIDEO = 'demo.mp4'
DEMO_IMAGE = 'demo.jpg'

my_list = []

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Sign Language Detection - Sameer Edlabadkar')
st.sidebar.subheader('-Parameter')


@st.cache_data
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    # Check to see if the width is None
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


app_mode = st.sidebar.selectbox('Choose the App mode', ['About App', 'Sign Language to Text', 'Speech to Sign Language'])

if app_mode == 'About App':
    st.title('Sign Language Detection Using MediaPipe with Streamlit GUI')
    st.markdown('In this application, we are using **MediaPipe** for detecting Sign Language. **SpeechRecognition** library of Python to recognize the voice and a machine learning algorithm which converts speech to Indian Sign Language. **Streamlit** is used to create the Web Graphical User Interface (GUI).')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.video('https://youtu.be/NYAFEmte4og')
    st.markdown('''
    # About Me
    Hey, this is **Sameer Edlabadkar**. Working on technologies such as **TensorFlow, MediaPipe, OpenCV, ResNet50**.
    Also, check me out on Social Media:
    - [YouTube](https://www.youtube.com/@edlabadkarsameer/videos)
    - [LinkedIn](https://www.linkedin.com/in/sameer-edlabadkar-43b48b1a7/)
    - [GitHub](https://github.com/edlabadkarsameer)
    
    If you are facing any issues while working, feel free to mail me at **edlabadkarsameer@gmail.com**.
    ''')
elif app_mode == 'Sign Language to Text':
    st.title('Sign Language to Text')

    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox("Record Video")
    if record:
        st.checkbox("Recording", value=True)

    st.sidebar.markdown('---')
    sameer = ""
    st.markdown('## Output')
    st.markdown(sameer)

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v'])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO
    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    codec = cv2.VideoWriter_fourcc('V', 'P', '0', '9')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.sidebar.markdown('---')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4

    while True:
        ret, img = vid.read()
        if not ret:
            break
        img = cv2.flip(img, 1)
        h, w, c = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img)

        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                lm_list = []
                for id, lm in enumerate(hand_landmark.landmark):
                    lm_list.append(lm)
                finger_fold_status = []
                for tip in finger_tips:
                    x, y = int(lm_list[tip].x * w), int(lm_list[tip].y * h)

                    if lm_list[tip].x < lm_list[tip - 2].x:
                        finger_fold_status.append(True)
                    else:
                        finger_fold_status.append(False)

                print(finger_fold_status)
                x, y = int(lm_list[8].x * w), int(lm_list[8].y * h)

                # Recognition Logic
                if lm_list[3].x < lm_list[4].x and lm_list[8].y > lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
                    cv2.putText(img, "fuck off !!!", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    sameer = "fuck off"

                # One
                if lm_list[3].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y > lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y and lm_list[4].y < lm_list[12].y:
                    cv2.putText(img, "ONE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("1")

                # Two
                if lm_list[3].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
                    cv2.putText(img, "TWO", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("2")
                    sameer = "two"

                # Three
                if lm_list[2].x < lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
                    cv2.putText(img, "THREE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("3")
                    sameer = "three"

                # Four
                if lm_list[2].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[2].y < lm_list[8].y:
                    cv2.putText(img, "FOUR", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("4")
                    sameer = "four"

                # Five
                if lm_list[3].x < lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                        lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y:
                    cv2.putText(img, "FIVE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    my_list.append("5")
                    sameer = "five"

                # Send Output to App
                st.markdown('## Output')
                st.markdown(sameer)
                stframe.image(img, channels="BGR")

        if record:
            out.write(img)

    vid.release()
    out.release()
    cv2.destroyAllWindows()

elif app_mode == 'Speech to Sign Language':
    st.title('Speech to Sign Language')
    st.sidebar.markdown('---')
    st.sidebar.subheader('Speech Recognizer')
    st.markdown('## Speech to Text')
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.sidebar.text("Listening...")
        st.sidebar.markdown("Press `CTRL + A` to record")
        audio_data = r.listen(source)
        st.sidebar.text("Done recording")
    
    st.markdown("## Transcribing...")
    try:
        st.sidebar.text("Transcribing...")
        text = r.recognize_google(audio_data, language='en-IN')
        st.success(text)
        st.sidebar.success(text)
        st.markdown("## Text to Sign Language")
        if "hello" in text:
            st.image('hello.jpg')
        elif "thanks" in text:
            st.image('thanks.jpg')
        elif "bye" in text:
            st.image('bye.jpg')
        else:
            st.image('default.jpg')
    except sr.UnknownValueError:
        st.error("Sorry, I could not understand the audio")
    except sr.RequestError:
        st.error("Could not request results from Google Speech Recognition service")
