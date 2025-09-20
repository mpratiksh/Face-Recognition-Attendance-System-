import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import face_recognition
import os
import cv2
import datetime

st.set_page_config(page_title="AI Face Recognition Attendance System", page_icon=":white_check_mark:")
with st.sidebar:
    selected = option_menu("ATTENDANCE SYSTEM MENU", ["HOME", 'ADD_EMPLOYEE', "ATTENDANCE", "DATA"],
                           icons=['house', 'person-add', 'person-check', 'database-add'], menu_icon="cast",
                           default_index=0)

def check():
    st.title("FACE RECOGNITION ATTENDANCE SYSTEM")

    # Initialize Streamlit session state
    if 'run' not in st.session_state:
        st.session_state.run = False

    # Function to mark attendance
    from datetime import datetime

    def markAttendance(name):
        with open('face_detection/Attendance.csv', 'a+') as f:
            # Move the file pointer to the beginning to read existing data
            f.seek(0)
            myDataList = f.readlines()
            nameList = [line.split(',')[0] for line in myDataList]
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%Y-%m-%d %H:%M:%S')  # Include date and time
                f.write(f'\n{name},{dtString}')

    # Function to find encodings for known images
    def findencodings(images):
        encodelist = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            try:
                encode = face_recognition.face_encodings(img)[0]
                encodelist.append(encode)
            except IndexError:
                print(f"Encoding failed for {img}.")
        return encodelist

    # Main Streamlit app
    if st.button("Start"):
        st.session_state.run = True
    st.write("PRESS 'q' to Stop the camera! ")

    if st.session_state.run:
        path = 'face_detection/known_people'
        images = []
        classnames = []
        mylist = os.listdir(path)

        for cl in mylist:
            curimage = cv2.imread(f'{path}/{cl}')
            images.append(curimage)
            classnames.append(os.path.splitext(cl)[0])

        encodelistknown = findencodings(images)
        st.write("Encoding complete")

        cap = cv2.VideoCapture(0)

        while st.session_state.run:
            success, img = cap.read()
            if not success:
                st.error("Failed to access the webcam.")
                break

            imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

            facecurframe = face_recognition.face_locations(imgs)
            if facecurframe:
                encodecurframe = face_recognition.face_encodings(imgs, facecurframe)

                for encodeFace, faceLoc in zip(encodecurframe, facecurframe):
                    matches = face_recognition.compare_faces(encodelistknown, encodeFace)
                    faceDis = face_recognition.face_distance(encodelistknown, encodeFace)
                    matchIndex = np.argmin(faceDis)

                    if matches[matchIndex]:
                        name = classnames[matchIndex].upper()
                    else:
                        name = "Unknown"

                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    markAttendance(name)

            cv2.imshow('Webcam', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        st.subheader("Complete")


def add():
    def save_image(image, username, user_id):
        image_path = f"face_detection/known_people/{username}_{user_id}.jpg"
        image.save(image_path)
        st.success(f"Image saved successfully as {username}_{user_id}.jpg")

        # Append user details to Attendance.csv
        attendance_path = "face_detection/Attendance_marked.csv"
        if not os.path.exists(attendance_path):
            with open(attendance_path, "w") as f:
                f.write("Username,User_ID\n")
        df = pd.read_csv(attendance_path)
        new_row = pd.DataFrame({"Username": [username], "User_ID": [user_id]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(attendance_path, index=False)
        st.success(f"User ID {user_id} for {username} saved to Attendance.csv")

    def capture_image():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imshow('Live Image', frame)
                if cv2.waitKey(1) & 0xFF == ord('c'):
                    break
        cap.release()
        cv2.destroyAllWindows()
        return frame

    def face_detection(image):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces

    def mark_attendance(username, user_id):
        attendance_path = "face_detection/Attendance_marked.csv"  # Change the file path
        if not os.path.exists(attendance_path):
            with open(attendance_path, "w") as f:
                f.write("Username,User_ID,Attendance\n")

        df = pd.read_csv(attendance_path)
        # Check if the user is already marked present
        if not ((df['Username'] == username) & (df['User_ID'] == user_id)).any():
            new_row = pd.DataFrame({"Username": [username], "User_ID": [user_id], "Attendance": ["Present"]})
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(attendance_path, index=False)
            st.success(f"Attendance marked for {username} with User ID {user_id}")
        else:
            st.warning(f"Attendance already marked for {username} with User ID {user_id}")

    def main():
        st.subheader("USERNAME_USER_ID")
        user_input = st.text_input("Enter Employee name_Employee ID")
        # Check if an image already exists for the user
        existing_images = []
        if os.path.exists("face_detection/known_people"):
            existing_images = [file.split("_")[0] for file in os.listdir("face_detection/known_people")]

        # Split user input into username and user_id
        user_info = user_input.split("_")
        if len(user_info) != 2:
            st.error("Please enter both Employee name and Employee ID separated by '_'")
            st.stop()  # Stop further execution of the script if input is invalid

        username, user_id = user_info

        if username in existing_images:
            st.warning("An image already exists for this user. You cannot add another.")
            st.stop()  # Stop further execution of the script if an image exists

        st.title("Live Camera Capture")

        capture_button = st.button("Capture Live Image")
        if capture_button:
            st.write("Press 'c' to capture an image.")
            frame = capture_image()
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            st.image(image, caption="Live Image", use_column_width=True)

            faces = face_detection(frame)
            if len(faces) > 0:
                save_image(image, username, user_id)
                mark_attendance(username, user_id)
            else:
                st.warning("No face detected. Please try again.")
        else:
            st.write("Click capture to take a picture.")

    if __name__ == "__main__":
        main()


def data():
    data = pd.read_csv("face_detection/Attendance.csv")
    st.title("Your attendance data")
    st.subheader(f"Today : {datetime.date.today()}")
    st.dataframe(data)


if selected == "HOME":
    st.title("Face Recognition Attendance System")
    st.write("This project is a demonstration of a face recognition-based attendance system.")
    st.write(
        "It uses computer vision techniques to recognize faces in images or video streams, and records attendance based on recognized faces.")

    # About page
    st.write("This project was created by Preksha Gohel")
    st.write("For any inquiries or feedback, please contact Preksha Gohel")


elif selected == "ATTENDANCE":
    check()

elif selected == "ADD_EMPLOYEE":
    add()

elif selected == "DATA":
    data()