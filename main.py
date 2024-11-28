############################################# IMPORTING ################################################
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess
import tkinter.simpledialog as tsd
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time

############################################# FUNCTIONS ################################################

def assure_path_exists(path):
    """Ensure a directory path exists."""
    if not os.path.exists(path):
        os.makedirs(path)


def tick():
    """Update the clock display every 200ms."""
    time_string = time.strftime('%H:%M:%S')
    clock.config(text=time_string)
    clock.after(200, tick)


def contact():
    """Show contact information."""
    mess.showinfo(title='Contact Us', message="For support, contact us at: 'arunvijo2004@gmail.com'")


def check_haarcascadefile():
    """Check if the Haar Cascade file exists."""
    if not os.path.isfile("haarcascade_frontalface_default.xml"):
        mess.showerror(title='Missing File', message='The Haar Cascade file is missing. Please contact support.')
        window.destroy()


def save_pass():
    """Save or change the system password."""
    assure_path_exists("TrainingImageLabel/")
    file_path = "TrainingImageLabel/psd.txt"

    if os.path.isfile(file_path):
        with open(file_path, "r") as tf:
            key = tf.read().strip()
    else:
        key = None

    old_pass = old.get().strip()
    new_pass = new.get().strip()
    confirm_pass = nnew.get().strip()

    if key is None or old_pass == key:
        if new_pass == confirm_pass:
            with open(file_path, "w") as tf:
                tf.write(new_pass)
            mess.showinfo(title='Success', message='Password changed successfully!')
            master.destroy()
        else:
            mess.showerror(title='Error', message='New passwords do not match.')
    else:
        mess.showerror(title='Error', message='Incorrect old password.')


def change_pass():
    """Open a window to change the password."""
    global master, old, new, nnew
    master = tk.Toplevel(window)
    master.geometry("400x160")
    master.resizable(False, False)
    master.title("Change Password")
    master.configure(background="white")

    tk.Label(master, text='Enter Old Password:', bg='white', font=('Helvetica', 12)).place(x=10, y=10)
    old = tk.Entry(master, width=25, fg="black", show='*')
    old.place(x=180, y=10)

    tk.Label(master, text='Enter New Password:', bg='white', font=('Helvetica', 12)).place(x=10, y=45)
    new = tk.Entry(master, width=25, fg="black", show='*')
    new.place(x=180, y=45)

    tk.Label(master, text='Confirm New Password:', bg='white', font=('Helvetica', 12)).place(x=10, y=80)
    nnew = tk.Entry(master, width=25, fg="black", show='*')
    nnew.place(x=180, y=80)

    tk.Button(master, text="Cancel", command=master.destroy, bg="red", fg="white").place(x=200, y=120)
    tk.Button(master, text="Save", command=save_pass, bg="green", fg="white").place(x=10, y=120)


def validate_inputs(id, name):
    """Validate user inputs for ID and Name."""
    if not id.isdigit():
        mess.showerror("Invalid Input", "ID must be a numeric value.")
        return False
    if not name.replace(" ", "").isalpha():
        mess.showerror("Invalid Input", "Name must contain only alphabetic characters.")
        return False
    return True


def take_images():
    """Capture and save face images."""
    check_haarcascadefile()
    assure_path_exists("StudentDetails/")
    assure_path_exists("TrainingImage/")

    id = txt.get().strip()
    name = txt2.get().strip()

    if not validate_inputs(id, name):
        return

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    sample_num = 0

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            sample_num += 1
            cv2.imwrite(f"TrainingImage/{name}.{id}.{sample_num}.jpg", gray[y:y + h, x:x + w])
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow('Capturing Images', img)
        if cv2.waitKey(100) & 0xFF == ord('q') or sample_num >= 100:
            break

    cam.release()
    cv2.destroyAllWindows()

    with open("StudentDetails/StudentDetails.csv", 'a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([id, name])

    mess.showinfo("Success", f"Images saved for ID: {id}, Name: {name}")


def train_images():
    """Train the model using saved images."""
    check_haarcascadefile()
    assure_path_exists("TrainingImageLabel/")
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces, ids = [], []
    for img_path in os.listdir("TrainingImage/"):
        if img_path.endswith(".jpg"):
            img = Image.open(os.path.join("TrainingImage/", img_path)).convert('L')
            faces.append(np.array(img, 'uint8'))
            ids.append(int(img_path.split(".")[1]))

    recognizer.train(faces, np.array(ids))
    recognizer.save("TrainingImageLabel/Trainer.yml")

    mess.showinfo("Success", "Model trained and saved successfully!")


def track_images():
    """Recognize faces and record attendance."""
    check_haarcascadefile()
    assure_path_exists("Attendance/")
    assure_path_exists("StudentDetails/")

    # Clear treeview entries
    for child in tv.get_children():
        tv.delete(child)

    col_names = ['Id', '', 'Name', '', 'Date', '', 'Time']
    attendance = []

    # Load recognizer and Haar Cascade
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if os.path.isfile("TrainingImageLabel/Trainer.yml"):
        recognizer.read("TrainingImageLabel/Trainer.yml")
    else:
        mess.showerror("Error", "Trainer file not found. Please train the model first.")
        return

    if os.path.isfile("StudentDetails/StudentDetails.csv"):
        df = pd.read_csv("StudentDetails/StudentDetails.csv")
    else:
        mess.showerror("Error", "Student details file not found. Please register students first.")
        return

    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Capture a single frame for attendance recognition
    ret, img = cam.read()
    if not ret:
        mess.showerror("Error", "Failed to capture image. Try again.")
        cam.release()
        return

    # Convert the frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    # Process each detected face
    for (x, y, w, h) in faces:
        id, conf = recognizer.predict(gray[y:y + h, x:x + w])
        if conf < 50:  # Confidence threshold for face recognition
            ts = time.time()
            date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
            time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

            name = df.loc[df['Id'] == id]['Name'].values[0]
            attendance.append([id, '', name, '', date, '', time_stamp])

            # Draw a rectangle around the face and add text
            cv2.putText(img, f"ID: {id}, Name: {name}", (x, y - 10), font, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(img, "Unknown", (x, y - 10), font, 0.8, (0, 0, 255), 2)

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the captured image with face recognition information
    cv2.imshow("Taking Attendance", img)
    cv2.waitKey(2000)  # Wait for a brief moment to show the image

    # Close the camera window after capturing the image
    cam.release()
    cv2.destroyAllWindows()

    # Save attendance to CSV
    date = datetime.datetime.now().strftime('%d-%m-%Y')
    file_name = f"Attendance/Attendance_{date}.csv"
    
    # Check if the attendance file exists, if not, create it
    file_exists = os.path.isfile(file_name)

    try:
        with open(file_name, 'a+', newline='') as csv_file:
            writer = csv.writer(csv_file)
            
            # Write header if it's a new file
            if not file_exists:
                writer.writerow(col_names)
            
            # Append the attendance data
            for row in attendance:
                writer.writerow(row)

        # Update treeview with the newly saved attendance data
        with open(file_name, 'r') as csv_file:
            reader = csv.reader(csv_file)
            for i, line in enumerate(reader):
                if i == 0:
                    continue  # Skip header
                tv.insert('', 'end', text=line[0], values=(line[2], line[4], line[6]))

        mess.showinfo("Success", "Attendance recorded successfully!")
    except Exception as e:
        mess.showerror("Error", f"Failed to save attendance: {e}")



    
global key
key = ''

ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
day,month,year=date.split("-")

mont={'01':'January',
      '02':'February',
      '03':'March',
      '04':'April',
      '05':'May',
      '06':'June',
      '07':'July',
      '08':'August',
      '09':'September',
      '10':'October',
      '11':'November',
      '12':'December'
      }


window = tk.Tk()
window.geometry("1280x720")
window.resizable(True, False)
window.title("Face Recognition Attendance System")
window.configure(background='#1E1E1E')

# Frames
frame1 = tk.Frame(window, bg="#2E2E2E")
frame1.place(relx=0.11, rely=0.17, relwidth=0.39, relheight=0.80)

frame2 = tk.Frame(window, bg="#2E2E2E")
frame2.place(relx=0.51, rely=0.17, relwidth=0.38, relheight=0.80)

# Header
message3 = tk.Label(
    window,
    text="Face Recognition Based Attendance System",
    fg="white",
    bg="#262523",
    width=55,
    height=1,
    font=('Helvetica', 29, 'bold')
)
message3.place(x=10, y=10)

# Date and Time
frame3 = tk.Frame(window, bg="#3A3A3A")
frame3.place(relx=0.52, rely=0.09, relwidth=0.09, relheight=0.07)

frame4 = tk.Frame(window, bg="#3A3A3A")
frame4.place(relx=0.36, rely=0.09, relwidth=0.16, relheight=0.07)

datef = tk.Label(
    frame4,
    text=time.strftime('%d-%b-%Y') + "  |  ",
    fg="orange",
    bg="#262523",
    width=55,
    height=1,
    font=('Helvetica', 22, 'bold')
)
datef.pack(fill='both', expand=1)

clock = tk.Label(
    frame3,
    fg="orange",
    bg="#262523",
    width=55,
    height=1,
    font=('Helvetica', 22, 'bold')
)
clock.pack(fill='both', expand=1)
tick()

# Headings
head2 = tk.Label(
    frame2,
    text="                 For New Registrations                ",
    fg="white",
    bg="#4CAF50",
    font=('Helvetica', 18, 'bold')
)
head2.grid(row=0, column=0)

head1 = tk.Label(
    frame1,
    text="                   For Already Registered                 ",
    fg="white",
    bg="#4CAF50",
    font=('Helvetica', 18, 'bold')
)
head1.place(x=0, y=0)

# Registration Form
lbl = tk.Label(frame2, text="Enter ID", width=20, fg="white", bg="#2E2E2E", font=('Helvetica', 17, 'bold'))
lbl.place(x=80, y=55)

txt = tk.Entry(frame2, width=32, fg="black", font=('Helvetica', 15, 'bold'))
txt.place(x=30, y=90)

lbl2 = tk.Label(frame2, text="Enter Name", width=20, fg="white", bg="#2E2E2E", font=('Helvetica', 17, 'bold'))
lbl2.place(x=80, y=140)

txt2 = tk.Entry(frame2, width=32, fg="black", font=('Helvetica', 15, 'bold'))
txt2.place(x=30, y=173)

message1 = tk.Label(
    frame2,
    text="1) Take Images  >>>  2) Save Profile",
    bg="#2E2E2E",
    fg="white",
    width=39,
    height=1,
    font=('Helvetica', 15, 'bold')
)
message1.place(x=7, y=230)

message = tk.Label(
    frame2,
    text="",
    bg="#2E2E2E",
    fg="white",
    width=39,
    height=1,
    font=('Helvetica', 16, 'bold')
)
message.place(x=7, y=450)

# Attendance Section
lbl3 = tk.Label(
    frame1,
    text="Attendance",
    width=20,
    fg="white",
    bg="#2E2E2E",
    font=('Helvetica', 20, 'bold')
)
lbl3.place(x=80, y=115)

# Total Registrations
res = 0
if os.path.isfile("StudentDetails/StudentDetails.csv"):
    with open("StudentDetails/StudentDetails.csv", 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            res += 1
    res = res - 1
message.configure(text='Total Registrations till now: ' + str(res))

# Menu Bar
menubar = tk.Menu(window, relief='ridge')
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label='Change Password', command=change_pass)
filemenu.add_command(label='Contact Us', command=contact)
filemenu.add_command(label='Exit', command=window.destroy)
menubar.add_cascade(label='Help', menu=filemenu)

# Attendance Table Setup
tv = ttk.Treeview(frame1, height=13, columns=('name', 'date', 'time'))

# Configure column widths and alignments
tv.column('#0', width=82, anchor='center')  # ID Column
tv.column('name', width=130, anchor='center')  # Name Column
tv.column('date', width=133, anchor='center')  # Date Column
tv.column('time', width=133, anchor='center')  # Time Column

# Place the Treeview in the grid layout
tv.grid(row=2, column=0, padx=(10, 10), pady=(150, 10), columnspan=4)

# Set column headings
tv.heading('#0', text='ID', anchor='center')
tv.heading('name', text='NAME', anchor='center')
tv.heading('date', text='DATE', anchor='center')
tv.heading('time', text='TIME', anchor='center')


# Scrollbar
scroll = ttk.Scrollbar(frame1, orient='vertical', command=tv.yview)
scroll.grid(row=2, column=4, padx=(0, 100), pady=(150, 0), sticky='ns')
tv.configure(yscrollcommand=scroll.set)

# Buttons
clearButton = tk.Button(
    frame2,
    text="Clear",
    command=lambda: txt.delete(0, 'end'),
    fg="white",
    bg="#4CAF50",
    width=11,
    font=('Helvetica', 12, 'bold')
)
clearButton.place(x=335, y=86)

clearButton2 = tk.Button(
    frame2,
    text="Clear",
    command=lambda: txt2.delete(0, 'end'),
    fg="white",
    bg="#4CAF50",
    width=11,
    font=('Helvetica', 12, 'bold')
)
clearButton2.place(x=335, y=172)

takeImg = tk.Button(
    frame2,
    text="Take Images",
    command=take_images,
    fg="white",
    bg="#2196F3",
    width=34,
    height=1,
    font=('Helvetica', 15, 'bold')
)
takeImg.place(x=30, y=300)

trainImg = tk.Button(
    frame2,
    text="Save Profile",
    command=train_images,
    fg="white",
    bg="#f44336",
    width=34,
    height=1,
    font=('Helvetica', 15, 'bold')
)
trainImg.place(x=30, y=380)

trackImg = tk.Button(
    frame1,
    text="Take Attendance",
    command=track_images,
    fg="white",
    bg="#2196F3",
    width=35,
    height=1,
    font=('Helvetica', 15, 'bold')
)
trackImg.place(x=30, y=50)

quitWindow = tk.Button(
    frame1,
    text="Quit",
    command=window.destroy,
    fg="white",
    bg="#f44336",
    width=35,
    height=1,
    font=('Helvetica', 15, 'bold')
)
quitWindow.place(x=30, y=450)

# Finalize and Run
window.configure(menu=menubar)
window.mainloop()
