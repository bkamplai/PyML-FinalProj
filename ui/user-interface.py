from tkinter import *
import cv2 
from PIL import Image, ImageTk 

width: int = 600
height: int = 400

# Create a GUI app 
app = Tk() 
app.geometry(str(width) + "x" + str(height))

# Bind the app with Escape keyboard to quit app whenever pressed 
# app.bind('<Escape>', lambda e: app.quit()) 

# Create a label and display it on app 
label_widget = Label(app) 
label_widget.pack() 

# Define a video capture object 
vid = cv2.VideoCapture(0,cv2.CAP_DSHOW) 

def open_camera(): 

	# Capture the video frame by frame 
	_, frame = vid.read()

	# Convert image from one color space to other 
	opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) 

	# Capture the latest frame and transform to image 
	captured_image = Image.fromarray(opencv_image) 

	label_width = label_widget.winfo_width() # Resize the image to fit the label while maintaining aspect ratio
	label_height = label_widget.winfo_height()
	captured_image.thumbnail((label_width, label_height))  # Resize in place while preserving aspect ratio	
 
 	# Convert captured image to photoimage 
	photo_image = ImageTk.PhotoImage(image=captured_image)
	# photo_image.pack(fill=BOTH, expand=True) 
	
	# Displaying photoimage in the label 
	label_widget.photo_image = photo_image 
	label_widget.pack(fill=BOTH, expand=True)
	
 	# Configure image in the label 
	label_widget.configure(image=photo_image) 

	# Repeat the same process after every 5 seconds 
	label_widget.after(7, open_camera) 

open_camera()

# Create an infinite loop for displaying app on screen 
app.mainloop() 
