from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import numpy as np
import cv2 as cv
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

model = load_model('mnist_recognizer.h5')


def predict_digit(img):
    # resize image to 28x28 pixels
    img = img.resize((28, 28))
    # convert rgb to grayscale
    img = img.convert('L')
    img = np.invert(np.array(img))
    print('Image shape before reshaping:', img.shape)
    # show picture
    # plt.imshow(img)
    # plt.show()
    # reshaping to support our model input and normalizing
    img = img.reshape(1,-1)
    print('After reshaping:', img.shape)
    img = img / 255.0
    # predicting the class
    res = model.predict(img)[0]
    print("Probabilities:")
    print(res.tolist())
    return np.argmax(res), max(res)


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        # Creating elements:
        #Canvas - widget to draw digits in
        #Label - text
        #Two buttons - to recognize digit and clear canvas
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="-", font=("Arial", 36))
        self.classify_btn = tk.Button(self, text="Recognise digit", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear canvas", command=self.clear_all)
        # Grid structure:
        #Placing our elements in grid
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        #B1-Motion - The mouse is moved, with mouse button 1 being held down
        #Connecting this event with draw_lines func
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        #Clear canvas
        self.canvas.delete("all")

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()  # get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND)  # get the coordinate of the canvas
        img = ImageGrab.grab(rect) # grab an image
        digit, acc = predict_digit(img) #predict digit
        self.label.configure(text=str(digit) + ', ' + str(int(acc * 100)) + '%') #show info

    def draw_lines(self, event):
        #Draw figure
        #'r' is needed for line thickness
        self.x = event.x
        self.y = event.y
        r = 10
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')

app = App()
mainloop()