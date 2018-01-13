from tkinter import *
from tkinter import messagebox as tkMessageBox

# from Tkinter import *
# import tkMessageBox
# from PIL import Image
# from resizeimage import resizeimage
import timeit
# import numpy as np

root = Tk()
root.geometry('{}x{}'.format(600, 400))
t = timeit.Timer()
clicked_shape = 0
t1 = 0
t2 = 0


def helloCallBack(number=None):
   global t2
   # tkMessageBox.showinfo( "Hello Python", "Hello World")
   t2 = t.timer()
   print(t2-t1)
   # PID

def Start_callback(number=None):
   global t1
   # nao says: .... todo: connect nao
   print('a')
   t1 = 0
   t1 = t.timer()

# B = Tkinter.Button(top, text ="Hello", command = helloCallBack)

i1 = PhotoImage(file='01circle.png')
i2 = PhotoImage(file='02triangle.png')
i3 = PhotoImage(file='03square.png')

path1 = '/home/torr/PycharmProjects/HRI_PID/red.gif'

px = 30

b1= Button(root, image=i1, text="BUTTON1", command=lambda: helloCallBack(1))
b1.grid(row=0, column=0, padx =px, pady = 40, columnspan = 2)


b2 = Button(root, image=i2, text="BUTTON2", command=lambda: helloCallBack(2))
b2.grid(row=0, column=2, padx = px, columnspan = 2)

b3 = Button(root, image=i3, text="BUTTON3", command=lambda: helloCallBack(3))
b3.grid(row=0, column=4, padx = px, columnspan = 2)

b4 = Button(root, text="Start/ Next", command=lambda: Start_callback(0))
b4.grid(row=1, column=2, padx = px, columnspan = 2)


# B.pack()
root.mainloop()

print('welcome to our game')


# nao says hello
shapes = {'circle':'red','triangle':'yellow', 'square':'blue' }
shapes = ['circle','triangle', 'square']
colors = ['red','yellow', 'blue']

# chose color or shape as A
# text2speach: can you show me the A shape
# Timer
# Person clicked -> close timer
# check right/ wrong
# take Timer to PID
# robot response


