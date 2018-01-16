import sys
if sys.version_info[0] > 2:
   from tkinter import *
   from tkinter import messagebox as tkMessageBox
else:
   from Tkinter import *
   import tkMessageBox
   from PIL import Image

import numpy as np
import timeit
import pandas as pd

import nao_pid
from nao_example import init_nao
tts, managerProxy = init_nao()
tts.setParameter("speed", 85)

root = Tk()
root.geometry('{}x{}'.format(600, 400))
t = timeit.Timer()
t1 = 0
t2 = 0
rt_kp, rt_ki, rt_kd = 2, 0.05, 0.5
level = 0
right_shape = 0

shapes = ['circle','triangle', 'square']
colors = ['red','yellow', 'blue']
objects = pd.DataFrame(data = np.array([shapes,colors]).T, columns=['shape','color'])


def nao_say(level):
   i=np.random.randint(0,3)
   if level == 0:
      nao_says = 'can you show me the shape with the' + objects.color[i] +' color'
   if level == 1:
      nao_says = 'can you show me the,' + objects.shape[i] + ', shape'
   tts.post.say(nao_says)
   return i+1

def nao_do(atribute, level):
   pass

def helloCallBack(shape_clicked , kid_df):
   global t1, t2, right_shape, level
   print(right_shape, shape_clicked)
   # tkMessageBox.showinfo( "Hello Python", "Hello World")
   t2 = t.timer()
   rt = t2-t1 # response time

   s = shape_clicked - right_shape # right/ wrong = 0/1
   if s == 0:
      tts.post.say('you are right')
   else:
      tts.post.say('you are wrong')
   # PID
   robot = nao_pid.robot()

   temp_val = [s, rt]
   temp_val = np.array(temp_val, 'float')
   temp_df = pd.DataFrame(data=temp_val.reshape(1, 2), columns=['right','rt'])
   kid_df = kid_df.append(temp_df.round(2))

   rt_pid = nao_pid.PID(rt_kp, rt_ki, rt_kd, np.array(kid_df.rt), robot.setpoint['response_time'], np.float(temp_df.rt))
   vp, lvl = robot.pid_action(s, rt_pid)
   level += lvl
   if level < 0:
      level = 0
   elif level > 1:
      level = 1
       # todo: good job, game over.

   # todo: nao do vp atribute and level
   # nao_do(vp, level)

   right_shape = nao_say(level)
   t1 = t.timer()

   # shape_clicked = 9
   # while shape_clicked == 9:
   #    print('a')

   print('response time: ', np.round(rt,2), np.round(rt_pid,2))
   print('robot actions: ', vp, level)


def Start_callback(number, textWidget):
   global t1, right_shape
   tts.setParameter("speed", 85)
   tts.setParameter("pitchShift", 1.15)
   # tts.post.say("Hi my name is Who!")
   # tts.post.say("Today we are going to play together.")
   # tts.post.say("Lets begin! Good Luck!")


   t1 = 0
   t1 = t.timer()
   textWidget.insert(END, "Game starting\n Good Luck!\n")

# B = Tkinter.Button(top, text ="Hello", command = helloCallBack)

i1 = PhotoImage(file='01circle.png')
i2 = PhotoImage(file='02triangle.png')
i3 = PhotoImage(file='03square.png')

path1 = '/home/torr/PycharmProjects/HRI_PID/red.gif'

px = 30

kid_df = pd.DataFrame(data=np.zeros([1,2]), columns=['right','rt'])

textWidget = Text(root, width=20, height =5, pady=2)

textWidget.grid(row=2, column=2)
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

b1= Button(root, image=i1, text="BUTTON1", command=lambda: helloCallBack(1, kid_df))
b1.grid(row=0, column=0, padx =px, pady = 40, columnspan = 2)


b2 = Button(root, image=i2, text="BUTTON2", command=lambda: helloCallBack(2, kid_df))
b2.grid(row=0, column=2, padx = px, columnspan = 2)

b3 = Button(root, image=i3, text="BUTTON3", command=lambda: helloCallBack(3, kid_df))
b3.grid(row=0, column=4, padx = px, columnspan = 2)

b4 = Button(root, text="Start/ Next", command=lambda: Start_callback(0, textWidget))
b4.grid(row=1, column=2, padx = px, columnspan = 2)


root.geometry('600x400')
root.after(1000)
root.mainloop()

# nao says hello


# chose color or shape as A
# text2speach: can you show me the A shape
# Timer
# Person clicked -> close timer
# check right/ wrong
# take Timer to PID
# robot response

#
# if __name__ == '__main__':
#