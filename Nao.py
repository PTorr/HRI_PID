from __future__ import print_function
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
from time import sleep

import nao_pid
from nao_example import init_nao

tts, managerProxy = init_nao()
tts.setParameter("speed", 85)

root = Tk()
root.geometry('{}x{}'.format(600, 400))


# global variables.
# init timers .
t = timeit.Timer()
t1 = 0
t2 = 0
# pid parameters.
rt_kp, rt_ki, rt_kd = 2, 0.05, 0.5
# intial game parameters.
level = 0
right_shape = 0
count_right = 0

# all the objects properties
shapes = ['circle','triangle', 'square']
colors = ['red','yellow', 'blue']

# images
i1 = PhotoImage(file='01circle.png')
i2 = PhotoImage(file='02triangle.png')
i3 = PhotoImage(file='03square.png')

px = 30 # geometry for the spaces in the layout.

kid_df = pd.DataFrame(data=np.zeros([1,2]), columns=['right','rt']) # data frame for all the history of the kid's actions.

# robot actions imported from the previous file.
r = nao_pid.robot()
actions = r.actions

def nao_say(level):
    '''
    Function for what the nao say each level.
    :param level: what level we are in.
    :return: which shape is the right shape for next time.
    '''
    global t1
    i=np.random.randint(0,3)
    if level == 0:
      nao_says = 'can you show me the shape with the' + colors[i] +' color'
    if level == 1:
      nao_says = 'can you show me the,' + shapes[i] + ', shape'
    tts.post.say(nao_says)

    t1 = t.timer()
    return i+1

def nao_do(atribute):
    '''
    Function for what the nao do each level.
    :param atribute: what atribute to do
    :param level: what level we are in.
    '''
    p = ['clapping_2', 'excellent', 'you_are_so_smart_with_cheering'] # positive
    e = ['hm_lets_try_that_again','oh_no_lets_try_that_again','im_not_sure_if_thats_right'] # encourage
    # e = ['If you are going through hell, keep going', 'this too shall pass', 'Do what you can, with what you have, where you are']
    i=np.random.randint(0,3)
    at = actions['verbal'][atribute]

    textWidget.insert('1.0','\n')
    if at == 'encourage':
        # tts.post.say(e[i])
        textWidget.insert('1.0', [at,e[i]])
        managerProxy.post.runBehavior(e[i])
    elif at == 'positive':
        textWidget.insert('1.0', [at,p[i]])
        managerProxy.post.runBehavior(p[i])
        print(at, p[i])
    elif at == 'dance':
        textWidget.insert('1.0', [at,'dance_move'])
        managerProxy.post.runBehavior('dance_move')
        print(at, 'dance_move')



def helloCallBack(shape_clicked):
    '''
    The button function.
    :param shape_clicked: which shape was pressed.
    '''
    global t1, t2, right_shape, level, kid_df
    robot = nao_pid.robot()
    t2 = t.timer()
    rt = t2-t1 # response time

    s = shape_clicked - right_shape # right/ wrong = 0/1

    if s == 0:
       s1 = 100
    else:
       s1 = -100

    # this is the history.
    temp_val = [s1, rt, level]
    temp_val = np.array(temp_val, 'float')
    temp_df = pd.DataFrame(data=temp_val.reshape(1, 3), columns=['right','rt', 'level'])
    kid_df = kid_df.append(temp_df)

    # PID
    rt_pid = nao_pid.PID(rt_kp, rt_ki, rt_kd, np.array(kid_df.rt), robot.setpoint['response_time'], np.float(temp_df.rt))
    print('right = ', s, 'pid = ', rt_pid)
    vp, lvl = robot.pid_action(s, rt_pid)


    # look at the 2 last actions of the kid. and decide if the game is too hard/ easy.
    if (np.abs(np.sum(kid_df.right[-2:])) == 200) and (np.sum(kid_df.level[-2:]) != 1):
       level += lvl
       if level < 0:
          level = 0
       elif level > 1:
          level = 1
          nao_says = 'good job, you won the game is over'
          tts.post.say(nao_says)
          exit()

    nao_do(vp) # nao reaction
    st = 5
    if vp == 2:
        st = 8
    sleep(st)
    right_shape = nao_say(level) # nao instruction

    # print('response time: ', np.round(rt,2), np.round(rt_pid,2))
    # print('robot actions: ', vp, level)


def Start_callback(textWidget):
    '''
    Starting the game.
    :param textWidget: text
    '''
    global t1, right_shape
    tts.setParameter("speed", 85)
    tts.setParameter("pitchShift", 1.15)
    tts.post.say("Hi my name is Who!")
    tts.post.say("Today we are going to play together.")
    tts.post.say("Lets begin! Good Luck!")


    t1 = 0
    textWidget.insert(END, "Game starting\n Good Luck!\n")

    right_shape = nao_say(0)
    t1 = t.timer()

textWidget = Text(root, width=40, height =5, pady=2)

textWidget.grid(row=2, columnspan=7)
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

b1= Button(root, image=i1, text="BUTTON1", command=lambda: helloCallBack(1))
b1.grid(row=0, column=0, padx =px, pady = 40, columnspan = 2)

b2 = Button(root, image=i2, text="BUTTON2", command=lambda: helloCallBack(2))
b2.grid(row=0, column=2, padx = px, columnspan = 2)

b3 = Button(root, image=i3, text="BUTTON3", command=lambda: helloCallBack(3))
b3.grid(row=0, column=4, padx = px, columnspan = 2)

b4 = Button(root, text="Start/ Next", command=lambda: Start_callback(textWidget))
b4.grid(row=1, column=2, padx = px, columnspan = 2)


root.mainloop()