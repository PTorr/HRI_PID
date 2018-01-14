import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.animation as animation
# from __future__ import print_function

class robot():
    '''This is an object for robot simulation.'''

    def __init__(self):
        actions = {'verbal': ['encourage', 'positive', 'brake'], 'physical': ['encourage', 'positive', 'dance'],
                   'level': ['decrease', 'stay', 'increase']}
        # self.actions       = pd.DataFrame.from_dict(actions)
        self.actions  = actions
        self.setpoint = {'response_time': 1, 'gaze': 0, 'lips': 15}


    def pid_action(self, right, rt_pid):
        '''
        Make action based on PID values.
        :param rt_pid: response iteration PID.
        :return: vp - which tribute to take.
                 lvl - which level to take.
        '''
        rt_th = 0.5
        rt_max = 4
        if np.abs(rt_pid) > rt_max: # looking away
            vp = 2 # break + dance
            lvl = 0 # stay at the same level
        elif np.abs(rt_pid) <= rt_max: # looking at
            if right == 0:
                if np.abs(rt_pid) <= rt_th: # fast
                    vp  = 1 # positive
                    lvl = 1 # increase level
                elif np.abs(rt_pid) >= rt_th: # slow
                    vp  = 0 # encouraging
                    lvl = 0 # stay at the same level
            elif right != 0:
                vp = 1 # encouraging
                lvl = -1 # decrease level
        return vp, lvl


    def prnt_selected_action(self,i,j,k):
        '''
        Print the action the robot did.
        :param i: which physical tribute.
        :param j: which verbal tribute.
        :param k: which level.
        :return: None
        '''
        if type(i) == int:
            print('robot action = ', self.actions['verbal'][i], 'level = ', self.actions['level'][k])
        elif type(i) == list:
            print('robot action= ', self.actions['verbal'][i[0]], self.actions['verbal'][i[1]], 'level = ', self.actions['level'][k])
        print()

def PID(kp, ki, kd, pv, sp, rt):
    '''
    PID calculations.
    :param kp: Kp for PID.
    :param ki: Ki for PID.
    :param kd: Kd for PID.
    :param pv: a data frame with kid's values.
    :param sp: setpoint.
    :param rt: response rate.
    :return: PID value.
    '''
    # http: // code.activestate.com / recipes / 577231 - discrete - pid - controller /

    et = -(pv - sp)

    P = et[-1] # PV(t) = et[-1] # propotional

    I = np.nansum(et) # integral

    D = (et[-1] - et[-2])/rt # derivative

    pid = kp*P + ki*I + kd*D

    return pid


def simulate_kid_data(path, kid, robot, N, prt = True):
    '''
    Simulation of the interaction.
    :param path: where to save the data.
    :param kid: kid object.
    :param robot: robot object.
    :param N: number of iterations.
    :return: data frame of everything.
    '''
    krand = 1
    # PID parameters for all the PID process that we calculate.
    rt_kp, rt_ki, rt_kd       = 2, 0.05, 0.5

    ttl = 'rt: '+str(rt_kp)+'/'+str(rt_ki)+'/'+str(rt_kd)

    kid.simulate() # intial state
    # Creating the dataframe to save the history of the kid's action.
    temp_val = kid.__dict__.values() # taking the values.
    temp_val = np.append(temp_val, 0) # adding 1st step.
    temp_val = np.array(temp_val, 'float') # converting to float.
    cls_names_kid = kid.__dict__.keys() # taking the names of the fields in kid.
    cls_names_kid.append('iteration') # addin 't' to the iteration column as a name.
    kid_df = pd.DataFrame(data=temp_val.reshape(1, 6), columns=cls_names_kid) # create the dataframe
    cls_names_pid = ['rt_pid', 'iteration']
    cls_names_robot = ['verbal', 'physical', 'level', 'iteration']
    # Simulating the interaction.


    right_counter = 0
    for t in range(1, N):  # N iteration steps simulation
        temp_val = kid.__dict__.values()
        temp_val.append(t)
        temp_val = np.array(temp_val, 'float')
        temp_df = pd.DataFrame(data=temp_val.reshape(1, 6), columns=cls_names_kid)
        kid_df = kid_df.append(temp_df.round(2))

        # calculate PID for response iteration
        rt_pid   = PID(rt_kp, rt_ki, rt_kd, np.array(kid_df.rt), robot.setpoint['response_iteration'], np.float(temp_df.rt))
        pid = {'rt': rt_pid}


        temp_pid = np.array([rt_pid, t])
        temp_pid = np.array(temp_pid, 'float')  # converting to float.
        temp_pid = pd.DataFrame(data=temp_pid.reshape(1, 2), columns=cls_names_pid)
        if t == 1:
            pid_df = pd.DataFrame(data=temp_pid, columns=cls_names_pid)  # create the dataframe
        else:
            pid_df = pid_df.append(temp_pid)

        if prt:
            print('Iteration = %d' %(t))
            print('KID: rt = %.2f') % (kid.rt)
            print('PID: rt = %.2f') % (rt_pid)

        if t != 0:
            if (kid_df.right[-2:].sum == 1): # if the kid wasn't right or wrong twice
                vp = 0 # do nothing
                p = 0 # do nothing
                lvl = 1 # do nothing
            else:
                # What is the action the robot took based on the PIDs values.
                vp, p, lvl = robot.pid_action(kid, rt_pid)

        vp1 = np.copy(vp)
        if type(vp) != int:
            vp1 = 3
            p = 3
        temp_robot = np.array([vp1, p, lvl, t])
        temp_robot = pd.DataFrame(data=temp_robot.reshape(1, 4), columns=cls_names_robot)
        if t == 1:
            robot_df = pd.DataFrame(data=temp_robot, columns=cls_names_robot)  # create the dataframe
        else:
            robot_df = robot_df.append(temp_robot)

        # What is the action the kid took based on the robot actions.
        kid.response2robot(robot, vp, lvl, pid, krand)

    # saving the response history.
    kid_df.to_csv(path)
    pid_df.to_csv(path+'_pid')
    robot_df.to_csv(path+'_robot')

    summary_plot(kid_df, pid_df, robot_df, ttl)

def summary_plot(kid_df, pid_df, robot_df, title):
    fig, ax = plt.subplots(1, 3)
    fig.suptitle('kp/ki/kd: '+title, size = 24)
    kid_df.plot(x='iteration', y=['rt', 'lips', 'gaze', 'right'], legend=True, ax=ax[0])
    # ax1 = ax[0].twinx()
    # kid_df.plot(x='iteration', y=['rt', 'lips', 'gaze', 'right'], legend=True, ax=ax1)
    nclrs = plt.rcParams['axes.prop_cycle']
    ax[0].hlines(robot.setpoint['lips'], 0, kid_df.iteration[-1:], linestyles='dashed', color=nclrs._left[1].values())
    ax[0].hlines(robot.setpoint['gaze'], 0, kid_df.iteration[-1:], linestyles='dashed', color=nclrs._left[2].values())
    ax[0].hlines(robot.setpoint['response_iteration'], 0, kid_df.iteration[-1:], linestyles='dashed', color=nclrs._left[0].values())
    ax[0].set_title('KID\'s parameters')

    pid_df.plot(x='iteration', y=['rt_pid', 'lips_pid', 'gaze_pid'], legend=True, ax=ax[1])
    ax[1].set_title('PID\'s parameters')

    robot_df.plot(x = 'iteration', y = ['verbal', 'physical', 'level'], legend = True, ax = ax[2])
    ax[2].set_title('Robot\'s actions')

    fig.set_size_inches(24, 12)
    title = title.replace('.', '_')
    title = title.replace('/', '_')
    title = title.replace(': ', '_')
    title = title.replace(', ', '_')
    title += '.png'
    fig.savefig(title, dpi=300, format = 'png')

def save_maxfig(fig, fig_name, transperent = False, frmt='eps', resize=None):
    '''Save figure in high resultion'''
    # matplotlib.rcParams.update({'font.size': 40})



if __name__ == '__main__':
    path = 'kid_data'
    simulate = 1
    style.use(['ggplot', 'presentation'])
    if simulate == 1:
        kid = kid()
        robot = robot()
        # np.random.seed(0) # Reproducing the same randomness each iteration we run the code.
        simulate_kid_data('kid_data', kid, robot, 100, prt = False)

    # plt.show()

    # todo: unless looking away. wrong & right - one action

