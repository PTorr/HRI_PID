import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

class kid():
    '''This is an object for kid simulation.'''

    def __init__(self):
        self.rt = None # Response time.
        self.push = 1 # The kid pushed or not.
        self.right = None # Did the kid pushed on the right object?
        self.gaze = None # The kid's gaze direction.
        self.lips = None # Lips angle

    def simulate(self):
        '''
        Random simulation of the kid's response.
        :return: random values for all kid's parameters.
        '''
        self.rt = np.random.uniform(0.5, 5)
        if self.rt > 4.5:
            self.push = 0
            self.right = 0
        # KID's actions distribution.
        self.gaze = np.random.normal(loc = 0, scale = 20)
        self.lips = np.random.normal(loc = 15, scale = 5) - 0.5*np.random.normal(loc = 0, scale = 10)
        self.right = np.random.randint(2)

    def response2robot(self, robot, vp, lvl, pid):
        '''
        Kid's response to robot's action.
        :param robot: robot class.
        :param vp: which tribute.
        :param lvl: which level.
        :return: kid's state after responding to the robot's actions.
        '''
        r = self.rand_true(.9)
        # r = self.rand_true(1)
        if r:
            level = robot.actions['level'][lvl]
            if type(vp) == int:
                tribute = robot.actions['physical'][vp][0]
            else:
                tribute = robot.actions['physical'][vp[0]][0] + robot.actions['physical'][vp[1]][0]

            if tribute == 'dance':
                pass
            else:
                self.lips  = self.up_lips(robot, level, tribute, pid)
                self.gaze  = self.up_gaze(robot, level, tribute, pid)
                self.rt  = self.up_rt(robot, level, tribute, pid)
        else:  # kid respond randomly, 1 in 10.
            self.simulate()

    def up_lips(self, robot, lvl, trib, pid):
        '''
        update lips based on robot's parameters.
        :param robot: robot object.
        :param lvl: robot's level action.
        :param trib: robot's tribute action.
        :return: lips response robot's action.
        '''
        action_0 = self.lips
        # response to robot's level action
        action_1 = self.up_action(pid, action_0, [-50,50], lvl, ['d','s','i'], [.2,.5,.8], [.5,.7,.5], robot.setpoint['lips'], kid_action = 'lips')
        # response to robot's tribute
        action_2 = self.up_action(pid, action_0, [-50,50], trib, ['p','e','pe'], [.8,.8,.9], [.7,.7,.8], robot.setpoint['lips'], kid_action = 'lips')
        return np.mean((action_1, action_2))

    def up_gaze(self, robot, lvl, trib, pid):
        '''
        update gaze based on robot's parameters.
        :param robot: robot object.
        :param lvl: robot's level action.
        :param trib: robot's tribute action.
        :return: gaze response robot's action.
        '''
        action_0 = self.gaze
        # response to robot's level action
        action_1 = self.up_action(pid, action_0, [-30,30], lvl, ['d','s','i'], [.15,.1,.05], [.95,.8,.95], robot.setpoint['gaze'], kid_action = 'gaze')
        # response to robot's tribute
        action_2 = self.up_action(pid, action_0, [-30,30], trib, ['p','e','pe'], [.8,.8,.9], [.7,.7,.85], robot.setpoint['gaze'], kid_action = 'gaze')
        return np.mean((action_1, action_2))

    def up_rt(self, robot, lvl, trib, pid):
        '''
        update response time based on robot's parameters.
        :param robot: robot object.
        :param lvl: robot's level action.
        :param trib: robot's tribute action.
        :return: response time response robot's action.
        '''
        action_0 = self.rt
        # response to robot's level action
        action_1 = self.up_action(pid, action_0, [.5, 5], lvl, ['d', 's', 'i'], [.6, .1, .7], [.6, .5, .4], robot.setpoint['response_time'], kid_action = 'rt')
        # response to robot's tribute
        action_2 = self.up_action(pid, action_0, [.5, 5], trib, ['p', 'e', 'pe'], [.2, .2, .3], [.5, .5, .5], robot.setpoint['response_time'], kid_action = 'rt')
        return np.mean((action_1, action_2))

    def up_action(self, pid, action, action_th, rbt_act, act_opt, rbt_act_resp, rbt_act_up, sp, kid_action):
        '''
        Function that update a selected action.
        :param action: which of the kid's action to update.
        :param action_th: action's range.
        :param rbt_act: what was the robot current action.
        :param act_opt: what was the robot action's options.
        :param rbt_act_resp: probability to respond to current robot action.
        :param rbt_act_up: probability to increase the kid's action.
        :return: updated value of the kid's action.
        '''
        if rbt_act[0] == act_opt[0]:
            action = self.resp_action(action, rbt_act_resp[0], rbt_act_up[0], pid[kid_action])
        elif rbt_act[0] == act_opt[1]:
            action = self.resp_action(action, rbt_act_resp[1], rbt_act_up[1], pid[kid_action])
        elif rbt_act[0] == act_opt[2]:
            action = self.resp_action(action, rbt_act_resp[2], rbt_act_up[2], pid[kid_action])

        # Edge values
        if action > action_th[1]:
            action = action_th[1]
        elif action < action_th[0]:
            action = action_th[0]
        
        return action


    def resp_action(self, action, respond, up, pid = None):
        '''
        Kid's response.
        :param action: kid's action value.
        :param respond: respond probability.
        :param up: increase probability.
        :return: updated action.
        '''
        if self.rand_true(respond):
            if self.rand_true(up):
                # action = val + np.random.uniform(-.05, .05) * val
                action += pid
                # action -= 0.1 * pid
            else:
                # action = val - np.random.uniform(.3, .6) * val
                action -= pid
            return action
        else:
            return action

    def rand_true(self, th):
        '''
        Probability generator.
        :param th: what is the probability for True.
        :return: True in probability th.
        '''
        r = np.random.uniform(0.01,0.99)
        if r < th:
            return True
        else:
            return False
class robot():
    '''This is an object for robot simulation.'''

    def __init__(self):
        actions = {'verbal': ['encourage', 'positive', 'brake'], 'physical': ['encourage', 'positive', 'dance'],
                   'level': ['decrease', 'stay', 'increase']}
        # self.actions       = pd.DataFrame.from_dict(actions)
        self.actions  = actions
        self.setpoint = {'response_time': 1, 'gaze': 0, 'lips': 15}


    def pid_action(self, kid, rt_pid, gaze_pid, lips_pid):
        '''
        Make action based on PID values.
        :param kid: kid's object.
        :param rt_pid: response time PID.
        :param gaze_pid: gaze PID.
        :param lips_pid: lips PID.
        :return: vp - which tribute to take.
                 lvl - which level to take.
        '''
        gaze_th = 0
        lips_th = 0
        rt_th = 0
        if gaze_pid <= gaze_th: # looking away
            vp = 2 # break + dance
            lvl = 1 # stay at the same level
        elif gaze_pid > gaze_th: # looking at
            if kid.right == 1:
                if lips_pid <= lips_th: # not happy
                    if rt_pid <= rt_th: # fast
                        vp  = [0,1]
                        lvl = 2
                    elif rt_pid > rt_th: # slow
                        vp  = [0,1]
                        lvl = 1
                elif lips_pid > lips_th: # happy
                    if rt_pid <= rt_th: # fast
                        vp  = 1
                        lvl = 2
                    elif rt_pid > rt_th: # slow
                        vp  = 1
                        lvl = 1
            elif kid.right == 0:
                vp = 0
                lvl = 0
        # self.prnt_selected_action(vp, vp, lvl)
        return vp, vp, lvl


    def prnt_selected_action(self,i,j,k):
        '''
        Print the action the robot did.
        :param i: which physical tribute.
        :param j: which verbal tribute.
        :param k: which level.
        :return: None
        '''
        if type(i) == int:
            print 'robot action = ', self.actions['verbal'][i], 'level = ', self.actions['level'][k]
        elif type(i) == list:
            print 'robot action= ', self.actions['verbal'][i[0]], self.actions['verbal'][i[1]], 'level = ', self.actions['level'][k]
        print

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
    # PID parameters for all the PID process that we calculate.
    rt_kp, rt_ki, rt_kd       = 2, 0, 0
    gaze_kp, gaze_ki, gaze_kd = 1, 0, 0
    lips_kp, lips_ki, lips_kd = 2, 0, 0

    kid.simulate() # intial state
    # Creating the dataframe to save the history of the kid's action.
    temp_val = kid.__dict__.values() # taking the values.
    temp_val = np.append(temp_val, 0) # adding 1st step.
    temp_val = np.array(temp_val, 'float') # converting to float.
    cls_names_kid = kid.__dict__.keys() # taking the names of the fields in kid.
    cls_names_kid.append('time') # addin 't' to the time column as a name.
    kid_df = pd.DataFrame(data=temp_val.reshape(1, 6), columns=cls_names_kid) # create the dataframe
    cls_names_pid = ['rt_pid', 'gaze_pid', 'lips_pid', 'time']
    cls_names_robot = ['verbal', 'pysical', 'level', 'time']
    # Simulating the interaction.
    for t in range(1, N):  # N time steps simulation
        temp_val = kid.__dict__.values()
        temp_val.append(t)
        temp_val = np.array(temp_val, 'float')
        temp_df = pd.DataFrame(data=temp_val.reshape(1, 6), columns=cls_names_kid)
        kid_df = kid_df.append(temp_df.round(2))

        # calculate PID for response time
        rt_pid   = PID(rt_kp, rt_ki, rt_kd, np.array(kid_df.rt), robot.setpoint['response_time'], np.float(temp_df.rt))
        # calculate PID for gaze
        gaze_pid = PID(gaze_kp, gaze_ki, gaze_kd, np.array(kid_df.gaze), robot.setpoint['gaze'], np.float(temp_df.rt))
        # calculate PID for lips
        lips_pid = PID(lips_kp, lips_ki, lips_kd, np.array(kid_df.lips), robot.setpoint['lips'], np.float(temp_df.rt))
        pid = {'rt': rt_pid, 'gaze': gaze_pid, 'lips': lips_pid}


        temp_pid = np.array([rt_pid, gaze_pid, lips_pid, t])
        temp_pid = np.array(temp_pid, 'float')  # converting to float.
        temp_pid = pd.DataFrame(data=temp_pid.reshape(1, 4), columns=cls_names_pid)
        if t == 1:
            pid_df = pd.DataFrame(data=temp_pid, columns=cls_names_pid)  # create the dataframe
        else:
            pid_df = pid_df.append(temp_pid)

        if prt:
            print 'Iteration = %d' %(t)
            print 'KID: rt = %.2f, gaze = %.2f, lips = %.2f' % (kid.rt, kid.gaze, kid.lips)
            print 'PID: rt = %.2f, gaze = %.2f, lips = %.2f' % (rt_pid, gaze_pid, lips_pid)

        # What is the action the robot took based on the PIDs values.
        vp, p, lvl = robot.pid_action(kid, rt_pid, gaze_pid, lips_pid)
        temp_robot = np.array([vp, p, lvl, t])
        temp_robot = np.array(temp_robot, 'float')  # converting to float.
        temp_robot = pd.DataFrame(data=temp_robot.reshape(1, 4), columns=cls_names_pid)
        if t == 1:
            robot_df = pd.DataFrame(data=temp_robot, columns=cls_names_robot)  # create the dataframe
        else:
            robot_df = pid_df.append(temp_robot)
        # What is the action the kid took based on the robot actions.
        kid.response2robot(robot, vp, lvl, pid)
    # saving the response history.
    kid_df.to_csv(path)
    pid_df.to_csv(path+'_pid')
    robot_df.to_csv(path+'_robot')

    ttl = 'rt_'+str(rt_kp)+'_'+str(rt_ki)+'_'+str(rt_kd)+'__gaze_'+str(gaze_kp)+'_'+str(gaze_ki)+'_'+str(gaze_kd)\
          +'__lips_' + str(lips_kp) + '_' + str(lips_ki) + '_' + str(lips_kd)
    summary_plot(kid_df, pid_df, ttl)

def summary_plot(kid_df, pid_df, title):
    fig, ax = plt.subplots(1, 2)
    fig.suptitle('ks: '+title, size = 24)
    kid_df.plot(x='time', y=['lips', 'gaze'], legend=True, ax=ax[0])
    ax1 = ax[0].twinx()
    kid_df.plot(x='time', y=['rt', 'lips', 'gaze'], legend=True, ax=ax1)
    nclrs = plt.rcParams['axes.prop_cycle']
    ax[0].hlines(robot.setpoint['lips'], 0, kid_df.time[-1:], linestyles='dashed', color=nclrs._left[1].values())
    ax[0].hlines(robot.setpoint['gaze'], 0, kid_df.time[-1:], linestyles='dashed', color=nclrs._left[2].values())
    ax[0].hlines(robot.setpoint['response_time'], 0, kid_df.time[-1:], linestyles='dashed', color=nclrs._left[0].values())
    ax[0].set_title('KID\'s parameters')

    pid_df.plot(x='time', y=['rt_pid', 'lips_pid', 'gaze_pid'], legend=True, ax=ax[1])
    ax[1].set_title('PID\'s parameters')

    fig.set_size_inches(12, 18)
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
        # np.random.seed(0) # Reproducing the same randomness each time we run the code.
        simulate_kid_data('kid_data', kid, robot, 100, prt = False)

    plt.show()

    # todo: unless looking away. wrong & right - one action

