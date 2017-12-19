import numpy as np
import pandas as pd

class kid():
    '''This is an object for kid simulation.'''

    def __init__(self):
        self.rt = None # Response time.
        self.push = 1 # The kid pushed or not.
        self.right = None # Did the kid pushed on the right object?
        self.gaze = None # The kid's gaze direction.
        self.lips = None # Lips angle

    def simulate(self):
        self.rt = np.random.uniform(0.5, 5)
        if self.rt > 4.5:
            self.push = 0
            self.right = 0
        self.gaze = np.random.randint(-180, 180)
        self.lips = np.random.randint(-30, 30)
        self.right = np.random.randint(2)

    def analyze_robot_state(self, robot):
        #  todo think how the kid sees the robot.
        pass

class robot():
    '''This is an object for robot simulation.'''

    def __init__(self):
        self.actions  = robot_actions()
        self.response_time = setpoint(1)
        self.gaze         = setpoint(0)
        self.lips         = setpoint(15)

    def analyze_kid_state(self, kp, ki, kd, pv, rt):
        # todo: pv
        # pv_rt, pv_gaze, pv_lips
        rt_pid   = PID(kp, ki, kd, pv, self.response_time.setpoint, rt)
        gaze_pid = PID(kp, ki, kd, pv, self.gaze.setpoint, rt)
        lips_pid = PID(kp, ki, kd, pv, self.lips.setpoint, rt)

        return rt_pid * gaze_pid * lips_pid

class robot_actions():
    '''This is an object for robot actions.'''

    def __init__(self):
        self.verbal = ['encourage', 'positive', 'brake']  # verbal gestures
        self.physical = ['encourage', 'positive', 'dace']  # physical gestures
        self.level = [-1, 0, 1]

class setpoint():
    '''This is an object for setpoints'''
    def __init__(self, sp):
        self.setpoint = sp

def PID(kp, ki, kd, pv, sp, rt):
    '''PID calculations'''
    # https: // codereview.stackexchange.com / questions / 155205 / python - pid - simulator - controller - output

    et = -(pv - sp)

    P = et[-1] # PV(t) = et[-1] # propotional

    I = np.sum(et) # integral

    D = (et[-1] - et[-2])/rt # derivative

    pid = kp*P + ki*I + kd*D

    return pid


def simulate_kid_data(path, kid1):
    temp_val = kid1.__dict__.values()
    temp_val = np.append(temp_val, 0)
    temp_val = np.array(temp_val, 'float')
    cls_names = kid1.__dict__.keys()
    cls_names.append('t')
    kid1_df = pd.DataFrame(data=temp_val.reshape(1, 6), columns=cls_names)
    for t in range(1, 100):  # 100 time steps simulation
        kid1.simulate()
        temp_val = kid1.__dict__.values()
        temp_val.append(t)
        temp_val = np.array(temp_val, 'float')
        np.round(temp_val)
        temp_df = pd.DataFrame(data=temp_val.reshape(1, 6), columns=cls_names)
        kid1_df = kid1_df.append(temp_df)
    # response_history =
    kid1_df.to_csv(path)
    return kid1_df


if __name__ == '__main__':
    path = 'kid_data'
    kid = kid()
    robot = robot()
    np.random.seed(0) # Reproducing the same randomness each time we run the code.

    kp, ki, kd = 0.5, 0.5, 0.5
    kid.simulate()
    a = robot.analyze_kid_state(kp, ki, kd, np.array([0.1,0.2,0.3,0.5,0.6]), 0.2)
    print a

    # todo: if right -> PID/   If wrong - only one action
    # The PID number differentiate between the 4 cases of being right,
    # todo: unless looking away. wrong & right - one action

