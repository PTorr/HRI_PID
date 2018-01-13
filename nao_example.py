from naoqi import ALProxy

IP =str
PORT=9559
#
# Initialize the API's
motionProxy = ALProxy("ALMotion",IP ,PORT )
tts = ALProxy("ALTextToSpeech", IP, PORT)
managerProxy = ALProxy("ALBehaviorManager", IP, PORT)

# Move hands:
# information about joints: http://doc.aldebaran.com/2-1/family/nao_t14/joints_t14.html
pNames=[]              # joint name (in order to move more then one - can be a list)
pTargetAngles=[]       # angels to be set for the joints in radians (if more the one - list)
pMaxSpeedFraction=int  # play with this to have a nice movment.

motionProxy.post.angleInterpolationWithSpeed(pNames, pTargetAngles, pMaxSpeedFraction)  # command to nao

# Nao Speak:
tts.post.say("This is a sample text!")


# managerProxy.post.runBehavior(<Behavior name>)