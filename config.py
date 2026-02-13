#########################################################################
# YOU CAN MODIFY THESE FOUR CONFIG PARAMS                               #
# MODE, SEED, WINDOW_SIZE, and FRAME_RATE                               #
# BUT DO NOT ADD ANY OTHER PARAMS TO THIS FILE                          #
# IF YOU WANT TO STORE SOME OTHER CONFIG PARAMS, STORE THEM IN ROBOT.PY #
#########################################################################

import time

# Set the mode.
# 'development' is what you should use when you are developing your algorithm.
# In development mode, you have access to the true dynamics, and the true observation function.
# This enables you to visualise the plans and the dynamics model.
# 'evaluation' is what will be used when we evaluate your algorithm.
# Therefore, before you submit your code on Scientia, you should make sure that your entire algorithm (training and testing) runs well in evaluation mode.

MODE = 'development'
#MODE = 'evaluation'

# Set the random seed.
# During our evaluation of your algorithm, this will be a random number.
# If you use a specific number for the seed, the environment will always be the same.
# Note that for the specific case of SEED = 0, the environment will actually be different each time it is created.
SEED = 42 #int(time.time())

# Set the size of the window.
WINDOW_SIZE = 400

# Set the frame rate for pygame, which determines how quickly the program runs.
# In our evaluation of your code, this will be set at 30.
FRAME_RATE = 100
