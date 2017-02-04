"""
Once a model is learned, use this to play it.
"""

#from flat_game import 
import numpy as np
from nn import neural_net
from flat_game import test_carmunk
import time

NUM_SENSORS = 3


def play(model):

    car_distance = 0
    start_time=time.time()
    game_state = test_carmunk.GameState()

    # Do nothing to get initial.
    _, state = game_state.frame_step((2))
   
    print("---%s seconds---" % (time.time() - start_time))
    # Move
    test=True
    state1=[]
    action1=[]
    action2=[]
    while test:
        car_distance += 1

        # Choose action.
        action = (np.argmax(model.predict(state, batch_size=1)))
        actiont = model.predict(state, batch_size=1)

        # Take action.
        _, state = game_state.frame_step(action)
        

        # Tell us something.
        if car_distance % 1000 == 0:
            print("Current distance: %d frames." % car_distance)
            print(state)
            state1=np.append(state1,state)
            action1=np.append(action1,action)
            action2=np.append(action2,actiont)
#            state1=np.concatenate((state1,state))
#            action1=np.concatenate((action1,action))
#            action2=np.concatenate((action2,actiont))
        if car_distance % 10000==0:
            test=False
    return state1,action1,action2


if __name__ == "__main__":
    saved_model = 'save_model/50-50-100-500-10000.h5'
    model = neural_net(NUM_SENSORS, [50, 50], saved_model)
    [state,action1,action2]=play(model)
    state=state.reshape(10,3)
    action1=action1.reshape(10,1)
    action2=action2.reshape(10,3)
    np.savetxt('tests.out',state)
    np.savetxt('testa1.out',action1)
    np.savetxt('testa2.out',action2)

