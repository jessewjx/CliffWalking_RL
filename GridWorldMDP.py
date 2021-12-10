from typing import List, Callable, Tuple, Any




def startState() -> List:
    """
    Return the start state.
    [x,y,t]
    """
    return [0, 0, 0]



def actions(state: Tuple) -> List[str]:
    """
    Return set of actions possible from |state|.
    """
    return ['Up', 'Down', 'Left','Right']



def IsEnd(state: List) -> bool:
    if state[2] == 249 or state[0:2] == [11, 0]:
        return True
    else:
        return False



def succAndReward(state: Tuple, action: str) -> List:
    """
    Given a |state| and |action|, return a list of [newState, reward]
    corresponding to the states reachable from |state| when taking |action|.
    """
    # End state
    if IsEnd(state):
        return []

    if action == "Up":
        # out of border
        if state[1] == 3:
            nextState = [state[0], state[1], state[2] + 1]
            nextState.append(-1)
            return nextState
        # normal move
        else:
            nextState = [state[0], state[1]+1, state[2] + 1]
            nextState.append(-1)
            return nextState



    if action == "Down":
        # hit the cliff
        if state[1] - 1 == 0 and state[0] > 0 and state[0] < 10:
            nextState = [0, 0, state[2] + 1]
            nextState.append(-100)
            return nextState
        # out of border
        elif state[1] == 0:
            nextState = [state[0], state[1], state[2] + 1]
            nextState.append(-1)
            return nextState
        # normal move
        else:
            nextState = [state[0], state[1] -1, state[2] + 1]
            nextState.append(-1)
            return nextState



    if action == "Left":
        # hit the cliff
        if state[1] == 0 and state[0] - 1 ==10:
            nextState = [0, 0, state[2] + 1]
            nextState.append(-100)
            return nextState
        # out of border
        elif state[0] == 0:
            nextState = [state[0], state[1], state[2] + 1]
            nextState.append(-1)
            return nextState
        # normal move
        else:
            nextState = [state[0]-1, state[1], state[2] + 1]
            nextState.append(-1)
            return nextState


    if action == "Right":
        # hit the cliff
        if state[1] == 0 and state[0] + 1 == 1:
            nextState = [0, 0, state[2] + 1]
            nextState.append(-100)
            return nextState
        # out of border
        elif state[0] == 11:
            nextState = [state[0], state[1], state[2] + 1]
            nextState.append(-1)
            return nextState
        # normal move
        else:
            nextState = [state[0]+1, state[1], state[2] + 1]
            nextState.append(-1)
            return nextState


