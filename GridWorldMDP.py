from typing import List, Callable, Tuple, Any


############################################################
# Problem 5

# Return the start state.
# Look closely at this function to see an example of state representation for our Blackjack game.
# Each state is a tuple with 3 elements:
#   -- The first element of the tuple is the sum of the cards in the player's hand.
#   -- If the player's last action was to peek, the second element is the index
#      (not the face value) of the next card that will be drawn; otherwise, the
#      second element is None.
#   -- The third element is a tuple giving counts for each of the cards remaining
#      in the deck, or None if the deck is empty or the game is over (e.g. when
#      the user quits or goes bust).
def startState() -> Tuple:
    return [0, 0, 0]

# Return set of actions possible from |state|.
# You do not need to modify this function.
# All logic for dealing with end states should be placed into the succAndProbReward function below.
def actions(state: Tuple) -> List[str]:
    return ['Up', 'Down', 'Left','Right']

def IsEnd(state: List) -> bool:
    print(f"Is {state} end?")
    if state[2] == 249 or state[0:2] == [11, 0]:
        print("Yes")
        return True
    else:
        print("No")
        return False
# Given a |state| and |action|, return a list of (newState, prob, reward) tuples
# corresponding to the states reachable from |state| when taking |action|.
# A few reminders:
# * Indicate a terminal state (after quitting, busting, or running out of cards)
#   by setting the deck to None.
# * If |state| is an end state, you should return an empty list [].
# * When the probability is 0 for a transition to a particular new state,
#   don't include that state in the list returned by succAndProbReward.
# Note: The grader expects the outputs follow the same order as the cards.
# For example, if the deck has face values: 1, 2, 3. You should order your corresponding
# tuples in the same order.
def succAndReward(state: Tuple, action: str) -> List[Tuple]:
    # BEGIN_YOUR_CODE (our solution is 38 lines of code, but don't worry if you deviate from this)
    # End state
    if IsEnd(state):
        return []
    # if state[2] == 249 or tuple(state[0:2]) == (11,0):
    #     return []

    if action == "Up":
        # out of border
        if state[1] == 3:
            print("=>Out of border")
            nextState = [state[0], state[1], state[2] + 1]
            nextState.append(-1)
            return nextState


        # normal move
        else:
            print("=>Go up")
            nextState = [state[0], state[1]+1, state[2] + 1]
            nextState.append(-1)

            return nextState
            # succState = (state[0] + 1, state[1], state[2] + 1, "Up", -1)


    if action == "Down":
        # hit the cliff
        if state[1] - 1 == 0 and state[0] > 0 and state[0] < 10:
            print("=>Hit the cliff")
            nextState = [0, 0, state[2] + 1]
            nextState.append(-100)
            return nextState
            # succState = (0, 0, state[2] + 1, "Down", -100)

        # out of border
        elif state[1] == 0:
            print("=>out of border")
            nextState = [state[0], state[1], state[2] + 1]
            nextState.append(-1)
            return nextState
            # succState = (state[0], state[1], state[2] + 1, "Down", -1)


        # normal move
        else:
            print("=>Go Down")
            nextState = [state[0], state[1] -1, state[2] + 1]
            nextState.append(-1)
            return nextState
            # succState = (state[0] - 1, state[1], state[2] + 1, "Down", -1)




    if action == "Left":
        # hit the cliff
        if state[1] == 0 and state[0] - 1 ==10:
            print("=>Hit the cliff")
            nextState = [0, 0, state[2] + 1]
            nextState.append(-100)
            return nextState
            # succState = (0, 0, state[2] + 1, "Left", -100)

        # out of border
        elif state[0] == 0:
            print("=>Out of border")
            nextState = [state[0], state[1], state[2] + 1]
            nextState.append(-1)
            return nextState
            # succState = (state[0], state[1], state[2] + 1, "Left", -1)


        # normal move
        else:
            print("=>Go left")
            nextState = [state[0]-1, state[1], state[2] + 1]
            nextState.append(-1)
            return nextState
            # succState = (state[0], state[1] - 1, state[2] + 1, "Left", -1)





    if action == "Right":
        # hit the cliff
        if state[1] == 0 and state[0] + 1 == 1:
            # print("=>Hit the cliff")
            nextState = [0, 0, state[2] + 1]
            nextState.append(-100)
            return nextState


        # out of border
        elif state[0] == 11:
            # print("=>Out of border")
            nextState = [state[0], state[1], state[2] + 1]
            nextState.append(-1)
            return nextState
            # succState = (state[0], state[1], state[2] + 1, "Right", -1)


        # normal move
        else:
            # print("=> go right")
            nextState = [state[0]+1, state[1], state[2] + 1]
            nextState.append(-1)
            return nextState
            # succState = (state[0], state[1] + 1, state[2] + 1, "Right", -1)

        # return succState



    # END_YOUR_CODE

def discount():
    return 1


