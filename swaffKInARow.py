'''
swaffKInARow.py
Author: Samuel Affolder
An agent for playing "K-in-a-Row with Forbidden Squares"
CSE 415, University of Washington
'''

import time
import copy
import math
from winTesterForK import winTesterForK

# Global variables to hold information about the opponent and game version:
INITIAL_STATE = None
OPPONENT_NICKNAME = 'Not yet known'
OPPONENT_PLAYS = 'O' # Update this after the call to prepare.

# Information about this agent:
MY_LONG_NAME = 'IG@guccishortshorts'
MY_NICKNAME = 'blackethCat'
I_PLAY = 'X' # Gets updated by call to prepare.

# GAME VERSION INFO
M = 0 # row count
N = 0 # col count
K = 0 # number in a row needed
TIME_LIMIT = 0
START_TIME = 0
MOVES_MADE = 0

############################################################
# INTRODUCTION
def introduce():
    intro = f'\nMy name is {MY_LONG_NAME}.\n'+\
            'Samuel Affolder swaff made me.\n'
    return intro

def nickname():
    return MY_NICKNAME

############################################################

# Receive and acknowledge information about the game from
# the game master:
def prepare(initial_state, k, what_side_I_play, opponent_nickname):
    # Write code to save the relevant information in either
    # global variables.
    #print("Made it to prepare")
    global INITIAL_STATE, M, N, K, I_PLAY, MOVES_MADE, OPPONENT_NICKNAME, OPPONENT_PLAYS
    INITIAL_STATE = initial_state
    M = len(INITIAL_STATE[0])
    N = len(INITIAL_STATE[0][0])
    K = k
    I_PLAY = what_side_I_play
    if what_side_I_play == "X":
        I_PLAY = what_side_I_play
        OPPONENT_PLAYS = "O"
    else:
        I_PLAY = "O"
        OPPONENT_PLAYS = "X"
        MOVES_MADE = 1
    OPPONENT_NICKNAME = opponent_nickname
    # find out what parts of the board are already played or unplayable
    for i in range(M):
        for j in range(N):
            if INITIAL_STATE[0][i][j] != " ":
                MOVES_MADE += 1
    return "OK"

############################################################

def makeMove(currentState, currentRemark, timeLimit=10000):
    global START_TIME, TIME_LIMIT, MOVES_MADE
    START_TIME = time.time()
    TIME_LIMIT = timeLimit
    searchDepth = 1
    xTurn = currentState[1] == "X"
    optScore = None # set the based on min or max player
    if xTurn: optScore = -math.inf
    else: optScore = math.inf
    bestMove = playableMoves(currentState).pop(0) # default move is any playable one
    # as long as I'm good on time and not searching too deep: LETS SEARCH!
    while time.time() - START_TIME < timeLimit and searchDepth < (M * N) - (MOVES_MADE):
        for possMove in playableMoves(currentState):
            if (time.time() - START_TIME > timeLimit * 0.95): break # another place to check i'm good on time
            newState = moveAndCopy(currentState, possMove)
            if xTurn: # maximize
                minVal = minimax(newState, searchDepth - 1)[0]
                if minVal is None: break # only happens when short on time
                if minVal > optScore:
                    optScore = minVal
                    bestMove = possMove
            else: # minimize
                maxVal = minimax(newState, searchDepth - 1)[0]
                if maxVal is None: break
                if maxVal < optScore:
                    optScore = maxVal
                    bestMove = possMove
        searchDepth += 1
    newRemark = f"YO, take this {OPPONENT_NICKNAME}\n"
    MOVES_MADE += 1
    return [[bestMove, moveAndCopy(currentState, bestMove)], newRemark]


##########################################################################
# The main adversarial search function:
def minimax(state, depthRemaining, pruning=False, alpha=None, beta=None, zHashing=None):
    xTurn = state[1] == "X"
    if (time.time() - START_TIME > TIME_LIMIT * 0.95): # check I'm good on time
        return [None]
    if not depthRemaining: # at target depth, get the value of this state
        return staticEval(state)
    optScore = None
    if xTurn:
        optScore = math.inf
    else:
        optScore = -math.inf
    # for every possible move on this board, check its optimal outcome paths
    # take the optimal route for whos playing aka it alternates
    for possMove in playableMoves(state):
        newState = moveAndCopy(state, possMove)
        if not (winTesterForK(newState, possMove, K) == 'No win'): # someone wins
            if xTurn: return [math.inf]
            else: return [-math.inf]
        if xTurn:
            minVal = minimax(newState, depthRemaining - 1)[0]
            if minVal is None: return [None]
            optScore = min(optScore, minVal)
        else:
            maxVal = minimax(newState, depthRemaining - 1)[0]
            if maxVal is None: return [None]
            optScore = max(optScore, maxVal)
    return [optScore]

##########################################################################

def staticEval(state):
    # Values should be higher when the states are better for X,
    # lower when better for O.
    score = 0
    # playerScore and spectatorScore based on corresponding OfK varaible at each i, j
    # for each K sized window in every direction
    playerScore = 0
    playerOfK = 0 # num. of players elements in a K sized window
    spectatorOfK = 0 # num. of spectators elements in a K sized window
    spectatorScore = 0
    board, player = state
    rows = len(board)
    cols = len(board[0])
    spectator = "O"
    if player == spectator: spectator = "X"
    directions  = [(0,1),(1,1),(1,0),(-1,1),(0,-1),(-1,-1),(-1,0),(1,-1)]
    for i in range(rows):
        for j in range(cols):
            for di in range(len(directions)):
                if board[i][j] == player: playerOfK += 1
                if board[i][j] == spectator: spectatorOfK += 1
                dir = directions[di]
                iTemp = i
                jTemp = j
                for step in range(K - 1): # check a k sized window in current direction
                    # check the coords are on the board
                    iTemp += dir[0]
                    if iTemp < 0 or iTemp >= rows:
                        playerOfK = 0
                        spectatorOfK = 0
                        break
                    jTemp += dir[1]
                    if jTemp < 0 or jTemp >= cols:
                        playerOfK = 0
                        spectatorOfK = 0
                        break
                    if board[iTemp][jTemp] == "-":
                        playerOfK = 0
                        spectatorOfK = 0
                        break
                    if board[iTemp][jTemp] == player: playerOfK += 1
                    if board[iTemp][jTemp] == spectator: spectatorOfK += 1
                # this weighting system only rewards strictly K sized windows with none of other players pieces in it
                if playerOfK > 0 and spectatorOfK == 0:
                    if playerOfK >= K - 1: playerScore += playerOfK**10
                    else: playerScore += 4**playerOfK
                if spectatorOfK > 0 and playerOfK == 0:
                    if spectatorOfK >= K - 1: spectatorScore += spectatorOfK**10
                    else: spectatorScore += 4**spectatorOfK
                playerOfK = 0
                spectatorOfK = 0
    if player == "X":
        score = playerScore - spectatorScore
    else:
        score = spectatorScore - playerScore
    return [score]

##########################################################################

## Helpers

# Check for empty spots to play
# Return a set of those available spots
def playableMoves(state):
    pMoves = []
    for i in range(len(state[0])):
        for j in range(len(state[0][0])):
            if state[0][i][j] == " ":
                pMoves.append([i, j])
    return pMoves

# Create a copy of current state and add the move
# then return the new state
def moveAndCopy(state, move):
    newState = copy.deepcopy(state)
    newState[0][move[0]][move[1]] = newState[1] # adds the move to the board
    if newState[1] == "X": # switch whos turn it is
        newState[1] = "O"
    else:
        newState[1] = "X"
    return newState
