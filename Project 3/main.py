import time

import numpy as np
import random

Rows = 3
Cols = 3
Actions_count = 4
Actions = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # Down, Left, Up, Right
Gamma = 0.99
MAX_ERROR = 1e-4  # Convergence threshold


def printGrid(arr, policy=False):
    res = ""
    for r in range(Rows):
        res += "|"
        for c in range(Cols):
            if policy:
                if r == 0 and (c == 0 or c == 2):
                    val = "T.State"
                else:
                    val = ["Down", "Left", "Up", "Right"][arr[r][c]]
            else:
                val = str(arr[r][c])
            res += " " + val[:7].center(7) + " |"
        res += "\n"
    print(res)


def getValue(U, r, c, action):
    rMove, cMove = Actions[action]
    newR, newC = r + rMove, c + cMove
    if newR < 0 or newC < 0 or newR >= Rows or newC >= Cols or (newR == newC == 1):  # invalid move
        return U[r][c]
    else:
        return U[newR][newC]


def calcValue(U, r, c, action):
    v = -1
    v += 0.1 * Gamma * getValue(U, r, c, (action - 1) % 4)
    v += 0.8 * Gamma * getValue(U, r, c, action)
    v += 0.1 * Gamma * getValue(U, r, c, (action + 1) % 4)
    return v


def valueIteration(arr, reward):
    count = 0
    print("During the value iteration:\n")
    while True:
        count += 1
        temp = [[reward, 0, 10], [0, 0, 0], [0, 0, 0]]
        error = 0
        for r in range(Rows):
            for c in range(Cols):
                if r == 0 and (c == 2 or c == 0):
                    continue
                temp[r][c] = max([calcValue(arr, r, c, action) for action in range(Actions_count)])
                error = max(error, abs(temp[r][c] - arr[r][c]))
        arr = temp
        printGrid(arr)
        if error < MAX_ERROR:
            break
    return arr, count


def getOptimalPolicy(arr):
    policy = np.zeros((Rows, Cols), dtype=int)
    for r in range(Rows):
        for c in range(Cols):
            if r == 0 and (c == 2 or c == 0):
                continue
            # Choose the action that maximizes the utility
            maxAction, maxv = None, -float("inf")
            for action in range(Actions_count):
                v = calcValue(arr, r, c, action)
                if v > maxv:
                    maxAction, maxv = action, v
            policy[r][c] = maxAction
    return policy


def policyEvaluation(policy, arr, reward):
    while True:
        temp = [[reward, 0, 10], [0, 0, 0], [0, 0, 0]]
        error = 0
        for r in range(Rows):
            for c in range(Cols):
                if r == 0 and (c == 2 or c == 0):
                    continue
                temp[r][c] = calcValue(arr, r, c, policy[r][c])
                error = max(error, abs(temp[r][c] - arr[r][c]))
        arr = temp
        if error < MAX_ERROR:
            break
    return arr


def policyIteration(policy, arr, reward):
    count = 0
    print("During the policy iteration:\n")
    while True:
        count+=1
        arr = policyEvaluation(policy, arr, reward)
        unchanged = True
        for r in range(Rows):
            for c in range(Cols):
                if r == 0 and (c == 2 or c == 0):
                    continue
                maxAction, maxv = None, -float("inf")
                for action in range(Actions_count):
                    value = calcValue(arr, r, c, action)
                    if value > maxv:
                        maxAction, maxv = action, value
                if maxv > calcValue(arr, r, c, policy[r][c]):
                    policy[r][c] = maxAction
                    unchanged = False
        if unchanged:
            break
        printGrid(policy, True)
    return policy,count


r_values = [100, 3, 0, -3]
for r in r_values:
    Grid = [
        [r, -1, 10],
        [-1, -1, -1],
        [-1, -1, -1]
    ]
    start = time.time()
    print("The initial Values for ", r, " is:\n")
    printGrid(Grid)
    final, count = valueIteration(Grid, r)
    policy = getOptimalPolicy(final)
    print("The time taken: ", time.time() - start, "Number of iterations: ", count, " and optimal policy is:\n")
    printGrid(policy, True)

print("\n\n\n//////////////////////////////////////////////////////////////////////////////\n\n\n")

r_values = [100, 3, 0, -3]
for r in r_values:
    Grid = [
        [r, -1, 10],
        [-1, -1, -1],
        [-1, -1, -1]
    ]
    start = time.time()
    policy = [[random.randint(0, 3) for j in range(Cols)] for i in range(Rows)]
    print("The initial random policy for ", r, " is:\n")
    printGrid(policy, True)
    finall, count = policyIteration(policy, Grid, r)
    print("The time taken: ", time.time() - start, "Number of iterations: ", count, " and optimal policy is:\n")
    printGrid(finall, True)
