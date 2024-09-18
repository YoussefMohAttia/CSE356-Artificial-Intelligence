import numpy as np
import heapq
from collections import deque

class Node:
    def __init__(self, board, parent=None, move=None, g=0, h=0):
        self.board = board
        self.parent = parent
        self.move = move
        self.g = g  # Cost from start to current node
        self.h = h  # Heuristic cost to goal
        self.f = g + h  # Total cost

    def __eq__(self, other):
        return np.array_equal(self.board, other.board)

    def __lt__(self, other):
        return self.f < other.f

    def __hash__(self):
        return hash(self.board.tostring())

class Puzzle:
    def __init__(self, initial_state, goal_state):
        self.initial_state = np.array(initial_state)
        self.goal_state = np.array(goal_state)
        self.size = self.initial_state.shape[0]
        self.goal_positions = self.calculate_goal_positions()

    def calculate_goal_positions(self):
        positions = {}
        for r in range(self.goal_state.shape[0]):
            for c in range(self.goal_state.shape[1]):
                positions[self.goal_state[r, c]] = (r, c)
        return positions

    def get_neighbors(self, node):
        neighbors = []
        zero_pos = np.argwhere(node.board == 0)[0]
        possible_moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right

        for move in possible_moves:
            new_pos = zero_pos + move
            if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
                new_board = node.board.copy()
                new_board[zero_pos[0], zero_pos[1]], new_board[new_pos[0], new_pos[1]] = new_board[new_pos[0], new_pos[1]], new_board[zero_pos[0], zero_pos[1]]
                neighbors.append(Node(new_board, node, move))

        return neighbors

    def heuristic(self, board):
        total_distance = 0
        for r in range(board.shape[0]):
            for c in range(board.shape[1]):
                if board[r, c] != 0:  # Assuming 0 is the blank tile
                    goal_r, goal_c = self.goal_positions[board[r, c]]
                    total_distance += abs(r - goal_r) + abs(c - goal_c)
        return total_distance

    def heuristic_2(self, board):
        total_distance = 0
        for r in range(board.shape[0]):
            for c in range(board.shape[1]):
                if board[r, c] != 0:  # Assuming 0 is the blank tile
                    goal_r, goal_c = self.goal_positions[board[r, c]]
                    total_distance += np.sqrt((r - goal_r) ** 2 + (c - goal_c) ** 2)
        return total_distance

    def is_solved(self, board):
        return np.array_equal(board, self.goal_state)

    def is_solvable(self):
        print("checking")
        flat_board = self.initial_state.flatten()
        inversions = 0
        for i in range(len(flat_board)):
            for j in range(i + 1, len(flat_board)):
                if flat_board[i] > flat_board[j] != 0:
                    inversions += 1
        if self.size % 2 != 0:
            return inversions % 2 == 0
        else:
            blank_row = np.argwhere(self.initial_state == 0)[0][0]
            if (self.size - blank_row) % 2 == 0:
                return inversions % 2 != 0
            else:
                return inversions % 2 == 0

class Solver:
    def __init__(self, puzzle):
        self.puzzle = puzzle

    def solve_bfs(self):
        initial_node = Node(self.puzzle.initial_state)
        if self.puzzle.is_solved(initial_node.board):
            return self.reconstruct_path(initial_node),0

        frontier = deque([initial_node])
        explored = set()
        explored.add(initial_node)

        while frontier:
            current_node = frontier.popleft()
            if self.puzzle.is_solved(current_node.board):
                return self.reconstruct_path(current_node),len(explored)

            for neighbor in self.puzzle.get_neighbors(current_node):
                if neighbor not in explored:
                    frontier.append(neighbor)
                    explored.add(neighbor)

        return None,0  # No solution found

    def solve_dfs(self):
        initial_node = Node(self.puzzle.initial_state)
        if self.puzzle.is_solved(initial_node.board):
            return self.reconstruct_path(initial_node),0

        frontier = [initial_node]
        explored = set()
        explored.add(initial_node)

        while frontier:
            current_node = frontier.pop()
            if self.puzzle.is_solved(current_node.board):
                return self.reconstruct_path(current_node),len(explored)

            for neighbor in self.puzzle.get_neighbors(current_node):
                if neighbor not in explored:
                    frontier.append(neighbor)
                    explored.add(neighbor)

        return None,0  # No solution found

    def solve_astar(self):
        initial_node = Node(self.puzzle.initial_state, h=self.puzzle.heuristic(self.puzzle.initial_state))
        open_set = []
        heapq.heappush(open_set, initial_node)
        closed_set = set()

        while open_set:
            current_node = heapq.heappop(open_set)

            if self.puzzle.is_solved(current_node.board):
                return self.reconstruct_path(current_node),len(open_set)

            closed_set.add(current_node)

            for neighbor in self.puzzle.get_neighbors(current_node):
                if neighbor in closed_set:
                    continue

                neighbor.g = current_node.g + 1
                neighbor.h = self.puzzle.heuristic(neighbor.board)
                neighbor.f = neighbor.g + neighbor.h

                if neighbor not in open_set:
                    heapq.heappush(open_set, neighbor)
                else:
                    for open_node in open_set:
                        if neighbor == open_node and neighbor.g < open_node.g:
                            open_node.g = neighbor.g
                            open_node.f = neighbor.f
                            open_node.parent = current_node
                            break

        return None,0  # No solution found

    def solve_astar_eucleadian(self):
        initial_node = Node(self.puzzle.initial_state, h=self.puzzle.heuristic(self.puzzle.initial_state))
        open_set = []
        heapq.heappush(open_set, initial_node)
        closed_set = set()

        while open_set:
            current_node = heapq.heappop(open_set)

            if self.puzzle.is_solved(current_node.board):
                return self.reconstruct_path(current_node),len(open_set)

            closed_set.add(current_node)

            for neighbor in self.puzzle.get_neighbors(current_node):
                if neighbor in closed_set:
                    continue

                neighbor.g = current_node.g + 1
                neighbor.h = self.puzzle.heuristic_2(neighbor.board)
                neighbor.f = neighbor.g + neighbor.h

                if neighbor not in open_set:
                    heapq.heappush(open_set, neighbor)
                else:
                    for open_node in open_set:
                        if neighbor == open_node and neighbor.g < open_node.g:
                            open_node.g = neighbor.g
                            open_node.f = neighbor.f
                            open_node.parent = current_node
                            break

        return None,0  # No solution found

    def reconstruct_path(self, node):
        path = []
        while node:
            path.append(node.board)
            node = node.parent

        return path[::-1]

if __name__ == "__main__":
    initial_state = [
        [1, 4, 2],
        [3, 5, 8],
        [0, 6, 7]
    ]

    goal_state = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ]

    puzzle = Puzzle(initial_state, goal_state)
    solver = Solver(puzzle)

    print("BFS Solution:")
    solution,bexp = solver.solve_bfs()
    if solution:
        for step in solution:
            print(step)
        print(bexp)
    else:
        print("No solution found.")

    print("\nDFS Solution:")
    solution,dexp = solver.solve_dfs()
    if solution:
        for step in solution:
            print(step)
        print(dexp)
    else:
        print("No solution found.")

    print("\nA* Solution:")
    solution,aexp = solver.solve_astar()
    if solution:
        for step in solution:
            print(step)
        print(aexp)
    else:
        print("No solution found.")

    print("\nA* eucleadian Solution:")
    solution,aexp2 = solver.solve_astar_eucleadian()
    if solution:
        for step in solution:
            print(step)
        print(aexp2)
    else:
        print("No solution found.")
