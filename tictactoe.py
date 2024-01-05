#----------------------------------------------------------------------------
# tictactoe.py
# Author: Keith Gladstone
# Purpose: Trying to eventually learn reinforcement learning,
#          by building the mechanics of a simple tic-tac-toe 
#          game. There are three abstractions that work together:
#            1 - Board: a representation of the environment, with 
#                       built-in play validation rules, ability 
#                       to provide the next available states, and 
#                       assess if a given environment position is game 
#                       over. Simply states truths. Does not make 
#                       any decisions. The key here is that the underlying
#                       data structure is an array, but instead
#                       the clients manipulate the environment in a
#                       more intuitive way (e.g., crawling through,
#                       setting values by "cell index", understanding
#                       a given state of the board in relation to others).
#            2 - Agent: a client of a given environment. Has the ability 
#                       to make basic selection decisions of the next move 
#                       by accessing truths about the state of the
#                       environment and assigning value to different
#                       choices.
#            3 - Controller: a global controller that runs a 
#                       game by resetting the environment, instantiating
#                       two agents, and running a step-based 
#                       simulation.
#----------------------------------------------------------------------------
import random
from itertools import product
import pandas as pd

class Board:
    #------------- Class Variables -------------------------------
    COLMAX = 3
    ROWMAX = 3
    
    BLANK_SQUARE = 1
    X_SQUARE = 2
    O_SQUARE = 3
    
    #------------- Static Methods -------------------------------
    
    @staticmethod
    def prepare_board_input_from_list(input_list):
        return [input_list[i:i+3] for i in range(0, len(input_list), 3)]
    
    @staticmethod
    def decode_board_string(board_string):
        char_list = list(board_string)
        int_list = [int(i) for i in char_list]
        board_representation = Board.prepare_board_input_from_list(int_list)
        return board_representation      
    
    @staticmethod
    def _encode_board_object_list(board_as_list):
        
        # Validate there are no negatives in the diff
        for element in board_as_list:
            if element not in [
                Board.BLANK_SQUARE, 
                Board.X_SQUARE, 
                Board.O_SQUARE
            ]:
                print("Error: Trying to encode invalid element {}".format(element))
                return ""
        
        board_string_list = [str(i) for i in board_as_list]
        board_string = ''.join(board_string_list)
        return str(board_string)
    
    #-------------  Initialization  -------------------------------
    
    def __init__(self, encoded_string=""):
        self.reset()
        if encoded_string != "":
            self.board = Board.decode_board_string(str(encoded_string))
            assert self._is_board_valid()
        self.next_value = self.which_value_is_next()
    
    # Reset environment to initial state
    def reset(self):
        self.board = [[self.BLANK_SQUARE, self.BLANK_SQUARE, self.BLANK_SQUARE],  # First row
             [self.BLANK_SQUARE, self.BLANK_SQUARE, self.BLANK_SQUARE],  # Second row
             [self.BLANK_SQUARE, self.BLANK_SQUARE, self.BLANK_SQUARE]]  # Third row
        return self
    
    # Receive action and return next state and key information
    def step(self, action):
        row, col, value = action
        next_state = self.safe_set_value(row, col, value)

        terminated = next_state.is_terminated()
        winner = next_state.winner()
        
        if winner == value:
            reward = 1000
        else:
            reward = 0    
        return next_state, reward, terminated
    
    #------------- Cell Operations  -------------------------------     
    
    def is_cell_empty(self, row, col):
        return int(self.board[row][col]) == self.BLANK_SQUARE
    
    def _is_board_valid(self):
        # Is board correct size
        
        # Check row count
        if len(self.board) != self.ROWMAX:
            print("Error: Wrong row count")
            return False
        
        # Check col count
        for row in self.board:
            if len(row) != self.COLMAX:
                print("Error: Wrong col count")
                return False
        
        # Are all cell values valid
        for i in range(0, self.ROWMAX):
            for j in range(0, self.COLMAX):
                if not self._is_cell_valid(i, j):
                    return False
        return True
    
    def _is_cell_valid(self, row, col):
        if not (row < self.ROWMAX and row >= 0):
            print("Error: Row {} out of range".format(row))
            return False
        if not (col < self.COLMAX and col >= 0):
            print("Error: Col {} out of range".format(col))
            return False
        return True
    
    def _is_set_value_valid(self, row, col, value):
        if not self._is_cell_valid(row, col):
            print("Error: Cell ({},{}) is not valid".format(row, col))
            return False
        if not(value in [self.X_SQUARE, self.O_SQUARE]):
            print("Error: Value {} illegal".format(value))
            return False
        if not (self.is_cell_empty(row, col)):
            print("Error: Cell ({},{}) is not empty".format(row, col))
            return False
        return True
    
    def _is_direction_valid(self, direction):
        return direction in ["N", "S", "E", "W", "NE", "SE", "SW", "NW"]
        
    def _set_value(self, row, col, value):
        self.board[row][col] = value
        return True
    
    def _get_cell_index_adjacent(self, origin_row, origin_col, direction):
        assert self._is_direction_valid(direction), "Direction {} illegal".format(direction)
        
        if direction == "N":        
            new_row = origin_row - 1
            new_col = origin_col
        elif direction == "S":        
            new_row = origin_row + 1
            new_col = origin_col
        elif direction =="E":        
            new_row = origin_row
            new_col = origin_col + 1
        elif direction =="W":        
            new_row = origin_row - 1
            new_col = origin_col
        elif direction =="NE":        
            new_row = origin_row - 1
            new_col = origin_col + 1
        elif direction =="SE":        
            new_row = origin_row + 1
            new_col = origin_col + 1
        elif direction =="SW":        
            new_row = origin_row + 1
            new_col = origin_col - 1
        else: # Assume NW
            new_row = origin_row - 1
            new_col = origin_col - 1    
        
        if not self._is_cell_valid(new_row, new_col):
            print("Error: Trying to reach invalid cell")
            return None
        return (new_row, new_col)      
    
    def _count_nonempty_cells(self):
        ct = 0
        for i in range(0, self.ROWMAX):
            for j in range(0, self.COLMAX):
                if not self.is_cell_empty(i, j):
                    ct += 1
        return ct
    
    def get_value(self, row, col):
        assert (self._is_cell_valid(row, col))
        return self.board[row][col]
    
    def safe_set_value(self, row, col, value):
        if not self._is_set_value_valid(row, col, value):
            raise ValueError("Error: Not valid to set value {} to this cell ({},{})".format(value, row, col))
        else:
            self._set_value(row, col, value)
            return self

    def _to_list(self):
        board_as_list = list()
        for i in range(0, self.ROWMAX):
            for j in range(0, self.COLMAX):
                board_as_list.append(self.get_value(i, j))
        return board_as_list    
    
    #------------- Rendering   -------------------------------

        
    def encode(self):
        return self._encode_board_object_list(self._to_list())
          
    def value_to_string(self, value):
        if int(value) == self.BLANK_SQUARE:
            return " "
        elif int(value) == self.X_SQUARE:
            return "X"
        elif int(value) == self.O_SQUARE:
            return "O"
        else:
            return value
    
    def render(self):
        result = ""
        for i in range(0, self.ROWMAX):
            row = ""
            for j in range(0, self.COLMAX):
                value = str(self.get_value(i, j))
                render_value = self.value_to_string(value)
                row += render_value
                if j < self.ROWMAX - 1:
                    row += "|"
            result += row + "\n"
            if i < self.ROWMAX - 1:
                rowborder = ""
                for j in range(0, 2*self.COLMAX - 1):
                    rowborder += "_"
                result += rowborder + "\n"
        print(result)
        return result

    #------------- State Transitions  -------------------------------
  
    def which_value_is_next(self):
        count_X = 0
        count_O = 0
        for i in range(0, self.ROWMAX):
            for j in range(0, self.COLMAX):
                value = self.get_value(i, j)
                if int(value) == self.X_SQUARE:
                    count_X += 1
                elif int(value) == self.O_SQUARE:
                    count_O += 1
        if count_X > count_O:
            return self.O_SQUARE
        else:
            return self.X_SQUARE

    # Generate the list of possible boards if the value is successfully placed
    # Warning: does not valid that the turn is correct for that value
    def next_available_states(self):        
        
        # Next up value
        next_value = self.which_value_is_next()
        
        # Maximum nine next states are possible
        self_string = self.encode()
        new_board_list = list()
        for i in range(0, self.ROWMAX):
            for j in range(0, self.COLMAX):
                new_board = Board(self_string)
                if new_board.is_cell_empty(i, j):
                    new_board.safe_set_value(i, j, next_value)
                    new_board_list.append(new_board)
        return new_board_list
               
    def next_available_moves(self):
        next_states = self.next_available_states()
        next_moves = [self.extract_move_transition(i) for i in next_states]
        return next_moves
    
    def get_first_observed_move(self):
        for i in range(0, self.ROWMAX):
            for j in range(0, self.COLMAX):
                if not self.is_cell_empty(i, j):
                    return (i, j, self.get_value(i, j))

    def equals(self, other_board):
        return self.encode() == other_board.encode()
                
    # Return as a board, but if diff is invalid then return None
    def diff(self, other):
        self_encoded = [int(i) for i in list(str(self.encode()))]
        other_encoded = [int(i) for i in list(str(other.encode()))]
        result = [a - b + 1 for a, b in zip(other_encoded, self_encoded)] 
        diff_board_input = Board._encode_board_object_list(result)
        if diff_board_input == "":
            return None
        return Board(diff_board_input)
    
    # Ensure the two boards differ by only one move
    def is_valid_transition(self, other):
        diff = self.diff(other)
        if diff is None:
            return False
        else:
            return diff._count_nonempty_cells() == 1

    def extract_move_transition(self, other):
        if not self.is_valid_transition(other):
            print("Invalid Transition")
            return (-1, -1, -1)
        diff = self.diff(other)
        move_made = diff.get_first_observed_move()
        return move_made

    #------------- End State Evaluation -------------------------------
    
    def _is_three_match_in_line(self, origin_row, origin_col, direction):
        first_value = self.get_value(origin_row, origin_col)
        
        # Check cell has nonzero value
        if str(first_value) == '1':
            return False, -1
        
        # Check two in a line
        next_row, next_col = self._get_cell_index_adjacent(origin_row, origin_col, direction)
        second_value = self.get_value(next_row, next_col)
        if first_value != second_value:
            return False, -1
        else:
            
            # Check three in a line
            next_row, next_col = self._get_cell_index_adjacent(next_row, next_col, direction)
            third_value = self.get_value(next_row, next_col)
            if second_value != third_value:
                return False, -1
            else:
                return True, first_value
    
    def _is_full(self):
        return self._count_nonempty_cells() == self.ROWMAX*self.COLMAX
        
    def winner(self):     
        three_match_list = list()
        # Generate all possible lines (each row, each column, both diagonals)
        line_list = [(0, 0, "E"),
                     (1, 0, "E"),
                     (2, 0, "E"),
                     (0, 0, "S"),
                     (0, 1, "S"),
                     (0, 2, "S"),
                     (0, 0, "SE"),
                     (2, 0, "NE")
        ]
        
        for row, col, direction in line_list:
            three_match_in_line, value = self._is_three_match_in_line(row, col, direction)
            if three_match_in_line:
                three_match_list.append((row, col, direction, value))
        
        if len(three_match_list) > 0:
            row, col, direction, value = three_match_list[0]
            return value
        else:
            return None
        
    def is_terminated(self):
        full = self._is_full()
        winner = self.winner()

        return full or (winner is not None)
        
class Agent:
    
    def __init__(self, name, is_human):
        self.name = name
        self.is_human = is_human
        
    def make_value_judgment(self, next_state, my_value): 
        # The value of a given state is the expected reward from the paths leading from that state
        
        # Therefore, if the state is terminal, then assign a real value
        # 1 if I win
        # 0 if no one wins
        
        # However, if a state is not terminal, then assess the subsequent states (i.e., opponent movespace)
        # to see if there are any bad outcomes. If there are, then assign a negative value to that state so we 
        # do not enable bad outcomes.
        
        terminated = next_state.is_terminated()
        winner = next_state.winner()
        
        if terminated:
            if winner == my_value:
                return 1
            else:
                return 0

        else:  
            further_states = next_state.next_available_states()
            for further_state in further_states:
                # If opponent's move ends the game, then very bad
                if further_state.winner() != my_value and further_state.winner() is not None:
                    return -1
            return 0

    def assess_next_states(self, state, my_value):
        next_states = state.next_available_states()
        next_state_assessment = pd.DataFrame(columns=['agent', 'next_state', 'agent_value_judgment'])
        
        for next_state in next_states:
            next_state_string = next_state.encode()
            value_judgment = self.make_value_judgment(next_state, my_value)
            new_row = pd.DataFrame({
                'agent' : str(self.name),
                'next_state': str(next_state_string), 
                'agent_value_judgment' : str(value_judgment)
            }, index=[0])
            next_state_assessment = pd.concat([next_state_assessment, new_row], ignore_index=True)
                    
        return next_state_assessment
    
    def policy(self, state):
        my_value = state.which_value_is_next()
        
        if self.is_human:
            row = int(input("Which row do you want to place in (0, 1, or 2)"))
            col = int(input("Which col do you want to place in (0, 1, or 2)"))    
        else:
            next_state_assessment = self.assess_next_states(state, my_value)
            #print(next_state_assessment)

            value_list = next_state_assessment['agent_value_judgment'].tolist()        
            value_max_indices = [
                index for index, current_value in enumerate(value_list) if current_value == max(value_list)
            ]     
            value_max_index = random.sample(value_max_indices, 1)[0]

            # Extract the value-maximizing next state
            value_max_state_encoded = str(next_state_assessment['next_state'].to_list()[value_max_index])
            value_max_state = Board(value_max_state_encoded)
            row, col, my_value = state.extract_move_transition(value_max_state)
        return (row, col, my_value)
           
class Controller:
    
    def __init__(self, agents):
        self.board = Board()
        self.agents = agents
        self.agent_index = 0
    
    def next_agent(self):
        # Retrieve the current item
        item = self.agents[self.agent_index]

        # Increment the index, using modulo to loop back to the start
        self.agent_index = (self.agent_index + 1) % len(self.agents)

        return item
    
    def simulate(self):        
        print("** Begin Game **")
        # Render environment
        self.board.render()
        for _ in range(1000):
                    
            # Select action via policy
            agent = self.next_agent()
            print(agent.name)
            action = agent.policy(self.board)
            
            observation, reward, done = self.board.step(action)
            self.board = observation

            winner = self.board.winner()
            
            # Render environment
            self.board.render()
            
            if done:
                print("** Game Over! The winning value is {}**".format(winner))
                break

            
def main():
    agents = [
                Agent("USA", is_human=False), 
                Agent("France", is_human=False)
    ]
    game = Controller(agents)
    game.simulate()  
    
main()
