#----------------------------------------------------------------------------
# tictactoe.py
# Author: Keith Gladstone
# Purpose: Trying to eventually learn reinforcement learning,
#          by building the mechanics of a simple tic-tac-toe 
#          game. There are three abstractions that work together:
#            1 - Board: a representation of the game board, with 
#                       built-in play validation rules, ability 
#                       to provide the next available states, and 
#                       assess if a given board position is game 
#                       over. Simply states truths. Does not make 
#                       any decisions. The key here is that the underlying
#                       data structure is an array, but instead
#                       the clients manipulate the board in a
#                       more intuitive way (e.g., crawling through,
#                       setting values by "cell index", understanding
#                       a given state of the board in relation to others).
#            2 - Player: a client of a given board within a 
#                       TicTacToeEnv. Has the ability to make basic
#                       selection decisions of the next move by 
#                       accessing truths about the state of the
#                       board and assigning value to different
#                       choices.
#            3 - TicTacToeEnv: a global environment that runs a 
#                       game by resetting the board, instantiating
#                       two players, and running a step-based 
#                       simulation.
#----------------------------------------------------------------------------
from functools import reduce
import random
from itertools import product
import pandas as pd


class Board:
    
    @staticmethod
    def increment(x):
        return x + 1
    
    @staticmethod
    def decrement(x):
        return x - 1
    
    @staticmethod
    def add(x, y):
        return x + y
    
    @staticmethod
    def multiply(x, y):
        return x * y
    
    @staticmethod
    def prepare_board_input_from_list(input_list):
        return [input_list[i:i+3] for i in range(0, len(input_list), 3)]
    
    @staticmethod
    def decode_board_string(board_string):
        char_list = list(board_string)
        int_list = [int(i) for i in char_list]
        int_list_reset = list(map(Board.decrement, int_list))
        board_representation = Board.prepare_board_input_from_list(int_list_reset)
        return board_representation      
    
    @staticmethod
    def _encode_board_object_list(board_as_list):
        
        # Validate there are no negatives in the diff
        for element in board_as_list:
            if element not in [0, 1, 2]:
                print("Error: Trying to encode invalid board object list")
                return ""
        
        board_nonzero_integer_list = list(map(Board.increment, board_as_list))
        board_string_list = [str(i) for i in board_nonzero_integer_list]
        board_string = ''.join(board_string_list)
        return str(board_string)
    
    def __init__(self, encoded_string=""):
        self.COLMAX = 3
        self.ROWMAX = 3
        if encoded_string == "":
            self.reset() 
        else:
            self.board = Board.decode_board_string(str(encoded_string))
            assert self.is_board_valid()
        return
    
    def reset(self):
        self.board = [[0, 0, 0],  # First row
             [0, 0, 0],  # Second row
             [0, 0, 0]]  # Third row
        
    def equals(self, other_board):
        return self.encode() == other_board.encode()
        
    def is_cell_empty(self, row, col):
        return str(self.board[row][col]) == "0"
    
    def is_board_valid(self):
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
                if not self.is_cell_valid(i, j):
                    return False
        return True
    
    def is_cell_valid(self, row, col):
        if not (row < self.ROWMAX and row >= 0):
            print("Error: Row {} out of range".format(row))
            return False
        if not (col < self.COLMAX and col >= 0):
            print("Error: Col {} out of range".format(col))
            return False
        return True
    
    def is_set_value_valid(self, row, col, value):
        if not self.is_cell_valid(row, col):
            print("Error: Cell ({},{}) is not valid".format(row, col))
            return False
        if not(value in [1, 2]):
            print("Error: Value {} illegal".format(value))
            return False
        if not (self.is_cell_empty(row, col)):
            print("Error: Cell ({},{}) is not empty".format(row, col))
            return False
        return True
    
    def assert_direction_valid(self, direction):
        assert direction in ["N", "S", "E", "W", "NE", "SE", "SW", "NW"], "Direction {} illegal".format(direction)
        return True
    
    def get_value(self, row, col):
        assert (self.is_cell_valid(row, col))
        return self.board[row][col]
    
    def safe_set_value(self, row, col, value):
        if not self.is_set_value_valid(row, col, value):
            print("Error: Not valid to set value {} to this cell ({},{})".format(value, row, col))
            return False
        else:
            self.set_value(row, col, value)
            return True
    
    def set_value(self, row, col, value):
        self.board[row][col] = value
        return True
    
    def get_cell_index_adjacent(self, origin_row, origin_col, direction):
        self.assert_direction_valid(direction)
        
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
        
        if not self.is_cell_valid(new_row, new_col):
            print("Error: Trying to reach invalid cell")
            return None
        return (new_row, new_col)        
    
    def _is_three_match_in_line(self, origin_row, origin_col, direction):
        first_value = self.get_value(origin_row, origin_col)
        
        # Check cell has nonzero value
        if str(first_value) == '0':
            return False, -1
        
        # Check two in a line
        next_row, next_col = self.get_cell_index_adjacent(origin_row, origin_col, direction)
        second_value = self.get_value(next_row, next_col)
        if first_value != second_value:
            return False, -1
        else:
            
            # Check three in a line
            next_row, next_col = self.get_cell_index_adjacent(next_row, next_col, direction)
            third_value = self.get_value(next_row, next_col)
            if second_value != third_value:
                return False, -1
            else:
                return True, first_value
            
    def three_match_exists(self):
        three_match_list = self.get_all_three_matches()
        if len(three_match_list) > 0:
            row, col, direction, value = three_match_list[0]
            return True, value
        else:
            return False, -1
        
    def get_all_three_matches(self):
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
        return three_match_list        
    
    # Invariant - board should not be attainable if multiple threematches exist; game should be over
    def is_board_reachable(self):
        return len(self.get_all_three_matches()) <= 1
    
    def count_nonempty_cells(self):
        ct = 0
        for i in range(0, self.ROWMAX):
            for j in range(0, self.COLMAX):
                if not self.is_cell_empty(i, j):
                    ct += 1
        return ct

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
   
    def which_value_is_next(self):
        count_X = 0
        count_O = 0
        for i in range(0, self.ROWMAX):
            for j in range(0, self.COLMAX):
                value = self.get_value(i, j)
                if value == 1:
                    count_X += 1
                elif value == 2:
                    count_O += 1
        if count_X > count_O:
            return 2
        else:
            return 1
                    
    def next_available_moves(self):
        next_states = self.next_available_states()
        next_moves = [self.extract_move_transition(i) for i in next_states]
        return next_moves
    
    def get_first_observed_move(self):
        for i in range(0, self.ROWMAX):
            for j in range(0, self.COLMAX):
                if not self.is_cell_empty(i, j):
                    return (i, j, self.get_value(i, j))
    
    # Return as a board, but if diff is invalid then return None
    def diff(self, other):
        self_encoded = [int(i) for i in list(str(self.encode()))]
        other_encoded = [int(i) for i in list(str(other.encode()))]
        result = [a - b for a, b in zip(other_encoded, self_encoded)]        
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
            return diff.count_nonempty_cells() == 1

    def extract_move_transition(self, other):
        if not self.is_valid_transition(other):
            print("Invalid Transition")
            return (-1, -1, -1)
        diff = self.diff(other)
        move_made = diff.get_first_observed_move()
        return move_made
        
    def get_game_over_conditions(self):
        three_match_exists, value = self.three_match_exists()
        if three_match_exists:
            return True, value
        elif self.is_full():
            return True, -1
        else:
            return False, -1
    
    def _to_list(self):
        board_as_list = list()
        for i in range(0, self.ROWMAX):
            for j in range(0, self.COLMAX):
                board_as_list.append(self.get_value(i, j))
        return board_as_list
    
    def encode(self):
        return self._encode_board_object_list(self._to_list())
    
    def apply_reduce(self, f):
        board_as_list = self._to_list()
        return reduce(f, board_as_list)
    
#     def apply_map(self, f):
#         board_as_list = self._to_list()
#         return list(map(f, board_as_list))
    
    def is_full(self):
        return self.apply_reduce(Board.multiply) != 0
    
    def value_to_string(self, value):
        if str(value) == "1":
            return "X"
        elif str(value) == "2":
            return "O"
        elif str(value) == "0":
            return " "
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

class Player:
    
    def __init__(self, name, value, env):
        self.name = name
        self.value = value
        self.board = env.board
    
    def to_string(self):
        render_value = self.board.value_to_string(self.value)
        return "Player {} has access to the value {}".format(self.name, render_value)
    
    def is_cell_empty(self, row, col):
        return self.board.is_cell_empty(row, col)
    
    def make_value_judgment(self, board_state):
        would_be_game_over, winning_piece = board_state.get_game_over_conditions()
        if not would_be_game_over:
            return 0
        else:
            if winning_piece == self.value:
                return 1
            else:
                return -1
    
    def assess_next_states(self):
        next_states = self.board.next_available_states()
        next_state_assessment = pd.DataFrame(columns=['board_string', 'next_move', 'value_judgment'])
        
        for next_state in next_states:
            next_state_string = next_state.encode()
            value_judgment = self.make_value_judgment(next_state)
            new_row = pd.DataFrame({
                'board_string': str(next_state_string), 
                'next_move' : str(self.board.value_to_string(self.value)),
                'value_judgment' : str(value_judgment)
            }, index=[0])
            next_state_assessment = pd.concat([next_state_assessment, new_row], ignore_index=True)
        return next_state_assessment
    
    def select_move(self):     
        next_state_assessment = self.assess_next_states()
        print("Next State Assessment")
        print(next_state_assessment)
        
        # If a winning move exists then play it
        value_list = next_state_assessment['value_judgment'].tolist()
        value_max = max(value_list)
        if float(value_max) > 0.9999:            
            value_max_index = value_list.index(value_max)

            # Extract the value-maximizing next state
            value_max_state_encoded = str(next_state_assessment['board_string'].to_list()[value_max_index])
            value_max_state = Board(value_max_state_encoded)
            
            value_max_state.render()

            game_ending_move = self.board.extract_move_transition(value_max_state)
            print("Player {} says: I know how to pick the winner!!!!!!!!".format(self.name))
            return game_ending_move
                    
        # Else choose randomly
        else:
            return self.select_move_naive()
        
    def select_move_naive(self):
        # Naively place in first available empty square
        next_states = self.board.next_available_states()
        random_next_state = random.sample(next_states, 1)[0]
        random_next_move = self.board.extract_move_transition(random_next_state)
        print("Player {} says: No clear winner, so I am choosing nonstrategically".format(self.name))
        return random_next_move
    
#     def test_board_string(self, board_string):
        
    
class TicTacToeEnv:
    
    def __init__(self, player_1_name, player_2_name):
        self.board = Board()
        self.player_1 = Player(player_1_name, 1, self)
        self.player_2 = Player(player_2_name, 2, self)
        self.turn = True
       
    def is_game_over(self):
        game_over, winning_piece = self.board.get_game_over_conditions()
        return game_over
    
    def render(self):
        self.board.render()
        
    def receive_play(self, row, col, value):
        print("Value {} received for row {} and col {}".format(value, row, col))
        self.board.safe_set_value(row, col, value)
        return
        
    def step(self):
        if self.turn:
            player = self.player_1
        else:
            player = self.player_2
        print("\n*****\nNew Turn")
        print(player.to_string())
        row, col, value = player.select_move()
        self.receive_play(row, col, value)
        
        # End Step by flipping turn
        self.render()
        print(self.board.encode())
        self.turn = not self.turn
        
    def simulate(self):
        LIMIT = 30
        counter = 0
        
        print("New Game")
        self.render()

        while (counter < LIMIT):
            # Check if game over
            game_over, winning_piece = self.board.get_game_over_conditions()
            if game_over:
                print("Game Over! The winning value is {}".format(winning_piece))
                break
            else:
                self.step()
                counter += 1
                
    def generate_board_string_universe(self):
        # Generate all combinations of 9 digit strings using only the digits 1, 2, and 3
        nine_digit_strings = [''.join(p) for p in product('123', repeat=9)]
        return nine_digit_strings

                
def main():
    game = TicTacToeEnv("Red", "Blue")
    game.simulate()  

# Execute
main()

#-------------------------------------------------------------------------------------------------------------
#### Random experimental stuff
#     universe = game.generate_board_string_universe()
#     print("Count of boards, pre-pruning: {}".format(len(universe)))
    
#     #LIMIT UNIVERSE
#     #universe = random.sample(universe, 100)
    
#     universe_df = pd.DataFrame(columns=['board_string', 'is_game_over', 'winning_piece'])

#     # Generate outcome data
#     for board_string in universe:
#         myboard = Board(board_string)
#         # Prune any boards that are unreachable
#         if myboard.is_board_reachable():
#             game_over, winning_piece = myboard.get_game_over_conditions()
#             new_row = pd.DataFrame({
#                 'board_string': board_string, 
#                 'is_game_over': str(game_over), 
#                 'winning_piece' : str(winning_piece)
#             }, index=[0])
#             universe_df = pd.concat([universe_df, new_row], ignore_index=True)
#             #print("{} | {} | {}".format(board_string, str(game_over), str(winning_piece)))
#             #myboard.render()

#     print("Count of boards, post-pruning: {}".format(len(universe_df)))
#     print(universe_df)
    
#     board = Board()
#     board.render()
#     board.set_value(0,0,1)
#     board.set_value(1,1,1)
#     board.set_value(2,2,1)
#     board.render()
#     print(board._is_three_match_in_line(0, 0, "E"))
#     print(board._is_three_match_in_line(0, 0, "SE"))
#     print(board.three_match_exists())
#     print(board.is_full())

#     board_string = "232323211"
