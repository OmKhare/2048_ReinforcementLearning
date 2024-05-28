from random import randint
from copy import copy

class Game2048():
    def __init__(self, human=True, print_board=True):
        self.print_bool = print_board
        self.board_x = 4
        self.board_y = 4
        self.board = [0 for _ in range(self.board_x * self.board_y)]
        self.turn = 0
        self.empties = []
        self.human = human
        self.terminal = False
        self.functions = {
            'w': self.move_up,
            'a': self.move_left,
            's': self.move_down,
            'd': self.move_right,
        }
        self.move('w')
    
    def get_board(self):
        return self.board
    
    def get_turn(self):
        return self.turn
    
    def get_terminal(self):
        return self.terminal
    
    def print_board(self):
        print("################################################")
        print("")
        for j in range(self.board_y):
            print(" {:5}      {:5}      {:5}      {:5}".format(
                self.board[4 * j],
                self.board[4 * j + 1],
                self.board[4 * j + 2],
                self.board[4 * j + 3]
            ))
            print("")
        print("################################################")
        return

    def add_random(self):
        self.board[self.empties[randint(0, len(self.empties) - 1)]] = 2
    
    def get_empties(self):
        self.empties = [i for i in range(len(self.board)) if self.board[i] == 0]
        return self.empties
        
    def coord_to_index(self, x, y):
        return 4 * y + x
    
    def index_to_coord(self, index):
        x = index % 4
        y = int((index - x) / 4)
        return x, y

    def get_user_input(self):
        entered = input()
        print("\n\n")
        if entered in ['w', 'a', 's', 'd', 'q']:
            return entered
        else:
            print("That is not valid input, try again!")
            return self.get_user_input()
    
    def move(self, action):
        put_random = True
        self.turn += 1
        self.functions[action]()
        
        if len(self.get_empties()) == 0:
            if self.check_end():
                self.end_game()
                return
            else:
                put_random = False

        if put_random:
            self.add_random()
        
        if self.print_bool:
            self.print_board()
        print("Turn: {}".format(self.turn))

        if self.human:
            new_action = self.get_user_input()
            if new_action == 'q':
                self.end_game()
                return 0
            self.move(new_action)

        return self.board
    
    def get_legal_moves(self):
        legal_moves = []
        move_names = ['w', 'a', 's', 'd']
        original = copy(self.board)

        for i in move_names:
            self.functions[i]()
            legal_moves.append(0 if original == self.board else 1)
            self.board = copy(original)
        
        return legal_moves
        
    def check_end(self):
        original = copy(self.board)
        self.move_up()
        if original == self.board:
            self.move_right()
            if original == self.board:
                self.move_down()
                if original == self.board:
                    self.move_left()
                    if original == self.board:
                        return True
        self.board = copy(original)
        return False

    def end_game(self):
        self.terminal = True
        print("################################################")
        print("##################GAME OVER#####################")
        self.print_board()
        pass

    def move_right(self):
        for j in range(4):
            for i in range(3):
                if self.board[self.coord_to_index(i + 1, j)] == 0:
                    self.board[self.coord_to_index(i + 1, j)] = self.board[self.coord_to_index(i, j)]
                    self.board[self.coord_to_index(i, j)] = 0
                elif self.board[self.coord_to_index(i + 1, j)] == self.board[self.coord_to_index(i, j)]:
                    self.board[self.coord_to_index(i + 1, j)] += self.board[self.coord_to_index(i, j)]
                    self.board[self.coord_to_index(i, j)] = 0
        for j in range(4):
            for i in range(3):
                if self.board[self.coord_to_index(i + 1, j)] == 0:
                    self.board[self.coord_to_index(i + 1, j)] = self.board[self.coord_to_index(i, j)]
                    self.board[self.coord_to_index(i, j)] = 0

    def move_left(self):
        for j in range(4):
            for i in range(3):
                if self.board[self.coord_to_index(2 - i, j)] == 0:
                    self.board[self.coord_to_index(2 - i, j)] = self.board[self.coord_to_index(3 - i, j)]
                    self.board[self.coord_to_index(3 - i, j)] = 0
                elif self.board[self.coord_to_index(2 - i, j)] == self.board[self.coord_to_index(3 - i, j)]:
                    self.board[self.coord_to_index(2 - i, j)] += self.board[self.coord_to_index(3 - i, j)]
                    self.board[self.coord_to_index(3 - i, j)] = 0
        for j in range(4):
            for i in range(3):
                if self.board[self.coord_to_index(2 - i, j)] == 0:
                    self.board[self.coord_to_index(2 - i, j)] = self.board[self.coord_to_index(3 - i, j)]
                    self.board[self.coord_to_index(3 - i, j)] = 0
    
    def move_down(self):
        for i in range(4):
            for j in range(3):
                if self.board[self.coord_to_index(i, j + 1)] == 0:
                    self.board[self.coord_to_index(i, j + 1)] = self.board[self.coord_to_index(i, j)]
                    self.board[self.coord_to_index(i, j)] = 0
                elif self.board[self.coord_to_index(i, j + 1)] == self.board[self.coord_to_index(i, j)]:
                    self.board[self.coord_to_index(i, j + 1)] += self.board[self.coord_to_index(i, j)]
                    self.board[self.coord_to_index(i, j)] = 0
        for i in range(4):
            for j in range(3):
                if self.board[self.coord_to_index(i, j + 1)] == 0:
                    self.board[self.coord_to_index(i, j + 1)] = self.board[self.coord_to_index(i, j)]
                    self.board[self.coord_to_index(i, j)] = 0
    
    def move_up(self):
        for i in range(4):
            for j in range(3):
                if self.board[self.coord_to_index(i, 2 - j)] == 0:
                    self.board[self.coord_to_index(i, 2 - j)] = self.board[self.coord_to_index(i, 3 - j)]
                    self.board[self.coord_to_index(i, 3 - j)] = 0
                elif self.board[self.coord_to_index(i, 2 - j)] == self.board[self.coord_to_index(i, 3 - j)]:
                    self.board[self.coord_to_index(i, 2 - j)] += self.board[self.coord_to_index(i, 3 - j)]
                    self.board[self.coord_to_index(i, 3 - j)] = 0

if __name__ == "__main__":
    my_game = Game2048(human=True)