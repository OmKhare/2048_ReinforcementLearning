from random import randint
from copy import copy
class Game2048():
    def __init__(self, human=True):
        self.boardx = 4
        self.boardy = 4
        self.board = [0 for i in range(self.boardx*self.boardy)]
        self.turn = 0
        self.empties = []
        self.human = human
        self.terminal = False
        self.functions =    {
                            'w': self.up,
                            'a': self.left,
                            's': self.down,
                            'd': self.right,
                            }
        self.step('w')
        pass
    
    def getBoard(self):
        return self.board
    
    def getTurn(self):
        return self.turn
    
    def getTerminal(self):
        return self.terminal
    def printboard(self):
        print("################################################")
        print("")
        for j in range(self.boardy):
            print(" {:5}      {:5}      {:5}      {:5}".format(self.board[4*j],self.board[4*j+1],self.board[4*j+2], self.board[4*j+3]))
            print("")
        print("################################################")
        print("Turn: {}".format(self.turn))
        return

    def addRandom(self):
        self.board[self.empties[randint(0,len(self.empties)-1)]]=2
    
    def getEmpties(self):
        self.empties = []
        for i in range(len(self.board)):
            if self.board[i]==0:
                self.empties.append(i)
        return self.empties
        
    def coordtoindx(self,x,y):
        return 4*y+x
    
    def indxtocoord(self, index):
        x = index%4
        y = int((index-x)/4)
        return x, y

    def getUserInput(self):
        entered = input()
        print("")
        print("")
        if (entered == 'w' or entered == 'a' or entered == 's' or entered == 'd' or entered == 'q'):
            return entered
        else:
            print("That is not valid input, try again!")
            return self.getUserInput()
    
    def step(self, action):
        put_random=True
        self.turn+=1
        self.functions[action]()
        
        if len(self.getEmpties())==0:
            if self.checkEnd():
                self.endGame()
                return
            else:
                put_random = False

        if put_random:
            self.addRandom()

        self.printboard()

        if self.human:
            new_action = self.getUserInput()
            if new_action == 'q':
                self.endGame()
                return 0
            self.step(new_action)

        return self.board
    
    def getLegalMoves(self):
        legal_moves = []
        move_names = ['w','a','s','d']
        original = copy(self.board)

        for i in move_names:
            self.functions[i]()
            if original==self.board:
                legal_moves.append(0)
            else:
                legal_moves.append(1)
            self.board == copy(original)
        
        return legal_moves


        
    def checkEnd(self):
        original = copy(self.board)
        self.up()
        if original==self.board:
            self.right()
            if original==self.board:
                self.down()
                if original==self.board:
                    self.left()
                    if original==self.board:
                        return True
        self.board = copy(original)    
        return False

    def endGame(self):
        self.terminal = True
        print("################################################")
        print("##################GAME OVER#####################")
        self.printboard()
        pass


    def right(self):
        for j in range(4):
            for i in range(3):
                if self.board[self.coordtoindx(i+1,j)]==0:
                    self.board[self.coordtoindx(i+1,j)] = self.board[self.coordtoindx(i,j)]
                    self.board[self.coordtoindx(i,j)] = 0
                elif self.board[self.coordtoindx(i+1,j)] == self.board[self.coordtoindx(i,j)]:
                    self.board[self.coordtoindx(i+1,j)] = self.board[self.coordtoindx(i+1,j)] + self.board[self.coordtoindx(i,j)]
                    self.board[self.coordtoindx(i,j)] = 0
        for j in range(4):
            for i in range(3):
                if self.board[self.coordtoindx(i+1,j)]==0:
                    self.board[self.coordtoindx(i+1,j)] = self.board[self.coordtoindx(i,j)]
                    self.board[self.coordtoindx(i,j)] = 0

    def left(self):
        for j in range(4):
            for i in range(3):
                if self.board[self.coordtoindx(2-i,j)]==0:
                    self.board[self.coordtoindx(2-i,j)] = self.board[self.coordtoindx(3-i,j)]
                    self.board[self.coordtoindx(3-i,j)] = 0
                elif self.board[self.coordtoindx(2-i,j)] == self.board[self.coordtoindx(3-i,j)]:
                    self.board[self.coordtoindx(2-i,j)] = self.board[self.coordtoindx(2-i,j)] + self.board[self.coordtoindx(3-i,j)]
                    self.board[self.coordtoindx(3-i,j)] = 0
        for j in range(4):
            for i in range(3):
                if self.board[self.coordtoindx(2-i,j)]==0:
                    self.board[self.coordtoindx(2-i,j)] = self.board[self.coordtoindx(3-i,j)]
                    self.board[self.coordtoindx(3-i,j)] = 0
    
    def down(self):
        for i in range(4):
            for j in range(3):
                if self.board[self.coordtoindx(i,j+1)]==0:
                    self.board[self.coordtoindx(i,j+1)] = self.board[self.coordtoindx(i,j)]
                    self.board[self.coordtoindx(i,j)] = 0
                elif self.board[self.coordtoindx(i,j+1)] == self.board[self.coordtoindx(i,j)]:
                    self.board[self.coordtoindx(i,j+1)] = self.board[self.coordtoindx(i,j+1)] + self.board[self.coordtoindx(i,j)]
                    self.board[self.coordtoindx(i,j)] = 0
        for i in range(4):
            for j in range(3):
                if self.board[self.coordtoindx(i,j+1)]==0:
                    self.board[self.coordtoindx(i,j+1)] = self.board[self.coordtoindx(i,j)]
                    self.board[self.coordtoindx(i,j)] = 0
    
    def up(self):
        for i in range(4):
            for j in range(3):
                if self.board[self.coordtoindx(i,2-j)]==0:
                    self.board[self.coordtoindx(i,2-j)] = self.board[self.coordtoindx(i,3-j)]
                    self.board[self.coordtoindx(i,3-j)] = 0
                elif self.board[self.coordtoindx(i,2-j)] == self.board[self.coordtoindx(i,3-j)]:
                    self.board[self.coordtoindx(i,2-j)] = self.board[self.coordtoindx(i,2-j)] + self.board[self.coordtoindx(i,3-j)]
                    self.board[self.coordtoindx(i,3-j)] = 0
        for i in range(4):
            for j in range(3):
                if self.board[self.coordtoindx(i,2-j)]==0:
                    self.board[self.coordtoindx(i,2-j)] = self.board[self.coordtoindx(i,3-j)]
                    self.board[self.coordtoindx(i,3-j)] = 0


if __name__== "__main__":
    myGame = Game2048(human=True)