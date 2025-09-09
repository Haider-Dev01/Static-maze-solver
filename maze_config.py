import random


#constants

EMPTY = 0
Wall = 1
START = 2
END = 3

DEFAULT_ROWS = 10
DEFAULT_COLS = 10



#initialize_maze
def initialize_maze(rows=DEFAULT_ROWS,cols=DEFAULT_COLS):
    maze=[[EMPTY for c in range(cols)] for r in range(rows)]

    for i in range(rows):
        maze[i][0]=Wall
        maze[i][cols-1]=Wall
    
    for j in range(cols):
        maze[0][j]=Wall
        maze[rows-1][j]=Wall 

    maze[1][1]=START
    maze[rows-2][cols-2]=END

    return maze

# update_maze

def update_maze(maze,state):
    rows = len(maze)
    cols = len(maze[0])

    for s in range(state):

        r,c=random.randint(1,rows-2),random.randint(1,cols-2)
        maze[r][c]=Wall if maze[r][c]==EMPTY else EMPTY 

    return maze

# print_maze

def print_maze(maze):

    symbols = {EMPTY:'.',Wall:'#',START:'S',END:'E', -1:"A"}
    for row in maze:
        print(" ".join(symbols.get(cell,'?') for cell in row))
