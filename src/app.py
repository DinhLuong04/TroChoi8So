import os
import tkinter as tk
import time
from tkinter import ttk
from PIL import Image, ImageTk
from threading import Thread

from src.config import *
from src.utils import algorithm, Board, A_STAR, BFS

class EightPuzzle(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title('Tro choi 8 so')
        self.geometry('750x750')
        self.resizable(False, False)
        self.iconbitmap('src/assets/images/app.ico')
        self.protocol('WM_DELETE_WINDOW', lambda: os._exit(0))
        
        self.container = tk.Frame(self)
        self.container.pack(side='top', fill='both', expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        
        self.show_frame(PuzzlePage, **BASIC_FRAME_PROPERTIES)
    
    def show_frame(self, page, *args, **kwargs):
        frame = page(self.container, self, *args, **kwargs)
        frame.grid(row=0, column=0, sticky='nsew')
        frame.tkraise()

class PuzzlePage(tk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.controller = controller
        
        self.moves = 0
        self.board = []
        
        self.available_algorithms = [
            algorithm('A*', A_STAR),
            algorithm('BFS', BFS)
        ]
        self.algorithm_index = 0
        self.algorithm = self.available_algorithms[0]
        
        self.current_board_state = tuple(range(9))
        self.goal_board_state = tuple(range(9))
        self.saved_board_state = tuple(range(9))
        
        self.tile_images = [ImageTk.PhotoImage(Image.open(f'src/assets/images/tile_{n}.png')) for n in range(9)]
        
        self.is_stopped = False
        self.is_solving = False
        self.is_done = False
        
        self.display_widgets()
    
    def display_widgets(self):
        # Title section
        self.frame_title = tk.Frame(self, **BASIC_FRAME_PROPERTIES)
        self.frame_title.pack(pady=25)
        
        self.label_heading = tk.Label(self.frame_title, text='Tro choi 8 so', **HEADING_LABEL_PROPERTIES)
        self.label_heading.pack()
        
        self.label_subheading = tk.Label(self.frame_title, text=f'Giai thuat su dung: {self.algorithm.name} ', **SUBHEADING_LABEL_PROPERTIES)
        self.label_subheading.pack()
        
        # Puzzle section
        self.frame_puzzle = tk.Frame(self, **BASIC_FRAME_PROPERTIES)
        self.frame_puzzle.pack(padx=10, pady=10)
        
        #Button section
        self.frame_buttons = tk.Frame(self, **BASIC_FRAME_PROPERTIES)
        self.frame_buttons.pack(pady=20)
        
        self.button_solve = tk.Button(self.frame_buttons, text='Giai', command=lambda: self.solve_board(), **PRIMARY_BUTTON_PROPERTIES)
        self.button_solve.grid(row=0, column=0, padx=10, pady=10)
        
        self.button_reset = tk.Button(self.frame_buttons, text='reset', command=lambda: self.reset_board(), **SECONDARY_BUTTON_PROPERTIES)
        self.button_reset.grid(row=0, column=1, padx=10, pady=10)
        
        self.button_shuffle = tk.Button(self.frame_buttons, text='Xao tron', command=lambda: self.shuffle_board(), **PRIMARY_BUTTON_PROPERTIES)
        self.button_shuffle.grid(row=0, column=2, padx=10, pady=10)
        
        self.button_change = tk.Button(self.frame_buttons, text='change', command=lambda: self.change_algorithm(), **TERTIARY_BUTTON_PROPERTIES)
        self.button_change.grid(row=0, column=3, padx=10, pady=10)
        
        self.label_moves = tk.Label(self.frame_puzzle, text=f'So buoc: {self.moves}', **TEXT_LABEL_PROPERTIES)
        self.label_moves.grid(row=0, column=0, sticky='w', padx=10, pady=5)
        
        self.label_status = tk.Label(self.frame_puzzle, text=f'Playing...', **TEXT_LABEL_PROPERTIES)
        self.label_status.grid(row=0, column=1, sticky='e', padx=10, pady=5)
        
        self.separator = ttk.Separator(self.frame_puzzle, orient='horizontal')
        self.separator.grid(row=1, columnspan=2, sticky='ew', pady=10)
        
        self.frame_board = tk.Frame(self.frame_puzzle, **BASIC_FRAME_PROPERTIES)
        self.frame_board.grid(row=2, columnspan=2)
        
        self.initialize_board()
        
        self.shuffle_board()
        
        self.controller.bind('<Up>', lambda event: self.transform_keys('D'))
        self.controller.bind('<Down>', lambda event: self.transform_keys('U'))
        self.controller.bind('<Left>', lambda event: self.transform_keys('R'))
        self.controller.bind('<Right>', lambda event: self.transform_keys('L'))
    
    def initialize_board(self):
        for index in range(9):
            self.board.append(tk.Button(self.frame_board, **TILE_BUTTON_PROPERTIES))
            self.board[index].grid(row=index // 3, column=index % 3, padx=10, pady=10)
    
    def populate_board(self, state, delay_time=0):
        for tile_index, tile_value in enumerate(state):
            self.board[tile_index].configure(
                    image=self.tile_images[tile_value],
                    text=tile_value,
                    state='normal',
                    command=lambda tile_index=tile_index: self.transform_click(tile_index)
                )
            
            if tile_value == 0:
                self.board[tile_index].configure(state='disabled')
        
        self.current_board_state = state
        
        time.sleep(delay_time)
    
    def solve_board(self):
        if not self.is_solving:
            self.reset_board()
            self.solution_thread = Thread(target=self.run_solution)
            self.solution_thread.start()
    
    def run_solution(self):
        self.is_stopped = False
        self.is_solving = True
        self.is_done = False
        self.update_status('Dang giai...')
        
        print('\nDang tim giai phap...')
        
        path_to_goal, nodes_expanded, max_search_depth, time_elasped = Board.solve(self.current_board_state, self.algorithm.func)
        
        if not self.is_stopped:
             print(f'Hoan thanh trong {round(time_elasped, 4)} giay voi {len(path_to_goal)} buoc su dung thuat toan {self.algorithm.name}')
             print(f'Do sau tim kiem toi da la {max_search_depth} va so nut mo rong la {nodes_expanded}')
             print('Cac hanh dong:', *path_to_goal)
        else:
            print('Stopped')
        
        if path_to_goal:
            print('\nMoving board...')
            self.update_status('Dang di chuyen...')
            
            delay_time = 0.75
            time.sleep(delay_time)
            
            for action in path_to_goal:
                if self.is_stopped:
                    print('Stopped')
                    self.update_status('Dang choi...')
                    break
                else:
                    self.transform_state(action, delay_time=0.5)
            else:
                print('Hoan thanh chuyen dong ban co')
                self.update_status('Da giai xong!')
                self.is_done = True
            
            self.is_solving = False
        
        else:
            self.is_solving = False
            self.update_status('Dang choi...')
    
    def reset_board(self):
        self.stop_solution()
        self.update_moves(0)
        self.update_status('Dang choi...')
        self.populate_board(state=self.saved_board_state)
    
    def shuffle_board(self):
        self.saved_board_state = Board.create_solvable_state()
        self.reset_board()
    
    def stop_solution(self):
        if self.is_solving and not self.is_stopped:
            self.is_stopped = True
        self.is_done = False
    
    def change_algorithm(self):
        self.reset_board()
        self.algorithm_index = (self.algorithm_index + 1) % len(self.available_algorithms)
        self.algorithm = self.available_algorithms[self.algorithm_index]
        self.label_subheading.configure(text=f'Giai bang thuat toan  {self.algorithm.name} ')
    
    def transform_click(self, tile_index):
        possible_actions = Board.valid_actions(self.current_board_state)
        blank_index = self.current_board_state.index(0)
        tile_value = int(self.board[tile_index].cget('text'))
        
        for action in possible_actions:
            if not self.is_solving and not self.is_done:
                if action == 'U' and self.current_board_state[blank_index - 3] == tile_value:
                    self.transform_state(action)
                
                elif action == 'D' and self.current_board_state[blank_index + 3] == tile_value:
                     self.transform_state(action)
                
                elif action == 'L' and self.current_board_state[blank_index - 1] == tile_value:
                     self.transform_state(action)
                
                elif action == 'R' and self.current_board_state[blank_index + 1] == tile_value:
                     self.transform_state(action)
        
        if not self.is_done and self.current_board_state == self.goal_board_state:
            self.update_status('Well done!')
            self.is_done = True
    
    def transform_keys(self, action):
        if not self.is_solving and not self.is_done:
           if action in Board.valid_actions(self.current_board_state):
                self.transform_state(action)
        
        if not self.is_done and self.current_board_state == self.goal_board_state:
            self.update_status('Well done!')
            self.is_done = True
    
    def transform_state(self, action, delay_time=0):
        new_state = Board.transform(self.current_board_state, action)
        
        current_index = self.current_board_state.index(0)
        new_index = new_state.index(0)
        
        first_tile = self.board[current_index]
        second_tile = self.board[new_index]
        
        first_tile_properties = self.get_tile_property(first_tile)
        second_tile_properties = self.get_tile_property(second_tile)
        
        self.set_tile_property(first_tile, second_tile_properties)
        self.set_tile_property(second_tile, first_tile_properties)
        
        self.current_board_state = new_state
        
        if not self.is_done:
            self.update_moves(self.moves + 1)
        
        time.sleep(delay_time)
    
    def get_tile_property(self, tile):
        return {
            'text': tile.cget('text'),
            'background': tile.cget('background'),
            'image': tile.cget('image'),
            'state': tile.cget('state')
        }
    
    def set_tile_property(self, tile, properties):
        tile.configure(**properties)
    
    def update_moves(self, moves):
        self.moves = moves
        self.label_moves.configure(text=f'Moves: {self.moves}')
    
    def update_status(self, status):
        self.label_status.configure(text=status)
