import os
import sys
import time
import datetime
import pandas as pd
import numpy as np
import csv
import logging
import cv2 as cv
from PIL import Image
import pillow_heif
import matplotlib.pyplot as plt

# Main script, which contains parent classes and methods with common functionality

# Rework Game class to create initial object with it and then pass it to other classes.
class Game():
    """
    A class to manage the main functionalities of the game.
    
    Methods:
    -------
    restart_game():
        Resets the game state.
    
    exit_game(code):
        Exits the game with a specified exit code.
    
    check_error():
        Checks for errors and exits the game if any are found.
    
    user_exit_prompt():
        Prompts the user to continue or exit the game.
    
    ensured_input(inp_text, class_type=None, y_or_n=False, range_=None):
        Ensures the user input is valid and returns it.
    
    save_game(game_name, players, scores_df=None):
        Saves the current game state.
    
    load_game():
        Loads a previously saved game.
    """
    def __init__(self):
        """Initializes the Main class with default error state."""
        self._error = False

    def restart_game(self):
        """Resets the game state."""
        self._error = False
        self._user_exit = False       

    def exit_game(self, code):
        """
        Exits the game with a specified exit code.

        Parameters:
        code (int): The exit code to use when exiting the game.
        """
        print(f"Exiting with code {code}")
        # ADD SAVE LOGS
        time.sleep(2)
        sys.exit(code)

    def check_error(self):
        """Checks for errors and exits the game if any are found."""
        if self._error:
            self.exit_game(1)
    
    def user_exit_prompt(self):
        """
        Prompts the user to continue or exit the game.

        Returns:
        None
        """
        esc = input("If you want to continue - type c\nIf you want to exit - type x\n").lower()
        if esc == 'x':
            self.exit_game(0)
        return None

    def ensured_input(self, inp_text, class_type=None, y_or_n=False, range_=None):
        """
        Returns input if it is correct, otherwise prompts again (asks if exit).

        Specify class_type to make sure that input belongs to class_type.
        Specify range_ as a tuple (min, max) to restrict integer inputs to a specific range.
        for y/n inputs use: y_or_n=True.

        Parameters:
        inp_text (str): The input prompt text.
        class_type (type, optional): The expected type of the input. Default is None.
        y_or_n (bool, optional): If True, expects 'y' or 'n' input. Default is False.
        range_ (tuple, optional): A tuple specifying the range for integer inputs. Default is None.

        Returns:
        The validated user input.
        """
        if class_type is None and not y_or_n:
            raise ValueError("You need to specify class_type or put y_or_n=True")

        def convert_input(item, class_type):
            try:
                if class_type == int:
                    value = int(item)
                    if range_ and (value < range_[0] or value > range_[1]):
                       return None
                    return value
                elif class_type == float:
                    return float(item)
                elif class_type == str:
                    if item.isalpha():
                        return item
                    else:
                        raise ValueError
            except ValueError:
                return None

        item = input(inp_text)
        if y_or_n:
            while item.lower() not in ['y', 'n']:
                item = input("Input wasn't recognized. Please enter (y/n): ")
                if item.lower() not in ['y', 'n']:
                    self.user_exit_prompt()
            return item
        else:
            item = convert_input(item, class_type)
            while not isinstance(item, class_type):
                item = input("Input wasn't recognized. Please enter correct input: ")
                item = convert_input(item, class_type)
                if not isinstance(item, class_type):
                    self.user_exit_prompt()
            return item

    def save_game(self, game_name, players, scores_df=None):
        """
        Saves the current game state.

        Parameters:
        game_name (str): The name of the game.
        players (list): A list of player objects.
        scores_df (DataFrame, optional): A DataFrame containing the game scores. Default is None.
        """
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        save_name = f"{game_name}_{time_stamp}"
        os.makedirs('../saved_games/' + save_name, exist_ok=True)
        with open('../saved_games/' + save_name + '/players.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'crowns'])
            for player in players:
                writer.writerow([player.name, player.crowns])
        print(f"Game saved")
        time.sleep(2)
        if scores_df is not None:
            try:
                scores_df.to_csv('../saved_games/' + save_name + '/scores.csv', index=False)
                print(f"Scores saved")
                time.sleep(2)
            except (NameError, AttributeError):
                print("Error saving scores")
                logging.error("Error saving scores")
                time.sleep(2)

    def load_game(self):
        """
        Loads a previously saved game.

        Returns:
        tuple: A tuple containing the players and scores DataFrame, or (None, None) if no saved games are found.
        """
        saved_games_path = '../saved_games'
        
        # Check if the directory exists and is not empty
        if not os.path.exists(saved_games_path) or not os.listdir(saved_games_path):
            print("No saved games found.")
            return None, None
        
        games = os.listdir(saved_games_path)
        print("Choose the game to load:")
        for i, game in enumerate(games):
            print(f"{i+1}. {game}")
        
        try:
            choice = self.ensured_input("Enter the number of the game: ", int, range_=(1, len(games)))
            if choice < 1 or choice > len(games):
                raise ValueError
        except ValueError:
            print("Wrong input")
            time.sleep(1)
            return None, None

        game = games[choice - 1]
        players = []
        
        try:
            with open(f'{saved_games_path}/{game}/players.csv', 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    player = Player(row[0])
                    player.crowns = int(row[1])
                    players.append(player)
            
            scores_df = pd.read_csv(f'{saved_games_path}/{game}/scores.csv')
        except FileNotFoundError:
            print("Error: Saved game files not found.")
            return None, None
        except Exception as e:
            print(f"An error occurred while loading the game: {e}")
            logging.error(f"An error occurred while loading the game: {e}")
            return None, None

        print(f"Game loaded")
        time.sleep(2)
        return players, scores_df

class Player(Game):
    """
    A class to represent a player in the game.
    
    Attributes:
    ----------
    name : str
        The name of the player.
    
    crowns : int
        The number of crowns the player has.
    """
    def __init__(self, name):
        """
        Initializes the Player class with a name and default crowns.
        
        Parameters:
        name (str): The name of the player.
        """
        super().__init__()
        self._name = name
        self._crowns = 0
        self._scores = []

    @property
    def name(self):
        """Gets the name of the player."""
        return self._name
    
    @name.setter
    def name(self, value):
        """Sets the name of the player."""
        self._name = value

    @property
    def crowns(self):
        """Gets the number of crowns the player has."""
        return self._crowns
    
    @crowns.setter
    def crowns(self, value):
        """Sets the number of crowns the player has."""
        self._crowns = value

    @crowns.deleter
    def crowns(self):
        """Deletes the crowns of the player (sets to 0)."""
        self._crowns = 0

class Board(Game):
    def __init__(self, img_path):
        super().__init__()
        self._img_path = img_path
        self._img = self._load_image()
        self._img_size = self._img_size()

    def _load_image(self):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        try:
            if not self._img_path.lower().endswith(valid_extensions):
                raise ValueError(f"File at path {self._img_path} is not a valid image file.")
        except ValueError:
            print("Wrong file extension")
            return None
        img = cv.imread(self._img_path, cv.IMREAD_COLOR)
        if img is None:
            try:
                os.remove(self._img_path)
                print(f"Image at path {self._img_path} could not be loaded and has been deleted.")
            except OSError as e:
                print(f"Error deleting file: {e}")
            raise ValueError(f"Image at path {self._img_path} could not be loaded.")
        
        return img


    def _img_size(self):
        return (self._img.shape[0], self._img.shape[1])

    # Do you know that matplotlib expects RGB and OpenCV uses BGR? That leads to "negative" colors
    def display_image_self(self, title='', bgr=False):
        if bgr:
            bgr_image = cv.cvtColor(self._img, cv.COLOR_RGB2BGR)
            cv.imshow(title, bgr_image)
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            cv.imshow(title, self._img)
            cv.waitKey(0)
            cv.destroyAllWindows()

    def display_image(self, img, title='', cmap='gray'):
        # make bigger image
        plt.figure(figsize=(10, 10))
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis('on')
        plt.xlim(0, self._img.shape[1])
        plt.ylim(self._img.shape[0], 0)
        plt.show()

    def get_image(self):
        return self._img    

    def save_image(self, filename):
        cv.imwrite(filename, self._img)
    
if __name__ == "__main__":
    pass