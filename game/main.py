import os
import sys
import time
import datetime
import pandas as pd
import csv
import logging

class Main:
    def __init__(self):
        self._error = False

    def restart_game(self):
        self._error = False
        self._user_exit = False       

    def exit_game(self, code):
        print(f"Exiting with code {code}")
        # ADD SAVE LOGS
        time.sleep(2)
        sys.exit(code)

    def check_error(self):
        if self._error:
            self.exit_game(1)
    
    def user_exit_prompt(self):
        """Returns None if continue"""
        esc = input("If you want to continue - type c\nIf you want to exit - type x\n").lower()
        if esc == 'x':
            self.exit_game(0)
        return None

    def ensured_input(self, inp_text, class_type=None, y_or_n=False, range_=None):
        """
        Returns input if it is correct, otherwise prompts again (ask if exit)\n
        Specify class_type to make sure that input belongs to class_type\n
        Specify range_ as a tuple (min, max) to restrict integer inputs to a specific range\n
        for y/n inputs use: y_or_n=True
        """
        if class_type is None and not y_or_n:
            raise ValueError("You need to specify class_type or put y_or_n=True")

        def convert_input(item, class_type):
            try:
                if class_type == int:
                    value = int(item)
                    if range_ and (value < range_[0] or value > range_[1]):
                       return None
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

class Player(Main):
    def __init__(self, name):
        super().__init__()
        self._name = name
        self._crowns = 0

    @property
    def name(self):
        return self._name
    
    # If you want to update player name
    @name.setter
    def name(self, value):
        self._name = value

    @property
    def crowns(self):
        return self._crowns
    
    # If you want to update number of crowns
    @crowns.setter
    def crowns(self, value):
        self._crowns = value

    @crowns.deleter
    def crowns(self):
        self._crowns = 0

if __name__ == "__main__":
    # I possibly will leave this script non-exec
    pass

