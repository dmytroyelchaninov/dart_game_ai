import time
import datetime
import sys
import logging
from main import Main, Player
import subprocess

class StartGame(Main):
    def __init__(self):
        super().__init__()
        self._error = False
        self._players = []
        self._game_name = "New game"

    @property
    def players(self):
        return self._players
    
    @players.setter
    def players(self, value):
        self._players = value

    @property
    def game_name(self):
        return self._game_name
    
    @game_name.setter
    def game_name(self, value):
        self._game_name = value

    def n_players(self):
        num_players = self.ensured_input(
            "How many are there?: \n",
            int
        )
        if num_players is not None:
            return num_players
        else:
            self._error = True
            self.exit_game(1)
    
    def select_players(self):
        pass

    def add_players(self, num_players): # REWORK, INCLUDE OPTION PLAYER ALREADY EXISTS BEFORE ADDING
        for i in range(num_players):
            player_name = self.ensured_input(
                f"Provide player {i+1} name:\n",
                str)
            if player_name is None:
                self._error = True
                self.exit_game(1)
            # If player exists, ask for another name
            while player_name in [player.name for player in self.players]:
                player_name = self.ensured_input(
                    "Player with this name already exists. Please provide another name:\n",
                    str)
                if player_name is None:
                    self._error = True
                    self.exit_game(1)
            player = Player(player_name)
            self.players.append(player)    
        return self.players



def start_game():
    """Starts the game, prompts for game_name, number of players and their names.\n
    Returns game-name and players"""
    game = StartGame()
    game_name = game.ensured_input("Provide game name, if you want to use default name - press enter:\n", str)
    if game_name != "":
        game.game_name = game_name
    game.check_error()
    print(f"Welcome to {game.game_name}!")
    time.sleep(2)
    num_players = game.n_players()
    game.check_error()
    print(f"Number of players is {num_players}")
    time.sleep(2)
    players = game.add_players(num_players)
    game.check_error()
    print("Players added successfully!")
    time.sleep(2)
    print("Players:")
    for player in players:
        print(player.name)
    time.sleep(2)
    game.save_game(game.game_name, players)
    return game.game_name, players


if __name__ == "__main__":
    
    print("Checking requirements...")
    result = subprocess.run([sys.executable, "requirements_install.py"], capture_output=True, text=True)
    if "True" in result.stdout:
        pass
    else:
        print("Error installing requirements")
        time.sleep(2)
        Main().exit_game(1)
    start_game()