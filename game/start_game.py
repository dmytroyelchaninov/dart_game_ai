import time
import datetime
import sys
import logging
from main import Game, Player
import subprocess

# BASIC CLASS FOR GAME SETUP

class StartGame(Game):
    """
    A class to manage the start of the game, including player setup and game name.

    Methods:
    -------
    n_players():
        Prompts the user to input the number of players.
    
    select_players():
        Placeholder for player selection logic.
    
    add_players(num_players):
        Adds players to the game, ensuring no duplicate names.
    """
    def __init__(self):
        """
        Initializes the StartGame class, sets the initial game name and player list.
        """
        super().__init__()
        self._error = False
        self._players = []
        self._game_name = "New game"

    @property
    def players(self):
        """
        Gets the list of players.

        Returns:
        list: A list of player objects.
        """
        return self._players
    
    @players.setter
    def players(self, value):
        """
        Sets the list of players.

        Parameters:
        value (list): A list of player objects.
        """
        self._players = value

    @property
    def game_name(self):
        """
        Gets the game name.

        Returns:
        str: The name of the game.
        """
        return self._game_name
    
    @game_name.setter
    def game_name(self, value):
        """
        Sets the game name.

        Parameters:
        value (str): The name of the game.
        """
        self._game_name = value

    def n_players(self):
        """
        Prompts the user to input the number of players.

        Returns:
        int: The number of players.
        """
        num_players = self.ensured_input(
            "How many players are there?: \n",
            int
        )
        if num_players is not None:
            return num_players
        else:
            self._error = True
            self.exit_game(1)
    
    def select_players(self):
        """
        Placeholder for player selection logic.
        """
        pass

    def add_players(self, num_players):
        """
        Adds players to the game, ensuring no duplicate names.

        Parameters:
        num_players (int): The number of players to add.

        Returns:
        list: A list of player objects.
        """
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


if __name__ == "__main__":
    pass