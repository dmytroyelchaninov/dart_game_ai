import sys
import time
import subprocess
import logging
from main import Main, Player
from requirements_install import RequirementsInstallation
from start_game import StartGame 


def exec_install():
    def check_install(installation, reinstall=False):
        installation.check_requirements()
        if installation._complete:
            return True
        #do install
        installation.install_requirements(reinstall=reinstall)
        installation.check_error()
        installation.check_requirements()

    installation = RequirementsInstallation()
    check_install(installation)
    if installation._complete:
        return True
    else:
        check_install(installation, reinstall=True)
        if installation._complete:
            return True
        else:
            return False

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
    print("Game started!")
    return game.game_name, players


if __name__ == "__main__":
    # Installation of libaries
    print("Checking requirements...")
    result = exec_install()
    if result:
        pass
    else:
        print("Error installing requirements")
        time.sleep(2)
        Main().exit_game(1)

    # Start the game
    game_name, players = start_game()

    
    




