import subprocess
import sys
import pkg_resources
import time
import logging
from main import Game

# This script was just to practice language, nothing crucial is here, just a simple script to install requirements


class RequirementsInstallation(Game):
    """
    A class to handle the installation of required packages.
    
    Methods:
    -------
    user_confirmation_install(reinstall=False):
        Prompts the user to confirm whether to install or reinstall the packages.
    
    check_requirements():
        Checks if all required packages are installed.
    
    install_requirements(reinstall=False):
        Installs or reinstalls the required packages.
    """
    def __init__(self):
        """
        Initializes the RequirementsInstallation class, sets the installation completion status,
        and configures logging.
        """
        super().__init__()
        self._complete = False
        logging.basicConfig(filename='installation.log', level=logging.ERROR)

    def user_confirmation_install(self, reinstall=False):
        """
        Prompts the user to confirm whether to install or reinstall the packages.
        
        Parameters:
        reinstall (bool): If True, prompts the user to confirm reinstallation.
        
        Returns:
        bool: True if the user confirms installation, False otherwise.
        """
        if reinstall:
            confirm_msg = "Would you like to try to reinstall? (y/n): "
        else:
            confirm_msg = "Would you like to install the missing packages? (y/n): "
        confirm = self.ensured_input(confirm_msg, y_or_n=True)
        if confirm.lower() == 'y':
            return True
        elif confirm.lower() == 'n':
            return False

    def check_requirements(self):
        """
        Checks if all required packages are installed.
        
        Returns:
        bool: True if all packages are installed, False if some packages are missing.
        """
        print("Checking requirements...")
        with open('./txt/requirements.txt', 'r') as f:
            try:
                requirements = f.read().splitlines()
                pkg_resources.require(requirements)
                print('All packages are installed!')
                self._complete = True
                time.sleep(0.5)
                return True
            except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict) as e:
                print('Some packages are missing...')
                print(f"Missing packages: {e}")
                time.sleep(2)
                return False
            except Exception as e:
                self._error = True
                self.exit_game(1)
                logging.error(f"An error occurred while checking requirements: {e}")

    def install_requirements(self, reinstall=False):
        """
        Installs or reinstalls the required packages.
        
        Parameters:
        reinstall (bool): If True, reinstalls the packages.
        
        Returns:
        bool: True if the installation is successful, exits the game otherwise.
        """
        try:
            if self.user_confirmation_install(reinstall=reinstall):
                print("Installing libraries...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "./txt/requirements.txt"])
                print("Libraries installed!")
                time.sleep(2)
                return True
            else:
                self.exit_game(0)
        except Exception as e:
            self._error = True
            logging.error(f"An error occurred while installing requirements: {e}")
            self.exit_game(1)


if __name__ == "__main__":
    pass

# This exact row and life substitutions caused creation of a game logic
# While building model that recognizes picture
# 'opencv_python==4.10.0, numpy==1.26.4'.split('==')[0].strip()