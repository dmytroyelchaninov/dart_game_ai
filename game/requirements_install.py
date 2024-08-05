import subprocess
import sys
import pkg_resources
import time
import logging
from main import Main

# This script was just to practice language, nothing crucial is here, just a simple script to install requirements
class RequirementsInstallation(Main):
    def __init__(self):
        super().__init__()
        self._complete = False
        logging.basicConfig(filename='installation.log', level=logging.ERROR)

    def user_confirmation_install(self, reinstall=False):
        if reinstall:
            confirm_msg = "Would you like to try to reinstall? (y/n): "
        else:
            confirm_msg = "Would you like to install the missing packages? (y/n): "
        confirm = self.ensured_input(confirm_msg, y_or_n=True)
        if confirm.lower() == 'y':
            return True
        elif confirm.lower() == 'n':
            return False
        
        # This will not be reached, actually
        # else:
        #     self._error = True
        #     self.exit_game(1)
        #     logging.error(f"An error occured while confirming installation")
        #     return None

    def check_requirements(self):
        # How huge is open() function, it serves like a factory function!
        print("Checking requirements...")
        with open('./txt/requirements.txt', 'r') as f:
            try:
                requirements = f.read().splitlines()
                pkg_resources.require(requirements)
                print('All packages are installed!')
                self._complete = True
                time.sleep(0.5)
                return True
            except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
                print('Some packages are missing...')
                print(f"Missing packages: {pkg_resources.DistributionNotFound, pkg_resources.VersionConflict}")
                time.sleep(2)
                return False
            except Exception as e:
                self._error = True
                self.exit_game(1)
                logging.error(f"An error occured while checking requirements: {e}")


    def install_requirements(self, reinstall=False): 
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
            logging.error(f"An error occured while installing requirements: {e}")
            self.exit_game(1)

def check_install(installation, reinstall=False):
        installation.check_requirements()
        if installation._complete:
            return True
        #do install
        installation.install_requirements(reinstall=reinstall)
        installation.check_error()
        installation.check_requirements()

def exec_install():
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
    
    
if __name__ == "__main__":
    result = exec_install()
    print(result)