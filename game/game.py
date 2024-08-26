import sys
import time
import os
import time
import subprocess
import logging
import numpy as np
import pandas as pd
from ultralytics import YOLO
import cv2 as cv
from main import Game, Player, Board
from requirements_install import RequirementsInstallation
from start_game import StartGame 
from final_transform import initial_prepare, transform, BoardTransform, PerspectiveTransform
from yolo import yolo_crop, yolo_predict
import traceback
import pandas as pd


class Scores():
    def __init__(self, board):
        self.board = board
        self.final_score = 0
        self._scores = []
        self.deltas = []

    def define_zones(self, show=True):
        img = self.board._img
        outer_ellipse = self.board._outer_ellipse
        inner_ellipse = self.board._inner_ellipse

        self.triple_outer = inner_ellipse
        self.double_outer = outer_ellipse
        self.triple_inner = (inner_ellipse[0], 
                        (inner_ellipse[1][0] * (97/107), inner_ellipse[1][1] * (97/107)), 
                        inner_ellipse[2])
        self.double_inner = (outer_ellipse[0], 
                        (outer_ellipse[1][0] * (159.5/170), outer_ellipse[1][1] * (159.5/170)), 
                        outer_ellipse[2])
        self.bulls_eye_inner_radius = int((outer_ellipse[1][0]/170)*3.6)
        self.bulls_eye_outer_radius = int((outer_ellipse[1][0]/170)*8.2)

        self.triple_inner_radius = int((self.triple_inner[1][0] + self.triple_inner[1][1]) / 4)
        self.triple_outer_radius = int((self.triple_outer[1][0] + self.triple_outer[1][1]) / 4)
        self.double_inner_radius = int((self.double_inner[1][0] + self.double_inner[1][1]) / 4)
        self.double_outer_radius = int((self.double_outer[1][0] + self.double_outer[1][1]) / 4)

        if show:
            img_copy = img.copy()
            self._draw_zones(img, self.board._center)

            cv.imshow("Dartboard Zones", img)
            cv.waitKey(0)
            is_good = input("Are the zones defined correctly? Pay attention to center! (y/n): ")
            cv.destroyAllWindows()

            if is_good.lower() == 'n':
                print("Relocating center...")
                alt_center = (self.board._img.shape[1]//2, self.board._img.shape[0]//2)
                self._draw_zones(img_copy, alt_center)
                cv.imshow("New center", img_copy)
                cv.waitKey(0)
                is_good = input("Is center now better defined? (y/n): ")
                cv.destroyAllWindows()
                if is_good.lower() == 'y':
                    self.board._center = alt_center

        self._center = self.board._center

    def _draw_zones(self, img, center):
        c = (68, 59, 196)
        cv.circle(img, center, self.triple_inner_radius, c, 2)
        cv.circle(img, center, self.triple_outer_radius, c, 2)
        cv.circle(img, center, self.double_inner_radius, c, 2)
        cv.circle(img, center, self.double_outer_radius, c, 2)
        cv.circle(img, center, 2, (255,0,0), 2)
        cv.circle(img, center, self.bulls_eye_inner_radius, c, 2)
        cv.circle(img, center, self.bulls_eye_outer_radius, c, 2)

    def calculate_distance_and_angle(self, prediction):
        distance = np.sqrt((prediction[0] - self._center[0]) ** 2 + (prediction[1] - self._center[1]) ** 2)
        angle = np.degrees(np.arctan2(prediction[1] - self._center[1], prediction[0] - self._center[0]))
        angle = angle if angle >= 0 else angle + 360
        return distance, angle

    def determine_sector(self, angle, shift_angle):
        angle_per_sector = 360 / 20
        adjusted_angle = (angle + 9 + shift_angle) % 360
        sector_index = int(adjusted_angle // angle_per_sector)
        sector_scores = [6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5, 20, 1, 18, 4, 13]
        sector = sector_scores[sector_index]
        return sector

    def check_bulls_eye(self, distance):
        if distance <= self.bulls_eye_inner_radius:
            return 50
        elif self.bulls_eye_inner_radius <= distance <= self.bulls_eye_outer_radius:
            return 25
        else:
            return None

    def check_triple(self, distance):
        return self.triple_inner_radius <= distance <= self.triple_outer_radius

    def check_double(self, distance):
        deltas = [abs(distance - self.double_inner_radius), abs(self.double_outer_radius - distance)]
        if min(deltas) < 1:
            return True 
        return self.double_inner_radius <= distance <= self.double_outer_radius

    def get_scores(self, predictions, shift_angle=0):
        scores = []
        for prediction in predictions:
            distance, angle = self.calculate_distance_and_angle(prediction)
            if distance > self.double_outer_radius + 1:
                score = 0
            else:
                score = self.check_bulls_eye(distance)
                if score is None:
                    sector = self.determine_sector(angle, shift_angle=shift_angle)
                    if self.check_triple(distance):
                        score = sector * 3
                    elif self.check_double(distance):
                        score = sector * 2
                    else:
                        score = sector
            scores.append(score)
        self._scores = scores
        self.final_score += sum(scores)
        return scores

    def draw_distances_and_angles(self, predictions, show=False):
        img = self.board._img.copy()
        self._draw_zones(img, self._center)
        
        for prediction, score in zip(predictions, self._scores):
            distance, angle = self.calculate_distance_and_angle(prediction)
            cv.line(img, self._center, (int(prediction[0]), int(prediction[1])), (40, 215, 56), 2)

            text = f'{score}'
            cv.putText(img, text, (int(prediction[0]) + 15, int(prediction[1]) - 15), 
                    cv.FONT_HERSHEY_SIMPLEX, 1.5, (56, 40, 215), 2, cv.LINE_AA)
        if show:
            cv.imshow("Distances and Angles", img)
            cv.waitKey(0)
            cv.waitKey(1)
            time.sleep(0.1)
            cv.destroyAllWindows()
            cv.waitKey(1)

        return img

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
    game_name = game.ensured_input("Provide game name:\n", str)
    if game_name != "":
        game.game_name = game_name
    game.check_error()
    print(f"Welcome to {game.game_name}!")
    time.sleep(1)
    num_players = game.n_players()
    game.check_error()
    # print(f"Number of players is {num_players}")
    # time.sleep(1)
    players = game.add_players(num_players)
    game.check_error()
    print("Players added successfully!")
    time.sleep(1)
    # print("Players:")
    # for player in players:
    #     print(player.name)
    time.sleep(1)
    game.save_game(game.game_name, players)
    print("Let's get started!")
    time.sleep(1)
    return game.game_name, players

def crop_around(board, predictions, square_size=50):
    img = board._img
    mask = np.zeros_like(img)

    padding = square_size // 2
    
    for prediction in predictions:
        x, y = prediction
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + padding)
        y2 = min(img.shape[0], y + padding)
        
        # Copy the area around the prediction point to the mask
        mask[y1:y2, x1:x2] = img[y1:y2, x1:x2]
        
    return mask

def wait_for_image(interval=1):
    downloads_dir = os.path.expanduser('~/Downloads')
    print(f"Airdrop picture!")
    already_existing_files = set(os.listdir(downloads_dir))

    while True:
        current_files = set(os.listdir(downloads_dir))
        new_files = current_files - already_existing_files

        image_files = [file for file in new_files if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        
        if image_files:
            image_path = os.path.join(downloads_dir, image_files[0])
            print(f"I got it...")

            return image_path
        
        time.sleep(interval)

def archive(image, image_path):
    if not os.path.exists('../ARCHIVE'):
        os.makedirs('../ARCHIVE')
    cv.imwrite('../ARCHIVE/' + os.path.basename(image_path), image)
    os.remove(image_path)
    # print("Image stored in /ARCHIVE folder!")

def predict(board, part1, part2):
    padding = 20
    img = board._img
    cropped_img, n_classes_initial = yolo_crop(img, part1, padding=padding)
    predictions, n_classes_predictions = yolo_predict(cropped_img, part2)
    if n_classes_predictions>min(n_classes_initial) or n_classes_predictions>3:
        while n_classes_predictions>min(n_classes_initial) or n_classes_predictions>3:
            cropped_img, n_classes_initial = yolo_crop(img, part1, padding=padding)
            predictions, n_classes_predictions = yolo_predict(cropped_img, part2)
            padding -= 2
        # print(f"padding: {padding}")
    elif n_classes_predictions<min(n_classes_initial):
        while n_classes_predictions<min(n_classes_initial):
            cropped_img, n_classes_initial = yolo_crop(img, part1, padding=padding)
            predictions, n_classes_predictions = yolo_predict(cropped_img, part2)
            padding += 2
        # print(f"padding: {padding}")
    if padding > 50 or padding < 4:
        print("Take picture from another angle")
        sys.exit(1)

    return predictions

def process_image(size=(1000, 1000), accuracy=0.05, iterations=5, show=False):
    
    img_path = wait_for_image()
    # img_path = '../UPLOAD/12.jpg'
    
    original = Board(img_path)
    board = BoardTransform(original)
    size = (1000, 1000)
    board = initial_prepare(board, size=size, crop_scale=1.3)
    
    # Models
    part1 = YOLO('./yolos/part1.pt')
    part2 = YOLO('./yolos/part2.pt')
    predictions = predict(board, part1, part2)

    transformed, processed_img, predictions, center20 = transform(img_path, predictions, size_transform=size, accuracy=accuracy, iterations=iterations)
    angle = np.degrees(np.arctan2(center20[1] - transformed._center[1], center20[0] - transformed._center[0]))
    angle = angle if angle >= 0 else angle + 360
    shift_angle = 270 - angle

    if show:
        print('Check this out!')
        cv.imshow('Dartboard', processed_img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # archive(processed_img, img_path)

    return transformed, predictions, img_path, shift_angle


def process_image_lib(img_path, size=(1000, 1000), accuracy=0.05, iterations=5, show=False):
    original = Board(img_path)
    board = BoardTransform(original)
    board = initial_prepare(board, size=size, crop_scale=1.3)
    
    # Models
    part1 = YOLO('./yolos/part1.pt')
    part2 = YOLO('./yolos/best.pt')
    predictions = predict(board, part1, part2)

    transformed, processed_img, predictions = transform(img_path, predictions, size_transform=size, accuracy=accuracy, iterations=iterations)
    
    if show:
        print('Check this out!')
        cv.imshow('Dartboard', processed_img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    archive(processed_img, img_path)

    return transformed, predictions, img_path

# This needed to aggregate stats on execution time of methods and overall accuracy
def process_image_library(directory, size=(1000, 1000), accuracy=0.05, iterations=5, show=False):
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    if not image_files:
        print(f"No images found in the directory: {directory}")
        return

    for image_file in image_files:
        img_path = os.path.join(directory, image_file)
        print(f"Processing {img_path}...")
        try:
            # Process each image
            board, predictions, path = process_image_lib(img_path=img_path, size=size, accuracy=accuracy, iterations=iterations, show=show)
            scores = Scores(board)
            scores.define_zones(show=show)
            scores.get_scores(predictions)
            result_img = scores.draw_distances_and_angles(predictions)
            score = scores._scores
            round_score = scores.final_score
            print(f"Processed {img_path}: Nice throw! You hit: {','.join(map(str, score))}")

            # Archive the processed image
            archive_dir = '../ARCHIVE'
            if not os.path.exists(archive_dir):
                os.makedirs(archive_dir)
            archived_image_path = os.path.join(archive_dir, os.path.basename(img_path))
            cv.imwrite(archived_image_path, result_img)
            print(f"Image archived to {archived_image_path}")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            traceback.print_exc()
            continue


if __name__ == "__main__":

    # board, predictions, path, shift_angle = process_image(size=(1000, 1000), accuracy=0.05, iterations=5, show=True)
    # scores = Scores(board)
    # scores.define_zones(show=False)
    # scores.get_scores(predictions, shift_angle=shift_angle)
    # img = scores.draw_distances_and_angles(predictions)
    # score = scores._scores
    # round_score = scores.final_score
    # print(f"Nice throw! You hit: {','.join(map(str, score))}")
    # result_img = scores.draw_distances_and_angles(predictions, show=True)
    # # archive(result_img, path)


    # INSTALL REQUIREMENTS
    print("Checking requirements...")
    time.sleep(1)
    # result = exec_install()
    # if result:
    #   print("Requirements check complete!")
    # else:
    #     print("Error installing requirements")
    #     time.sleep(2)
    #     Game().exit_game(1)
    print("Requirements check complete!")

    # START GAME
    game_name, players = start_game()
    is_win = False
    round_num = 1
    while is_win == False:
        print(f"\nRound {round_num}!")
        for player in players:
            print(f"\n{player.name} GET READY!")
            loaded = False
            while loaded == False:
                try:
                    # IMAGE RECOGNITION
                    board, predictions, path, shift_angle = process_image(size=(1000, 1000), accuracy=0.05, iterations=5, show=False)
                    scores = Scores(board)
                    loaded = True
                except ValueError:
                    print("Error processing image. Airdrop it again!")
                    board, predictions, path, shift_angle = process_image(size=(1000, 1000), accuracy=0.05, iterations=5, show=False)
                    scores = Scores(board)
                    loaded = True

            scores.define_zones(show=False)
            scores.get_scores(predictions, shift_angle=shift_angle)
            score = scores._scores
            print(f"Nice throw! You hit: {','.join(map(str, score))}")

            show_board = True
            if show_board:
                print(f"Check the dartboard!")           
            result_img = scores.draw_distances_and_angles(predictions, show=show_board)
            archive(result_img, path)

            is_true = input("Is this correct? (y/n): ")
            sys.stdout.flush() 
            if is_true.lower() == 'n':
                scores.final_score -= sum(score)
                sc = input("Enter the correct score: ")
                sys.stdout.flush() 
                score = list(map(int, sc.split(',')))
                scores._scores = score 
                scores.final_score += sum(score) 
                time.sleep(1)

            # SCORES
            round_score = scores.final_score
            print(f"{player.name}'s round score: {round_score}")
            player._scores.append(round_score)
            total_score = sum(player._scores)
            print(f"{player.name} total score: {total_score}")
            time.sleep(1)

            if total_score == 301:
                is_win = True
                print(f"{player.name} wins!")
                time.sleep(2)
                player._crowns += 1
                print(f"{player.name} has {player._crowns} crown(s)!")
                time.sleep(2)
                print("Total number of rounds played: ", round_num )
                break

            elif total_score > 301:
                player._scores.pop()
                print(f"{player.name} that's too much points!")
                time.sleep(2)
                continue
            else:
                print(f"{player.name} has {301 - total_score} points left!")
                time.sleep(2)
                continue
        players = [players[-1]] + players[:-1]
        round_num += 1


    # image_directory = "../pic"
    # process_image_library(image_directory)