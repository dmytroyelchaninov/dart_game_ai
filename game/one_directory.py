import os
import shutil

# SCRIPT TO COMPILE IMAGES FROM MULTIPLE DIRECTORIES INTO ONE

def compile_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                source_path = os.path.join(root, file)
                destination_path = os.path.join(output_dir, file)

                # To avoid overwriting files with the same name
                if os.path.exists(destination_path):
                    base_name, extension = os.path.splitext(file)
                    count = 1
                    while os.path.exists(destination_path):
                        destination_path = os.path.join(output_dir, f"{base_name}_{count}{extension}")
                        count += 1

                shutil.copy2(source_path, destination_path)
                print(f"Copied {source_path} to {destination_path}")

if __name__ == '__main__':
    input_dir = "../data/processed/angle_separate/"
    output_dir = "../data/processed/angle/"
    compile_images(input_dir, output_dir)