import os

# Function to recursively search for .mp4 files in a directory and its subdirectories
def find_mp4_files(directory):
    mp4_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".mp4"):
                mp4_files.append((os.path.join(root, file), os.path.basename(root)))
    return mp4_files

# Function to write the .csv file with video path and subfolder index
def write_csv(video_files, save_name):
    with open(f'{save_name}.csv', 'w', newline='') as csvfile:
        for video_file, subfolder in video_files:
            csvfile.write(f"{video_file} {1}\n")

# Main function to search for .mp4 files and write the .csv file
def main():
    main_directory = '/vast/eo41/data_video/SAY'  # Replace with your main directory path
    save_name = 'train'
    video_files = find_mp4_files(main_directory)
    write_csv(video_files, save_name)

if __name__ == '__main__':
    main()
