from time import sleep
from datetime import datetime
from sh import gphoto2 as gp
import signal, os, subprocess

os.chdir('/home/admin/Desktop/EDA')
def kill_gphoto2_process():
    p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
    out, err = p.communicate()
    for line in out.splitlines():
        if b'gvfsd-gphoto2' in line:
            pid = int(line.split(None,1)[0])
            os.kill(pid, signal.SIGKILL)
            print('KILLED')
clear_command = ["--folder"]
trigger_command = ["--trigger-capture"]

def get_name_of_image():
    files = gp("-L")
    file_lines = files.splitlines()

    # Initialize variables to track the maximum number and file name
    max_number = -1
    max_file_name = ""

    # Iterate through the lines to find the file with the largest number
    for line in file_lines:
        if line.startswith("#"):
            parts = line.split()
            print(parts)
            if len(parts) >= 2:
                file_name = parts[1]
                print(file_name)
                file_name_no_extension = file_name.split(".")[0]
                file_number_str = ''.join(filter(str.isdigit, file_name_no_extension))
                print(file_number_str)
                # Check if the file name has a numeric part
                if file_number_str:
                    file_number = int(file_number_str)
                    if file_number > max_number:
                        max_number = file_number
                        max_file_name = file_name

    # Print the file with the largest number
    print("File with the largest number:", max_file_name)
    return max_file_name

def get_latest_image(max_file_name):
    gp("--get-file=/store_00010001/DCIM/100EOS5D/"+max_file_name)

kill_gphoto2_process()
gp(trigger_command)
sleep(2)
max_file_name = get_name_of_image()
get_latest_image(max_file_name)