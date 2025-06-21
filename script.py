# import os

# file_loader = os.listdir("./checkpoints/")
# for file in file_loader:
#     if "50000000" in file:
        
#         path = f"./checkpoints/{file}"
#         newpath = path.replace("_50000000","")
#         print(f"rename {path} to {newpath}")
#         os.rename(path, newpath)

import subprocess

def run_python_script(script_path, args):
    try:
        # Run the Python script
        command = ['python', script_path] + args
        print(command)
        process = subprocess.Popen(command)
        process.wait()
        process.kill()
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    else:
        print("Script execution completed successfully.")

if __name__ == "__main__":
    # Replace 'path/to/your/script.py' with the actual path to your Python script
    envs = ['Bowling', 'Boxing', 'Breakout', 'Centipede', 'ChopperCommand']
    # envs = ['Hero', 'IceHockey', 'Kangaroo', 'Krull', 'KungFuMaster']
    for env in envs:
        script_path = "train.py"
        arg = ["--env_name", "gym:"+env] 
        run_python_script(script_path, arg)
    