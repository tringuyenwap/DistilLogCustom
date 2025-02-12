import subprocess


scripts = ["train.py","testteacher.py", "teach.py", "test.py","prune.py"]


for script in scripts:
    try:
        print(f"Run {script}...")
        result = subprocess.run(["python3", script], check=True, text=True, capture_output=True)
        print(f"Result of {script}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error when run {script}:\n{e.stderr}")
