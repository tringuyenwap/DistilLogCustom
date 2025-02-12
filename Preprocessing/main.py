import subprocess

scripts = [
    "get_sequence.py",
    "mapping.py", 
    "modified.py",
    "split_data.py",
    "get_embedded_vector.py"
]

for script in scripts:
    try:
        print(f"Đang chạy {script}...")
        result = subprocess.run(["python3", script], check=True, text=True, capture_output=True)
        print(f"Kết quả của {script}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi chạy {script}:\n{e.stderr}")