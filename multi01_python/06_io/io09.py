import os


path = ".."
# path = r"c:\workspaces\multi01_python" 절대경로

for dir_name, sub_dir_names, file_names in os.walk(path):
    print(dir_name)
    for sub_dir_name in sub_dir_names:
        print(f"\t [d] {sub_dir_name}")
    for file_name in file_names:
        print(f"\t [f] {file_name}")

