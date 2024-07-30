# remove_file_references.py

import re

# 경로를 설정합니다.
requirements_path = 'requirements.txt'
temp_path = 'requirements_clean.txt'

def clean_requirements():
    with open(requirements_path, 'r') as file:
        lines = file.readlines()

    with open(temp_path, 'w') as file:
        for line in lines:
            # @ file로 시작하는 줄을 제외합니다.
            if not re.search(r'@ file://', line):
                file.write(line)

    print(f'Cleaned requirements saved to {temp_path}')

if __name__ == "__main__":
    clean_requirements()
