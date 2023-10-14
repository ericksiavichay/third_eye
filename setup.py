import subprocess
import os

# Install packages from requirements.txt
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

for req in requirements:
    subprocess.check_call(["pip", "install", req])
