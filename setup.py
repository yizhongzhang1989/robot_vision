from setuptools import setup, find_packages

setup(
    name="robot_vision",
    version="1.0.0",
    author="msraig",
    description="Simple keypoint tracking for robotics",
    packages=find_packages() + ['core'],
    package_dir={'core': 'core'},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.3.0", 
        "Pillow>=8.0.0",
        "requests>=2.25.0"
    ],
    entry_points={
        "console_scripts": [
            "robot-vision-tracker=core.keypoint_tracker:main",
        ],
    },
)
