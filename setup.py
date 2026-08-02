from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "engine_core",
        ["src/engine.cpp", "src/bridge.cpp"],
        include_dirs=["src"],
    ),
]

setup(
    name="visual_ai",
    version="0.2.0",
    author="Visual AI Game Engine Developers",
    description="High-performance Computer Vision AI tracking and physics library for game developers",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
    ],
)

