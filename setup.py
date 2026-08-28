"""Install visual_ai, building the C++ engine_core extension when possible.

The extension is an accelerator, not a requirement. `visual_ai/__init__.py`
imports `engine_core` and falls back to `PythonFallbackEngine` — an identical
API in pure Python — when that import fails, so every consumer works either
way. Making the build mandatory therefore turned a missing C++ toolchain into
a failed install for no functional gain, which is the first thing anyone
cloning this on a fresh machine hit.

So the build is attempted and allowed to fail. A failure prints why and what
it costs, then installs the pure-Python package. Set VISUAL_AI_REQUIRE_CPP=1
to make a build failure fatal instead, which is what CI wants.
"""

import os
import sys

from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext as _base_build_ext

try:
    from pybind11.setup_helpers import Pybind11Extension
    from pybind11.setup_helpers import build_ext as _pybind_build_ext
except ImportError:  # pybind11 absent: nothing to build from, so don't try.
    Pybind11Extension = None
    _pybind_build_ext = _base_build_ext

REQUIRE_CPP = os.environ.get("VISUAL_AI_REQUIRE_CPP", "").lower() in ("1", "true", "yes")


def _warn_skipped(reason):
    rule = "=" * 70
    sys.stderr.write(
        f"\n{rule}\n"
        "visual_ai: the C++ engine_core extension was NOT built.\n"
        f"{rule}\n"
        f"Reason:\n    {reason}\n\n"
        "This is not fatal. visual_ai falls back to PythonFallbackEngine,\n"
        "which has the same API and the same physics - only slower. Games\n"
        "run normally; check visual_ai.CPP_ENGINE_AVAILABLE to see which\n"
        "engine you got.\n\n"
        "To build it, install a C++ compiler and reinstall:\n"
        "    Windows  Visual Studio Build Tools (Desktop development with C++)\n"
        "    macOS    xcode-select --install\n"
        "    Linux    build-essential\n"
        "Set VISUAL_AI_REQUIRE_CPP=1 to make this an error instead.\n"
        f"{rule}\n\n"
    )


class build_ext(_pybind_build_ext):
    """Build the extension, but treat a build failure as a downgrade."""

    def run(self):
        try:
            super().run()
        except Exception as exc:
            if REQUIRE_CPP:
                raise
            _warn_skipped(f"{type(exc).__name__}: {exc}")
            self.extensions = []
        self._evict_root_shadow()

    def _evict_root_shadow(self):
        """Delete any engine_core .pyd/.so sitting at the repo root.

        The built extension belongs in src/, next to the visual_ai package
        (package_dir maps there). An older build once dropped one at the repo
        root instead, and because sys.path[0] is the script directory, that
        stale copy silently shadowed every fresh build in src/ for any plain
        interpreter run from this directory - the same trap as the root-level
        visual_ai/ duplicate pytest.ini warns about. gitignore's *.pyd kept it
        out of git status, so nothing ever surfaced it. Evict on every build
        so the trap cannot re-arm; tests/test_no_root_shadow.py fails the
        suite if one appears between builds.
        """
        import glob

        root = os.path.dirname(os.path.abspath(__file__))
        for pattern in ("engine_core*.pyd", "engine_core*.so"):
            for path in glob.glob(os.path.join(root, pattern)):
                os.remove(path)
                sys.stderr.write(
                    f"visual_ai: removed stale root-level {os.path.basename(path)} - it would have "
                    "shadowed the build in src/.\n")

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:
            if REQUIRE_CPP:
                raise
            _warn_skipped(f"{type(exc).__name__}: {exc}")


if Pybind11Extension is not None:
    ext_modules = [
        Pybind11Extension(
            "engine_core",
            ["src/engine.cpp", "src/bridge.cpp"],
            include_dirs=["src"],
        ),
    ]
else:
    ext_modules = []
    _warn_skipped("pybind11 is not installed, so there is nothing to build with.")

setup(
    name="visual_ai",
    version="0.3.0",
    author="Visual AI Game Engine Developers",
    description="High-performance Computer Vision AI tracking and physics "
                "library for game developers",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    # Keep in step with pyproject.toml: 3.10 floor (PEP-604 signatures
    # evaluated at def time), 3.12 ceiling (mediapipe wheels).
    python_requires=">=3.10,<3.13",
    install_requires=[
        "numpy>=1.20.0",
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.14,<=0.10.35",
        # Kept in step with pyproject.toml; see the note there for why this is
        # a hard dependency rather than an optional one.
        "pygame>=2.5.0",
    ],
)
