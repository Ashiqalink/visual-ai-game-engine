"""
Guard against import-shadowing copies at the repo root.

The real package is src/visual_ai and the real extension is src/engine_core*.
Python puts the script directory (or cwd) at sys.path[0], so a copy of either
sitting at the repo root wins the import race for any plain interpreter run
from this directory - and .gitignore hides binaries from git status, so a
stale one can sit there for weeks masking fresh builds.

This has happened twice: a root-level visual_ai/ duplicate once silently
failed 32 tests (see pytest.ini's note), and a root-level engine_core .pyd
from an older build (2026-08-08) shadowed the freshly built src/ extension
during the E3 binding work. setup.py now deletes root-level engine_core
binaries after every build; this test makes a reappearance fail the suite
between builds too, instead of surfacing as behaviour that mysteriously
does not match the source.
"""

import glob
import os
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestNoRootShadow(unittest.TestCase):
    def test_no_engine_core_binary_at_the_repo_root(self):
        strays = [
            os.path.basename(path)
            for pattern in ("engine_core*.pyd", "engine_core*.so")
            for path in glob.glob(os.path.join(REPO_ROOT, pattern))
        ]
        self.assertEqual(
            strays, [],
            "engine_core binaries at the repo root shadow src/ for any plain "
            "interpreter run from this directory. Delete them (a fresh build "
            "via setup.py does this automatically); the real extension lives "
            "in src/.",
        )

    def test_no_visual_ai_package_at_the_repo_root(self):
        stray = os.path.join(REPO_ROOT, "visual_ai")
        self.assertFalse(
            os.path.isdir(stray),
            "A root-level visual_ai/ shadows src/visual_ai and once silently "
            "failed 32 tests - see pytest.ini. Delete it; the real package "
            "lives in src/visual_ai.",
        )

    def test_the_imported_modules_come_from_src(self):
        """The modules this very test run resolved must be the src/ ones."""
        import visual_ai

        self.assertEqual(
            os.path.dirname(os.path.abspath(visual_ai.__file__)),
            os.path.join(REPO_ROOT, "src", "visual_ai"),
        )
        try:
            import engine_core
        except ImportError:
            return  # no compiled core on this machine - nothing to shadow
        self.assertEqual(
            os.path.dirname(os.path.abspath(engine_core.__file__)),
            os.path.join(REPO_ROOT, "src"),
        )


if __name__ == "__main__":
    unittest.main()
