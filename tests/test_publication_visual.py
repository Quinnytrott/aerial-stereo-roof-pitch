import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "optional Matplotlib not installed")
class PublicationVisualTests(unittest.TestCase):
    def test_renderer_creates_public_synthetic_png_with_safe_metadata(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "visual.png"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "render_synthetic_example.py"),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("public synthetic scene", completed.stdout)
            with Image.open(output) as image:
                self.assertEqual(image.size, (1960, 1120))
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.info["Title"], "Synthetic stereo roof-plane reconstruction")
                metadata = " ".join(str(value) for value in image.info.values())
                self.assertNotIn("/Users/", metadata)
                self.assertNotIn("Ontario", metadata)
                extrema = image.convert("RGB").getextrema()
                self.assertTrue(all(low < high for low, high in extrema))

    def test_renderer_refuses_existing_output_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "visual.png"
            output.write_bytes(b"preserve-me")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "render_synthetic_example.py"),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(output.read_bytes(), b"preserve-me")
            self.assertTrue(completed.stderr.startswith("REFUSED:"))


if __name__ == "__main__":
    unittest.main()
