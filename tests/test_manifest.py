import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from agari_research.manifest import model_manifest


class ManifestTests(unittest.TestCase):
    def test_provenance_tracks_actual_artifact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights, report = Path(tmp) / "fake-test-weights", Path(tmp) / "report.json"
            weights.write_bytes(b"not-a-trained-model")
            report.write_text(json.dumps({"games": 4}))
            kwargs = dict(weights=weights, code_revision="a" * 40, architecture="test-only",
                          encoding_version="test-only-v1", dataset_sha256="b" * 64,
                          evaluation_report=report, upstream_revision="c" * 40)
            manifest = model_manifest(**kwargs)
            self.assertEqual(manifest["weights"]["sha256"], hashlib.sha256(weights.read_bytes()).hexdigest())
            self.assertNotIn(tmp, json.dumps(manifest))
            weights.write_bytes(b"changed")
            self.assertNotEqual(manifest["weights"]["sha256"], model_manifest(**kwargs)["weights"]["sha256"])
            with self.assertRaises(ValueError):
                model_manifest(**dict(kwargs, code_revision="main"))


if __name__ == "__main__":
    unittest.main()
