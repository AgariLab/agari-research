"""Exercise the actual TensorBoard CLI and its runtime dependencies."""

import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter


class TensorBoardTests(unittest.TestCase):
    def test_cli_inspects_a_real_scalar_event(self):
        with tempfile.TemporaryDirectory(prefix="agari-tensorboard-smoke-") as directory:
            writer = EventFileWriter(directory)
            try:
                writer.add_event(event_pb2.Event(
                    wall_time=time.time(), step=1,
                    summary=summary_pb2.Summary(value=[
                        summary_pb2.Summary.Value(tag="ui-smoke/value", simple_value=0.25),
                    ]),
                ))
            finally:
                writer.close()
            event_file, = Path(directory).glob("events.out.tfevents.*")
            result = subprocess.run(
                [sys.executable, "-m", "tensorboard.main", "--inspect",
                 "--event_file", str(event_file)],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("ui-smoke/value", result.stdout)
            self.assertIn("num_steps", result.stdout)


if __name__ == "__main__":
    unittest.main()
