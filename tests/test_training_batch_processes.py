"""Owned-process cancellation with tiny sleep processes; no training or artifacts."""
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import psutil


spec = importlib.util.spec_from_file_location("batch_process_test", Path(__file__).parents[1] / "scripts/phase2_training_batch.py")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)

CHILD = """
import pathlib, signal, sys, time
root = pathlib.Path(sys.argv[1])
if sys.argv[2] == 'stubborn':
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
print('original-child-stdout', flush=True)
print('original-child-stderr', file=sys.stderr, flush=True)
(root / 'child-ready').write_text('ready')
while True:
    time.sleep(.01)
"""

RUNNER = """
import json, os, pathlib, signal, subprocess, sys, time
root, mode = pathlib.Path(sys.argv[1]), sys.argv[2]
with (root / 'child.stdout').open('xb') as out, (root / 'child.stderr').open('xb') as err:
    child = subprocess.Popen([sys.executable, '-B', '-u', str(root / 'child.py'), str(root), mode],
                             stdout=out, stderr=err, start_new_session=True)
    (root / 'child-pid').write_text(str(child.pid))
    while not (root / 'child-ready').exists():
        time.sleep(.01)
    (root / 'runner-ready').write_text('ready')
    if mode == 'orphan':
        while not (root / 'exit-runner').exists():
            time.sleep(.01)
        os._exit(0)
    interrupted = False
    while True:
        try:
            code = child.wait()
            break
        except KeyboardInterrupt:
            interrupted = True
            try:
                os.killpg(child.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
    (root / 'finished.json').write_text(json.dumps({'exit_code': code, 'interrupted': interrupted}))
"""


class ProcessIntegrationTests(unittest.TestCase):
    def setUp(self):
        try:
            psutil.pids()
        except PermissionError:
            self.skipTest("Process-tree inspection requires host process-table permission")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def spawn(self, mode):
        (self.root / "child.py").write_text(CHILD)
        (self.root / "runner.py").write_text(RUNNER)
        stdout, stderr = (self.root / "runner.stdout").open("xb"), (self.root / "runner.stderr").open("xb")
        process = subprocess.Popen([sys.executable, "-B", "-u", str(self.root / "runner.py"), str(self.root), mode],
                                   stdout=stdout, stderr=stderr, start_new_session=True)
        active = {"fixture": {"process": process}}
        batch.register_owned(active)
        def cleanup():
            try:
                batch.interrupt_owned(active, grace_seconds=.05, terminate_seconds=.1, kill_seconds=2)
            finally:
                stdout.close()
                stderr.close()
        self.addCleanup(cleanup)
        deadline = time.monotonic() + 5
        while not (self.root / "runner-ready").exists():
            batch.discover_owned(active)
            self.assertLess(time.monotonic(), deadline, "sleep fixture did not start")
            self.assertIsNone(process.poll(), "sleep fixture exited before setup")
            time.sleep(.01)
        batch.discover_owned(active)
        child = int((self.root / "child-pid").read_text())
        self.assertIn(child, {key[0] for key in batch.owned_processes(active)})
        return active, process, child

    def test_sigint_delegation_preserves_original_runner_receipt_and_logs(self):
        active, process, child = self.spawn("normal")
        result = batch.interrupt_owned(active, grace_seconds=2, terminate_seconds=.2, kill_seconds=1)
        self.assertEqual(result["status"], "cleaned")
        self.assertEqual(process.returncode, 0)
        self.assertFalse(result["forced_termination"])
        self.assertFalse(result["orphaned_descendants"])
        self.assertTrue(json.loads((self.root / "finished.json").read_text())["interrupted"])
        self.assertIn("original-child-stdout", (self.root / "child.stdout").read_text())
        self.assertIn("original-child-stderr", (self.root / "child.stderr").read_text())
        self.assertFalse(any(batch._owned_alive(p) for p in batch.owned_processes(active).values()))

    def test_stubborn_new_session_child_is_killed_before_runner_can_seal(self):
        active, process, child = self.spawn("stubborn")
        result = batch.interrupt_owned(active, grace_seconds=.05, terminate_seconds=.1, kill_seconds=1)
        killed = [row["pid"] for row in result["signals"] if row["signal"] == signal.SIGKILL]
        self.assertEqual(killed, [child])
        self.assertEqual(process.returncode, 0)
        self.assertEqual(json.loads((self.root / "finished.json").read_text())["exit_code"], -signal.SIGKILL)
        self.assertTrue(result["forced_termination"])

    def test_saved_child_handle_survives_runner_exit_and_reparenting(self):
        active, process, child = self.spawn("orphan")
        original = next(p for key, p in batch.owned_processes(active).items() if key[0] == child)
        (self.root / "exit-runner").write_text("exit")
        self.assertEqual(process.wait(timeout=2), 0)
        self.assertIs(next(p for key, p in batch.discover_owned(active).items() if key[0] == child), original)
        result = batch.interrupt_owned(active, grace_seconds=.05, terminate_seconds=.2, kill_seconds=1)
        self.assertEqual([row["pid"] for row in result["orphaned_descendants"]], [child])
        self.assertTrue(all(row["pid"] == child for row in result["signals"]))
        self.assertFalse(batch._owned_alive(original))
        self.assertFalse((self.root / "finished.json").exists())

    def test_pre_spawn_runner_is_not_resumed_into_an_untracked_child(self):
        process = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(30)"], start_new_session=True)
        active = {"fixture": {"process": process}}
        batch.register_owned(active)
        self.addCleanup(lambda: batch.interrupt_owned(active, grace_seconds=.05, terminate_seconds=.1, kill_seconds=1))
        result = batch.interrupt_owned(active, grace_seconds=.05, terminate_seconds=.1, kill_seconds=1)
        self.assertEqual(process.returncode, -signal.SIGKILL)
        self.assertEqual([row["pid"] for row in result["forced_before_child"]], [process.pid])
        self.assertTrue(result["original_finished_may_be_missing"])
        self.assertNotIn(signal.SIGCONT, [row["signal"] for row in result["signals"]])


class ProcessIdentityTests(unittest.TestCase):
    def test_reused_identity_is_never_signalled_or_recreated_from_bare_pid(self):
        root = Mock()
        root.pid = 456789
        root.create_time.return_value = 1.
        root.is_running.return_value = False  # psutil reports PID reuse as not running.
        process = Mock(pid=root.pid, returncode=0)
        process.poll.return_value = 0
        tracker = batch._OwnedProcessTracker.__new__(batch._OwnedProcessTracker)
        tracker.process, tracker.root, tracker.root_key = process, root, (root.pid, 1.)
        tracker.owned = {tracker.root_key: root}
        active = {"fixture": {"process": process, "tracker": tracker}}
        with patch.object(psutil, "Process", side_effect=AssertionError("must retain original Process object")), \
                patch.object(psutil, "wait_procs", side_effect=AssertionError("must not reap a Popen root")):
            result = batch.interrupt_owned(active, grace_seconds=0, terminate_seconds=0, kill_seconds=0)
        root.send_signal.assert_not_called()
        root.children.assert_not_called()
        process.wait.assert_called_once_with(timeout=0)
        self.assertEqual(result["signals"], [])

    def test_discovery_keeps_old_descendants_when_parent_disappears(self):
        root, child = Mock(), Mock()
        for proc, pid, created in ((root, 101, 1.), (child, 102, 2.)):
            proc.pid = pid
            proc.create_time.return_value = created
            proc.is_running.return_value = True
            proc.status.return_value = psutil.STATUS_RUNNING
            proc.children.return_value = []
        root.children.return_value = [child]
        popen = Mock(pid=101)
        active = {"fixture": {"process": popen}}
        with patch.object(psutil, "Process", return_value=root) as constructor:
            batch.register_owned(active)
            first = batch.discover_owned(active)
            root.is_running.return_value = False
            second = batch.discover_owned(active)
        constructor.assert_called_once_with(101)
        self.assertIs(first[(102, 2.)], second[(102, 2.)])
        self.assertEqual(set(second), {(101, 1.), (102, 2.)})

    def test_permission_failure_cannot_be_reported_as_successful_cleanup(self):
        root = Mock(pid=101)
        root.is_running.return_value = True
        root.status.side_effect = psutil.AccessDenied(101)
        popen = Mock(pid=101)
        tracker = batch._OwnedProcessTracker.__new__(batch._OwnedProcessTracker)
        tracker.process, tracker.root, tracker.root_key = popen, root, (101, 1.)
        tracker.owned = {tracker.root_key: root}
        with self.assertRaises(psutil.AccessDenied):
            batch.interrupt_owned({"fixture": {"process": popen, "tracker": tracker}}, grace_seconds=0)
        root.send_signal.assert_not_called()


if __name__ == "__main__":
    unittest.main()
