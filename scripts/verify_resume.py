"""Compare full completed checkpoints from uninterrupted and resumed runs."""

from __future__ import annotations

import argparse

import torch
from agari_kit.collection import write_json
from agari_kit.storage import Storage
from agari_research.manifest import file_sha256
from agari_research.planning import selected_stages


def identical(left, right, path="checkpoint"):
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.dtype != right.dtype or not torch.equal(left, right):
            raise AssertionError(f"Tensor mismatch: {path}")
        return 1
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise AssertionError(f"Mapping mismatch: {path}")
        return sum(identical(value, right[key], f"{path}.{key}") for key, value in left.items())
    if isinstance(left, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            raise AssertionError(f"Sequence mismatch: {path}")
        return sum(identical(a, b, f"{path}[{index}]") for index, (a, b) in enumerate(zip(left, right)))
    if left != right:
        raise AssertionError(f"Value mismatch: {path}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True, help="Storage-relative uninterrupted checkpoint")
    parser.add_argument("--right", required=True, help="Storage-relative resumed checkpoint")
    parser.add_argument("--output", required=True, help="Storage-relative JSON evidence output")
    args = parser.parse_args()
    storage = Storage.from_env()
    storage.check(probe=True)
    left_path, right_path = storage.path(args.left), storage.path(args.right)
    left = torch.load(left_path, map_location="cpu", weights_only=True)
    right = torch.load(right_path, map_location="cpu", weights_only=True)
    names = [left.pop("run_name"), right.pop("run_name")]
    for stage in selected_stages(left["config"]):
        target = (left["training_plan"]["stages"][stage]["target_updates"] if "training_plan" in left
                  else left["config"][stage]["steps"])
        if left["progress"][stage] != target:
            raise AssertionError("Compare completed checkpoints")
    tensors = identical(left, right)
    result = {"status": "exact-match", "runs": names, "tensors_compared": tensors,
              "checked": "All checkpoint fields except distinct run_name, including optimizer/scheduler/RNG/data cursor",
              "dataset_sha256": left["dataset_sha256"], "implementation_sha256": left["implementation_sha256"],
              "left_sha256": file_sha256(left_path), "right_sha256": file_sha256(right_path)}
    write_json(storage.path(args.output), result)
    print(result)


if __name__ == "__main__":
    main()
