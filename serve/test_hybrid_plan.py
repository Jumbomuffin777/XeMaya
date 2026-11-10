# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""XeMaya: the hybrid segment's plan file (server.hybrid_plan, hybrid_args) and the engine's report of it
(engine_hybrid)."""
import json
import os
import tempfile
import unittest

from serve import server


class HybridPlan(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "m.hybrid.json")
        self.cfg = {"args": ["--native", "m.gguf"], "hybrid_file": self.path}
        self.token = os.environ.pop("XEMAYA_SEGMENT_TOKEN", None)

    def tearDown(self):
        self.dir.cleanup()
        os.environ.pop("XEMAYA_SEGMENT_TOKEN", None)
        if self.token is not None:
            os.environ["XEMAYA_SEGMENT_TOKEN"] = self.token

    def write(self, plan):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(plan, f)

    def test_no_plan_no_arguments(self):
        self.assertEqual(server.hybrid_args({"args": []}), [])
        self.assertEqual(server.hybrid_args(self.cfg), [])           # the file is not there
        open(self.path, "w").close()
        self.assertEqual(server.hybrid_args(self.cfg), [])           # ... or empty
        self.write({"enabled": False, "segment_last": 45, "remote": "pc:7411"})
        self.assertEqual(server.hybrid_args(self.cfg), [])           # ... or switched off

    def test_plan_becomes_arguments_and_the_token_stays_off_the_command_line(self):
        self.write({"segment_last": 45, "remote": "10.0.0.7:7411", "wire": "f16", "token": "s3cret"})
        args = server.engine_args(self.cfg)
        self.assertEqual(args[:2], ["--native", "m.gguf"])
        self.assertEqual(args[-6:], ["--segment-last", "45", "--remote-segment", "10.0.0.7:7411", "--segment-wire", "f16"])
        self.assertNotIn("s3cret", args)
        self.assertEqual(os.environ["XEMAYA_SEGMENT_TOKEN"], "s3cret")

    def test_a_broken_plan_stops_the_start(self):
        for bad in ({"segment_last": 0, "remote": "pc:1"}, {"segment_last": 45, "remote": "pc"},
                    {"segment_last": "45", "remote": "pc:1"}, {"segment_last": 45, "remote": "pc:1", "wire": "f8"}):
            self.write(bad)
            with self.assertRaises(ValueError):
                server.hybrid_args(self.cfg)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("{not json")
        with self.assertRaises(ValueError):
            server.hybrid_args(self.cfg)

    def test_the_engines_report(self):
        self.assertIsNone(server.engine_hybrid({}))
        got = server.engine_hybrid({"hybrid": "10.0.0.7:7411", "hybrid_blocks": "45-59",
                                    "hybrid_device": "NVIDIA_GeForce_RTX_4090", "hybrid_wire": "f32"})
        self.assertEqual(got, {"address": "10.0.0.7:7411", "device": "NVIDIA GeForce RTX 4090", "wire": "f32",
                               "first_block": 45, "last_block": 59})


if __name__ == "__main__":
    unittest.main()
