import random
import sys
import unittest

sys.path.insert(0, "src")

from promptlab.experiments import Experiments
from promptlab.registry import Registry
from promptlab.stats import welch_t_test


def make_platform():
    registry = Registry(":memory:")
    return registry, Experiments(registry.conn, registry)


class RegistryTests(unittest.TestCase):
    def test_versioning_rollback_and_audit(self):
        registry, _ = make_platform()
        pid = registry.create_prompt("p", "hello {{name}}", "v1")
        registry.add_version(pid, "hi {{name}}", "v2 shorter")
        registry.activate(pid, 2)
        self.assertEqual(registry.active_version(pid), 2)
        registry.activate(pid, 1, reason="rollback")
        self.assertEqual(registry.active_version(pid), 1)
        actions = [e["action"] for e in registry.audit_log()]
        self.assertIn("version_activated", actions)

    def test_template_variable_validation(self):
        registry, _ = make_platform()
        self.assertEqual(Registry.fill("hi {{name}}", {"name": "Ada"}),
                         "hi Ada")
        with self.assertRaises(ValueError):
            Registry.fill("hi {{name}}", {})

    def test_diff(self):
        registry, _ = make_platform()
        pid = registry.create_prompt("p", "alpha", "v1")
        registry.add_version(pid, "beta", "v2")
        self.assertIn("-alpha", registry.diff(pid, 1, 2))


class ExperimentTests(unittest.TestCase):
    def test_consistent_assignment(self):
        registry, experiments = make_platform()
        pid = registry.create_prompt("p", "x", "v1")
        registry.add_version(pid, "y", "v2")
        eid = experiments.create("e", pid, [{"version": 1, "split": 50},
                                            {"version": 2, "split": 50}],
                                 "quality")
        experiment = experiments.get(eid)
        for user in range(100):
            a = experiments.assign(experiment, f"u{user}")
            b = experiments.assign(experiment, f"u{user}")
            self.assertEqual(a, b)

    def test_split_roughly_respected(self):
        registry, experiments = make_platform()
        pid = registry.create_prompt("p", "x", "v1")
        registry.add_version(pid, "y", "v2")
        eid = experiments.create("e", pid, [{"version": 1, "split": 80},
                                            {"version": 2, "split": 20}],
                                 "quality")
        experiment = experiments.get(eid)
        assignments = [experiments.assign(experiment, f"u{i}")
                       for i in range(2000)]
        share_v1 = assignments.count(1) / len(assignments)
        self.assertAlmostEqual(share_v1, 0.8, delta=0.05)

    def test_auto_stop_on_error_rate(self):
        registry, experiments = make_platform()
        pid = registry.create_prompt("p", "x", "v1")
        registry.add_version(pid, "y", "v2")
        eid = experiments.create("e", pid, [{"version": 1, "split": 50},
                                            {"version": 2, "split": 50}],
                                 "quality")
        for i in range(40):
            experiments.record(eid, f"u{i}", 2, {"quality": 0.5},
                               error=(i % 3 == 0))
        message = experiments.check_guardrails(eid)
        self.assertIsNotNone(message)
        self.assertEqual(experiments.get(eid)["status"], "stopped")


class StatsTests(unittest.TestCase):
    def test_significant_difference_detected(self):
        rng = random.Random(1)
        a = [rng.gauss(0.60, 0.1) for _ in range(300)]
        b = [rng.gauss(0.70, 0.1) for _ in range(300)]
        result = welch_t_test(a, b)
        self.assertTrue(result.significant)
        self.assertLess(result.p_value, 0.001)

    def test_identical_distributions_not_significant(self):
        rng = random.Random(2)
        a = [rng.gauss(0.6, 0.1) for _ in range(300)]
        b = [rng.gauss(0.6, 0.1) for _ in range(300)]
        self.assertFalse(welch_t_test(a, b).significant)


if __name__ == "__main__":
    unittest.main()
