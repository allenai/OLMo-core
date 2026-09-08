"""Placement-only controller adapter, sent inline with the qualified source pin.

Only the not-yet-submitted 32Mi child changes. Keep the parent's exact specification
so its existing durable intent reconciles without changing or restarting training.
"""

import sys

sys.path.insert(0, "src/examples/olmo_ddp")

import olmoe3_medium_cbs_control as control

EXCLUDED_HOST = "holmes-cs-aus-503.reviz.ai2.in"
QUALIFIED_COMMIT = "85878d12be1198188863a68cabf04168b0fa8849"
original_training_spec = control.training_spec


def training_spec(template, run, **settings):
    """Remove only the confirmed bad host from the future production child."""
    spec = original_training_spec(template, run, **settings)
    if run == control.BRANCH:
        task = spec["tasks"][0]
        hosts = task["constraints"]["hostname"]
        task["constraints"]["hostname"] = [host for host in hosts if host != EXCLUDED_HOST]
        assert len(task["constraints"]["hostname"]) >= 16
        assert EXCLUDED_HOST not in task["constraints"]["hostname"]
        control.log("CBS_CHILD_PLACEMENT", run=run.run_id, excluded=EXCLUDED_HOST)
    return spec


def main():
    """Use the existing locked, gated, duplicate-safe CBS watcher without new training code."""
    import os

    assert os.environ["GIT_REF"] == QUALIFIED_COMMIT
    control.UPLOADER = "01M1YTRJV16A5YC300SEZAW6D1"
    control.training_spec = training_spec
    control.log("CBS_PLACEMENT_OVERRIDE_ENABLED", excluded=EXCLUDED_HOST, parent_unchanged=True)
    control.main()


if __name__ == "__main__":
    main()
