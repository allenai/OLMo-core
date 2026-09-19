"""Reuse the known-good hero Modal app; preserve independent sandbox isolation."""

APP = "swerex-7ee77286d853"
_installed = False


def install():
    """Fail closed instead of creating more apps when the account quota is full."""
    global _installed
    if _installed:
        return
    import modal

    original = modal.App.lookup
    # Provenance: successful hero code experiment 01M22D1V352S52XWTM8GPHTVXM.
    existing = original(APP, create_if_missing=False)

    def lookup(name, *args, **kwargs):
        if name == "swe-rex" or name.startswith("swerex-"):
            return existing
        return original(name, *args, **kwargs)

    modal.App.lookup = lookup
    _installed = True
    print("SFT_EXISTING_MODAL_APP", APP, flush=True)
