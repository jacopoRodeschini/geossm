"""Check that printing an object has no side effects (finding F12)."""


def _public_state(obj):
    # Identity of every public attribute: catches attributes added or rebound.
    # Private ones are allowed to change: they are memo caches of
    # deterministic values (e.g. `_aic`, `_coverage_probability_cache`).
    return {k: id(v) for k, v in vars(obj).items() if not k.startswith("_")}


def assert_printing_is_side_effect_free(obj):
    before = _public_state(obj)
    str(obj)
    repr(obj)
    obj.summary()
    assert _public_state(obj) == before
