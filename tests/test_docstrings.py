"""NumPy-style docstrings of the public API (Section R).

Every public class/function in the packages' ``__all__`` and every public
method/property of the exported classes has a docstring; a callable with
parameters has a ``Parameters`` section naming exactly its parameters (a
class may document them in its own or its ``__init__`` docstring); a function
that returns a value has a ``Returns`` section.
"""

import ast
import importlib
import inspect
import re
import textwrap

import pytest

PACKAGES = ["geossm", "geossm.ssm", "geossm.stmodel", "geossm.covmodel", "geossm.datasets"]
_SECTION = r"^\s*{}\s*\n\s*-{{3,}}\s*$"


def _parameters_section(doc):
    """Names listed in the ``Parameters`` section, or ``None`` without one."""
    m = re.search(r"^\s*Parameters\s*\n\s*-{3,}\s*\n(.*?)(?=^\s*[A-Z][A-Za-z ]+\s*\n\s*-{3,}\s*$|\Z)",
                  doc, re.S | re.M)
    if m is None:
        return None
    names = set()
    for line in m.group(1).splitlines():
        if line and not line[0].isspace():
            names.update(n.strip().lstrip("*") for n in line.split(":")[0].split(","))
    return names


def _returns_a_value(fn):
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return False
    try:
        body = ast.parse(src).body
    except SyntaxError:  # a docstring less indented than the def
        body = ast.parse("if True:\n" + textwrap.indent(src, " ")).body[0].body
    return any(isinstance(n, ast.Return) and n.value is not None
               and not (isinstance(n.value, ast.Constant) and n.value.value is None)
               for n in ast.walk(body[0]))


def _public_objects():
    seen = set()
    for pkg in PACKAGES:
        mod = importlib.import_module(pkg)
        for name in mod.__all__:
            obj = getattr(mod, name)
            if inspect.ismodule(obj) or not callable(obj) or id(obj) in seen:
                continue
            seen.add(id(obj))
            qual = f"{obj.__module__}.{name}"
            yield qual, obj, "class" if inspect.isclass(obj) else "function"
            if inspect.isclass(obj):
                for attr, val in vars(obj).items():
                    if attr.startswith("_"):
                        continue
                    if isinstance(val, property):
                        yield f"{qual}.{attr}", val, "property"
                    elif isinstance(val, (staticmethod, classmethod)):
                        yield f"{qual}.{attr}", val.__func__, "function"
                    elif inspect.isfunction(val):
                        yield f"{qual}.{attr}", val, "function"


def _problems(obj, kind):
    doc = inspect.getdoc(obj) or ""
    if not doc.strip():
        return ["no docstring"]
    if kind == "property":
        return []
    fn = obj.__init__ if kind == "class" else obj
    if kind == "class" and "__init__" in vars(obj) and vars(obj)["__init__"].__doc__:
        doc += "\n" + inspect.getdoc(obj.__init__)
    try:
        sig = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return []
    params = [p.name for p in sig
              if p.name not in ("self", "cls") and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    var_params = {p.name for p in sig if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
    out = []
    if params:
        documented = _parameters_section(doc)
        if documented is None:
            out.append(f"no Parameters section for {params}")
        else:
            if set(params) - documented:
                out.append(f"undocumented parameters {sorted(set(params) - documented)}")
            extra = documented - set(params) - var_params - {""}
            if extra:
                out.append(f"documented but not in the signature {sorted(extra)}")
    if kind == "function" and _returns_a_value(fn) and not re.search(_SECTION.format("(Returns|Yields)"), doc, re.M):
        out.append("no Returns section")
    return out


OBJECTS = list(_public_objects())


def test_the_public_api_is_found():
    assert len(OBJECTS) > 200


@pytest.mark.parametrize("qual, obj, kind", OBJECTS, ids=[q for q, _, _ in OBJECTS])
def test_numpy_docstring(qual, obj, kind):
    assert _problems(obj, kind) == []


def test_spde_docstring_documents_only_its_own_parameters():
    # gstools appends the CovModel constructor parameters (len_scale, anis, ...)
    # to subclass docstrings; SPDEApproxCov does not take them.
    from geossm.covmodel import SPDEApproxCov
    assert "len_scale" not in SPDEApproxCov.__doc__ and "anis" not in SPDEApproxCov.__doc__
