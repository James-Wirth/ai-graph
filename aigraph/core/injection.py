from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Annotated, get_origin, get_args
from pydantic import BaseModel

class Inject:
    __aigraph_inject__ = True

@dataclass(frozen=True)
class From(Inject):
    path: str  

@dataclass(frozen=True)
class FromVar(Inject):
    path: str  

@dataclass(frozen=True)
class ToolValue(Inject):
    name: str        
    field: str = "output"  

class Context(Inject): ...
class Payload(Inject): ...

def get_path(root: Any, path: str) -> Any:
    cur = root
    for part in (path or "").split("."):
        if part == "":
            continue
        cur = (cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None))
        if cur is None:
            break
    return cur

def _is_annotated_with(t, marker_type) -> tuple[bool, Any]:
    if get_origin(t) is Annotated:
        base, *extras = get_args(t)
        for e in extras:
            if isinstance(e, marker_type):
                return True, (base, e)
        return True, (base, None)
    return False, (t, None)

def _coerce(value, typ):
    try:
        if value is None:
            return None
        if isinstance(typ, type) and issubclass(typ, BaseModel):
            return typ.model_validate(value)
        return value
    except Exception:
        return value

def resolve_fn_args(
    param_specs: List[Tuple[str, Any]],
    payload: Any,
    ctx_vars: Dict[str, Any],
    tool_ns: Dict[str, Any],
) -> Dict[str, Any]:
    
    out: Dict[str, Any] = {}
    for name, ann in param_specs:
        is_anno, (base, injected) = _is_annotated_with(ann, Inject)
        if isinstance(injected, From):
            out[name] = _coerce(get_path(payload, injected.path), base); continue
        if isinstance(injected, FromVar):
            out[name] = _coerce(get_path(ctx_vars, injected.path), base); continue
        if isinstance(injected, ToolValue):
            t = tool_ns.get(injected.name) or {}
            if injected.field == "all":
                out[name] = t
            else:
                out[name] = t.get(injected.field) if isinstance(t, dict) else None
            continue
        if ann is Context:
            out[name] = ctx_vars; continue
        if ann is Payload:
            out[name] = payload; continue
        out[name] = _coerce(get_path(payload, name), ann if is_anno else ann)
    return out
