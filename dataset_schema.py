"""
Shared dataset schema helpers for demonstration files.
"""

import numpy as np


SCHEMA_VERSION = "1.0"
DATASET_FORMAT = "nemotron_vla.transitions"


def _as_object_array(value, length, field_name):
    if isinstance(value, str):
        return np.array([value] * length, dtype=object)

    arr = np.asarray(value, dtype=object)
    if arr.ndim == 0:
        return np.array([str(arr.item())] * length, dtype=object)
    if arr.ndim != 1:
        raise ValueError(f"{field_name} must be scalar text or rank-1 text array")
    if len(arr) != length:
        raise ValueError(f"{field_name} length mismatch: expected {length}, got {len(arr)}")
    return arr


def build_transition_payload(images, states, actions, instructions, env_names, source):
    """
    Build a schema-consistent transition dataset payload.
    """
    images = np.asarray(images, dtype=np.uint8)
    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)

    n = len(states)
    if len(images) != n or len(actions) != n:
        raise ValueError(
            f"Transition length mismatch: images={len(images)}, states={len(states)}, actions={len(actions)}"
        )

    instructions_arr = _as_object_array(instructions, n, "instructions")
    env_names_arr = _as_object_array(env_names, n, "env_names")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_format": DATASET_FORMAT,
        "source": source,
        "num_transitions": np.int64(n),
        "images": images,
        "states": states,
        "actions": actions,
        "instructions": instructions_arr,  # per-transition text
        "env_names": env_names_arr,        # per-transition env name
    }

    # Legacy compatibility keys for older notebook cells.
    unique_instr = np.unique(instructions_arr)
    if len(unique_instr) == 1:
        payload["instruction"] = str(unique_instr[0])
    unique_envs = np.unique(env_names_arr)
    if len(unique_envs) == 1:
        payload["env_name"] = str(unique_envs[0])

    return payload


def unpack_transition_payload(npz_data, default_env_name="unknown-v0"):
    """
    Read both old and new transition dataset files into a unified in-memory dict.
    """
    images = np.asarray(npz_data["images"], dtype=np.uint8)
    states = np.asarray(npz_data["states"], dtype=np.float32)
    actions = np.asarray(npz_data["actions"], dtype=np.float32)
    n = len(states)

    if "instructions" in npz_data:
        instructions = _as_object_array(npz_data["instructions"], n, "instructions")
    elif "instruction" in npz_data:
        instructions = _as_object_array(str(npz_data["instruction"]), n, "instructions")
    else:
        instructions = _as_object_array("", n, "instructions")

    if "env_names" in npz_data:
        env_names = _as_object_array(npz_data["env_names"], n, "env_names")
    elif "env_name" in npz_data:
        env_names = _as_object_array(str(npz_data["env_name"]), n, "env_names")
    else:
        env_names = _as_object_array(default_env_name, n, "env_names")

    source = str(npz_data["source"]) if "source" in npz_data else "legacy"

    return build_transition_payload(
        images=images,
        states=states,
        actions=actions,
        instructions=instructions,
        env_names=env_names,
        source=source,
    )
