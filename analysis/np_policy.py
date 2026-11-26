"""Run an SB3 PPO MlpPolicy + VecNormalize without torch/SB3 (numpy only)."""
import io, pickle, zipfile
import numpy as np

_DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64,
           "HalfStorage": np.float16, "LongStorage": np.int64,
           "IntStorage": np.int32, "ShortStorage": np.int16,
           "ByteStorage": np.uint8, "CharStorage": np.int8, "BoolStorage": np.bool_}


class _StorageType:
    def __init__(self, name):
        self.name = name


def _rebuild_tensor_v2(storage, offset, size, stride, *rest):
    item = storage.itemsize
    return np.lib.stride_tricks.as_strided(
        storage[offset:], shape=tuple(size), strides=tuple(s * item for s in stride)
    ).copy()


def load_state_dict(model_zip):
    outer = zipfile.ZipFile(model_zip)
    inner = zipfile.ZipFile(io.BytesIO(outer.read("policy.pth")))
    prefix = inner.namelist()[0].split("/")[0]

    class U(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
                return _rebuild_tensor_v2
            if module == "torch._utils" and name == "_rebuild_parameter":
                return lambda data, requires_grad, hooks, *a: data
            if module == "torch" and name in _DTYPES:
                return _StorageType(name)
            if module == "collections" and name == "OrderedDict":
                import collections
                return collections.OrderedDict
            raise pickle.UnpicklingError(f"unexpected class {module}.{name}")

        def persistent_load(self, pid):
            _, stype, key, _loc, numel = pid
            raw = inner.read(f"{prefix}/data/{key}")
            return np.frombuffer(raw, dtype=_DTYPES[stype.name])[: int(numel)]

    return U(io.BytesIO(inner.read(f"{prefix}/data.pkl"))).load()


class _Generic:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) == 2:
            state = {**(state[0] or {}), **(state[1] or {})}
        self.__dict__.update(state if isinstance(state, dict) else {"_state": state})


def load_vecnormalize(path):
    class U(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith(("stable_baselines3", "gymnasium")):
                return type(name, (_Generic,), {})
            return super().find_class(module, name)

    with open(path, "rb") as f:
        return U(f).load()


class NumpyPPO:
    def __init__(self, model_zip, stats_pkl):
        sd = load_state_dict(model_zip)
        self.layers = [(sd["mlp_extractor.policy_net.0.weight"], sd["mlp_extractor.policy_net.0.bias"]),
                       (sd["mlp_extractor.policy_net.2.weight"], sd["mlp_extractor.policy_net.2.bias"])]
        self.out = (sd["action_net.weight"], sd["action_net.bias"])
        vn = load_vecnormalize(stats_pkl)
        self.mean = np.asarray(vn.obs_rms.mean, dtype=np.float64)
        self.var = np.asarray(vn.obs_rms.var, dtype=np.float64)
        self.eps = float(getattr(vn, "epsilon", 1e-8))
        self.clip = float(getattr(vn, "clip_obs", 10.0))
        self.keys = list(sd.keys())

    def __call__(self, obs):
        x = np.clip((np.asarray(obs, np.float64) - self.mean) / np.sqrt(self.var + self.eps),
                    -self.clip, self.clip).astype(np.float32)
        for w, b in self.layers:
            x = np.tanh(w @ x + b)
        w, b = self.out
        return int(np.argmax(w @ x + b))
