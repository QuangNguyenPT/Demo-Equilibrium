import ctypes
import os
import platform
from typing import List, Tuple, Optional
import numpy as np


class HRMReasoningResult(ctypes.Structure):
    _fields_ = [
        ("steps_executed", ctypes.c_int64),
        ("halted_early", ctypes.c_int32),
        ("halt_confidence", ctypes.c_float),
    ]


class EngineBridge:
    """
    Zero-overhead cross-language execution bridge.
    Binds the Python blueprint layer to the bare-metal C++ engine
    via pure C-ABI ctypes interface.
    """

    def __init__(self, lib_path: Optional[str] = None):
        if lib_path is None:
            lib_path = self._discover_library_path()

        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"[BRIDGE] Could not find compiled C++ engine DLL at: {lib_path}\n"
                "Please run cmake / clang build first."
            )

        self.lib = ctypes.CDLL(lib_path)
        self._setup_function_signatures()
        self.engine_handle = None

    def _discover_library_path(self) -> str:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        system = platform.system()

        candidates = []
        if system == "Windows":
            candidates = [
                os.path.join(base_dir, "bin", "hrm_engine.dll"),
                os.path.join(base_dir, "build", "bin", "hrm_engine.dll"),
                os.path.join(base_dir, "build", "Release", "hrm_engine.dll"),
                os.path.join(base_dir, "hrm_engine.dll"),
            ]
        elif system == "Darwin":
            candidates = [
                os.path.join(base_dir, "bin", "libhrm_engine.dylib"),
                os.path.join(base_dir, "build", "libhrm_engine.dylib"),
            ]
        else:
            candidates = [
                os.path.join(base_dir, "bin", "libhrm_engine.so"),
                os.path.join(base_dir, "build", "libhrm_engine.so"),
            ]

        for path in candidates:
            if os.path.exists(path):
                return path

        return candidates[0]

    def _setup_function_signatures(self) -> None:
        c_void_p = ctypes.c_void_p
        c_int32 = ctypes.c_int32
        c_int64 = ctypes.c_int64
        c_float = ctypes.c_float
        c_size_t = ctypes.c_size_t
        c_char_p = ctypes.c_char_p

        # Engine
        self.lib.hrm_engine_create.argtypes = [c_size_t, c_int32]
        self.lib.hrm_engine_create.restype = c_void_p

        self.lib.hrm_engine_destroy.argtypes = [c_void_p]
        self.lib.hrm_engine_destroy.restype = None

        self.lib.hrm_engine_reset_arena.argtypes = [c_void_p]
        self.lib.hrm_engine_reset_arena.restype = None

        # Tensor
        self.lib.hrm_tensor_create.argtypes = [ctypes.POINTER(c_int64), c_int32, c_int32, c_void_p]
        self.lib.hrm_tensor_create.restype = c_void_p

        self.lib.hrm_tensor_wrap.argtypes = [c_void_p, ctypes.POINTER(c_int64), c_int32, c_int32]
        self.lib.hrm_tensor_wrap.restype = c_void_p

        self.lib.hrm_tensor_destroy.argtypes = [c_void_p]
        self.lib.hrm_tensor_destroy.restype = None

        self.lib.hrm_tensor_data_ptr.argtypes = [c_void_p]
        self.lib.hrm_tensor_data_ptr.restype = c_void_p

        self.lib.hrm_tensor_numel.argtypes = [c_void_p]
        self.lib.hrm_tensor_numel.restype = c_int64

        # Ops
        self.lib.hrm_op_gemm.argtypes = [c_void_p, c_void_p, c_void_p]
        self.lib.hrm_op_gemm.restype = c_int32

        self.lib.hrm_op_rms_norm.argtypes = [c_void_p, c_void_p, c_void_p, c_float]
        self.lib.hrm_op_rms_norm.restype = c_int32

        self.lib.hrm_op_swiglu.argtypes = [c_void_p, c_void_p, c_void_p]
        self.lib.hrm_op_swiglu.restype = c_int32

        self.lib.hrm_op_flash_attention.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int32]
        self.lib.hrm_op_flash_attention.restype = c_int32

        self.lib.hrm_op_reasoning_loop.argtypes = [
            c_void_p, c_void_p, c_void_p,
            c_void_p, c_void_p, c_void_p,
            c_void_p, c_void_p, c_void_p,
            c_void_p, c_void_p,
            c_int64, c_int64, c_float,
            ctypes.POINTER(HRMReasoningResult),
        ]
        self.lib.hrm_op_reasoning_loop.restype = c_int32

        self.lib.hrm_version_string.argtypes = []
        self.lib.hrm_version_string.restype = c_char_p

    def init_engine(self, arena_mb: int = 128, threads: int = 4) -> None:
        self.engine_handle = self.lib.hrm_engine_create(arena_mb, threads)

    def close(self) -> None:
        if self.engine_handle:
            self.lib.hrm_engine_destroy(self.engine_handle)
            self.engine_handle = None

    def reset_arena(self) -> None:
        if self.engine_handle:
            self.lib.hrm_engine_reset_arena(self.engine_handle)

    def get_version(self) -> str:
        return self.lib.hrm_version_string().decode("utf-8")

    def create_tensor(self, shape: List[int], dtype: int = 0) -> "NativeTensor":
        shape_arr = (ctypes.c_int64 * len(shape))(*shape)
        handle = self.lib.hrm_tensor_create(shape_arr, len(shape), dtype, self.engine_handle)
        return NativeTensor(handle, shape, self)

    def wrap_numpy(self, np_array: np.ndarray) -> "NativeTensor":
        """Zero-copy wrap of a contiguous NumPy array as a C++ tensor."""
        if not np_array.flags["C_CONTIGUOUS"]:
            np_array = np.ascontiguousarray(np_array)
        shape = list(np_array.shape)
        shape_arr = (ctypes.c_int64 * len(shape))(*shape)
        data_ptr = np_array.ctypes.data_as(ctypes.c_void_p)
        handle = self.lib.hrm_tensor_wrap(data_ptr, shape_arr, len(shape), 0)
        tensor = NativeTensor(handle, shape, self, keep_alive=np_array)
        return tensor

    def run_reasoning_loop(
        self,
        z_h: "NativeTensor",
        z_l: "NativeTensor",
        input_emb: "NativeTensor",
        w_low: "NativeTensor",
        u_low: "NativeTensor",
        b_low: Optional["NativeTensor"],
        w_high: "NativeTensor",
        u_high: "NativeTensor",
        b_high: Optional["NativeTensor"],
        w_qhead: "NativeTensor",
        b_qhead: Optional["NativeTensor"],
        max_steps: int = 8,
        low_steps: int = 2,
        act_threshold: float = 0.5,
    ) -> HRMReasoningResult:
        result = HRMReasoningResult()
        b_low_h = b_low.handle if b_low else None
        b_high_h = b_high.handle if b_high else None
        b_qhead_h = b_qhead.handle if b_qhead else None

        status = self.lib.hrm_op_reasoning_loop(
            z_h.handle, z_l.handle, input_emb.handle,
            w_low.handle, u_low.handle, b_low_h,
            w_high.handle, u_high.handle, b_high_h,
            w_qhead.handle, b_qhead_h,
            max_steps, low_steps, act_threshold,
            ctypes.byref(result),
        )
        if status != 0:
            raise RuntimeError(f"[BRIDGE] Reasoning loop failed with status: {status}")
        return result


class NativeTensor:
    """Wrapper around C++ hrm::Tensor handle with NumPy view support."""

    def __init__(self, handle: ctypes.c_void_p, shape: List[int], bridge: EngineBridge, keep_alive=None):
        self.handle = handle
        self.shape = shape
        self.bridge = bridge
        self._keep_alive = keep_alive

    def __del__(self):
        if self.handle and self.bridge and self.bridge.lib:
            try:
                self.bridge.lib.hrm_tensor_destroy(self.handle)
            except Exception:
                pass
            self.handle = None

    @property
    def numel(self) -> int:
        return self.bridge.lib.hrm_tensor_numel(self.handle)

    def numpy_view(self) -> np.ndarray:
        data_ptr = self.bridge.lib.hrm_tensor_data_ptr(self.handle)
        buffer = (ctypes.c_float * self.numel).from_address(data_ptr)
        arr = np.ctypeslib.as_array(buffer).reshape(self.shape)
        return arr
