"""Create a class that saves out data objects from Sionna and a class that reads these in and compares them with
TorchRF data objects at the same points. """

import pickle
import os
from pathlib import Path
import time
import tensorflow as tf
import torch
import json
import drjit
import mitsuba as mi
import numpy as np
import torchrf


# noinspection PyAttributeOutsideInit
class DataLogger(object):
    """
    This object is similar to a call stack but it only logs specific states that are manually recorded
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logger = super(DataLogger, cls).__new__(cls)
            logger.stack = []
            logger.dir = Path("logs")
            logger.index_file = Path(logger.dir, "index.json")
            logger.filenames = {}
            logger.mode = 'ignore'
            logger.focal_prefix = ''
            cls._instance = logger
            # special_classes gets strings rather than classes in order to avoid importing torchrf during sionna runs
            logger.special_classes = {"torchrf.rt.paths.Paths"}
        return cls._instance

    def focus(self, prefix):
        self.focal_prefix = prefix

    def unfocus(self):
        self.focal_prefix = ''

    def set_mode(self, mode):
        """
        The logger has 4 possible modes:
        1. write - allows writing data structures out to files
        2. ignore - ignore compare calls
        3. print - print messages for compare calls but never break
        4. break - break when a comparison fails

        States 2-4 are read-only states.  The user always needs to put the logger back into 'write' mode in order
        to write.  This helps prevent unwanted overwrites.

        Parameters
        ----------
        mode: str - the mode to set to

        Returns
        -------
        this object

        """
        known = set("write ignore print break".split())
        if mode in known:
            self.mode = mode
        else:
            raise ValueError (f"Unknown mode `{mode}`")
        return self

    def set_dir(self, dir):
        """

        Parameters
        ----------
        dir: str or Path - the directory in which to save data

        Returns
        -------
        not used
        """
        self.dir = Path(dir)

    def set_index(self, filename="index.json"):
        self.index_file = Path(self.dir, filename)


    @classmethod
    def push(cls, label):
        """

        Parameters
        ----------
        label: str - identifier for this position in the call stack

        Returns
        -------
        This object
        """
        logger = DataLogger()
        logger.stack.append(label)
        return logger

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.mode != 'ignore':
            self.stack.pop()
            if len(self.stack) == 0:
                self.save()

    def save(self):
        if self.mode == 'write':
            with open(self.index_file, 'w') as f:
                json.dump(self.filenames, f, indent=2)

    # TODO: should we require object_name?  When would we use None?
    def write(self, obj, object_name=None):
        """
        Write the data object to disk, labeled with the current log tags

        Parameters
        ----------
        obj - Any data object to be saved to disk
        frame - A string describing the position in the code

        Returns
        -------
        the filename to which the object was written
        """
        if self.mode == 'ignore':
            return None
        if self.mode != 'write':
            raise RuntimeError('Cannot write from a read-only DataLogger object')
        if object_name is not None:
            self.push(object_name)
        sid = self.stack_id()
        if sid in self.filenames:
            self.pop()
            raise RuntimeError(f"Trying to save a second object to {sid}")
        if not self.dir.exists():
            os.mkdir(self.dir)            
        filename = Path(self.dir, f'log_{time.time():0.0f}.pkl')
        while filename.exists():
            filename = Path(self.dir, f'log_{time.time():0.4f}'.replace('.', '_') + '.pkl')
        self.filenames[sid] = str(filename.absolute())
        data = self.prepickle(obj)
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        if object_name is not None:
            self.pop()
        return filename

    def prepickle(self, obj):
        """
        Convert data into picklable objects.
        Parameters
        ----------
        obj

        Returns
        -------

        """
        # Case 1: already picklable
        if obj is None or any([isinstance(obj, dtype) for dtype in
                               [str, bytes, bytearray, int, float]]):
            return obj
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.complex64) or isinstance(obj, np.complex128) or isinstance(obj, np.complex256):
            return complex(obj)
        # Case 2: a special class; ignored for writing
        if self.mode != 'write':
            for special_class in self.special_classes:
                if isinstance(obj, eval(special_class)):
                    return obj
        if "Scene" in str(type(obj)):
            return "BLACKBOX"
        # For singleton values like tensor(3.14), tf returns 3.14 where torch returns ndarray(3.14).
        #  Put these back in the box so we maintain type consistency.
        elif isinstance(obj, np.ndarray):
            if len(obj.shape) == 1 and obj.shape[0] == 1:
                return obj[0]
            else:
                return obj
        elif isinstance(obj, tf.Tensor):
            try:
                value = obj.numpy()
            except AttributeError:
                tf.compat.v1.enable_eager_execution()
                value = obj.numpy()
            return np.array(value)
        elif isinstance(obj, torch.Tensor):
            return obj.detach().numpy()
        elif isinstance(obj, drjit.llvm.ad.TensorXf):
            return obj.numpy()
        elif isinstance(obj, mi.Ray3f):
            return self.prepickle(['Ray3f', obj.d, obj.maxt, obj.o, obj.time, obj.wavelengths])
        # Recur on dictionary
        try:
            return {key: self.prepickle(value) for key, value in obj.items()}
        except AttributeError:
            # Recur on iterables
            try:
                return [self.prepickle(value) for value in obj]
            except TypeError:
                # Build dictionary on class objects:
                try:
                    return {key: self.prepickle(value) for key, value in obj.__dict__.items()}
                except AttributeError:
                    return str(obj)

    def pop(self):
        self.stack.pop()

    def stack_id(self):
        s = ':'.join(self.stack)
        return s

    def reset(self):
        """
        Delete all stored data from the DataLogger, as if the object is newly created.

        Returns
        -------
        Not used
        """
        self.stack = []
        self.filenames = {}
        self.mode = 'ignore'

    @classmethod
    def load(cls, filename=None):
        if cls._instance is None:
            logger = DataLogger()
        else:
            logger = cls._instance
            if not (len(logger.stack) == 0 and len(logger.filenames) == 0):
                raise RuntimeError("You cannot load into an existing DataLogger.  Run DataLogger.reset() before DataLogger.load().")
        if logger.mode == 'ignore':
            return logger
        if filename is None:
            filename = logger.index_file
        try:
            with open(filename) as f:
                logger.filenames = json.load(f)
        except FileNotFoundError:
            with open(Path(logger.dir, filename)) as f:
                logger.filenames = json.load(f)
        if logger.mode == 'write':
            logger.mode = 'break'
        return logger

    def compare(self, obj, object_name=None, tol=1e-6, unique=False, dim=1, no_break=False):
        if self.mode != 'ignore':
            self.unique = unique
            self.unique_dim = dim
            self.compare_error = None
            if tol is not None:
                self.tol = tol
            try:
                ref = self.get(object_name, keep=True)
                if self.stack_id().startswith(self.focal_prefix):
                    current = self.prepickle(obj)
                    self._compare(current, ref)
                    if self.compare_error:
                        if self.mode == 'break' and not no_break:
                            breakpoint()
                    else:
                        self.print()
            except KeyError:
                self.print("   No saved reference data")
                if self.mode == 'break' and not no_break:
                    breakpoint()
            except FileNotFoundError:
                self.print(f"File for {self.stack_id()} saved but not found")
                if self.mode == 'break' and not no_break:
                    breakpoint()
            if object_name is not None:
                self.pop()

    def print(self, msg=None):
        if self.mode == 'print' or self.mode == 'break':
            sid = self.stack_id()
            if sid != self.compare_error:
                print(f"At log point `{self.stack_id()}`:" + (" passed" if msg is None else ''))
            if msg is not None:
                print(msg)
                self.compare_error = sid

    def get(self, object_name, keep=False):
        if self.mode == 'ignore':
            return None
        if len(self.filenames) == 0:
            self.load()
        if object_name is not None:
            if ':' in object_name:
                sid = object_name
            else:
                self.push(object_name)
                sid = self.stack_id()
        else:
            sid = self.stack_id()
        try:
            filename = self.filenames[sid]
            with open(filename, 'rb') as f:
                data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Current log tag `{sid}` maps to `{filename}`, but no such file")
        except KeyError:
            raise KeyError(f"Current log tag `{sid}` was not logged in reference run")
        if not keep and object_name is not None:
            self.pop()
        return data

    def _type_mismatch(self, a, b):
        self.print(f"  current type [{type(a)}] does not match reference type [{type(b)}]")

    def _compare(self, current, ref):
        for special_class in self.special_classes:
            if isinstance(current, eval(special_class)):
                self._compare_special(special_class, current, ref)
                return
        if isinstance(current, dict):
            if isinstance(ref, dict):
                self._compare_dict(current, ref)
            else:
                self._type_mismatch(current, ref)
        elif isinstance(current, str) or isinstance(current, bytes) or isinstance(current, bytearray):
            if current != ref:
                self.print(f"  {current} does not match {ref}")
        elif isinstance(current, np.ndarray):
            if isinstance(ref, np.ndarray):
                self._compare_array(current, ref)
            else:
                self._type_mismatch(current, ref)
        elif hasattr(current, '__iter__'):
            if hasattr(ref, '__iter__'):
                self._compare_list(current, ref)
            else:
                self._type_mismatch(current, ref)
        elif isinstance(ref, float):
            if isinstance(current, float):
                if np.abs(current - ref) > self.tol:
                    self.print(f"   floats have different values")
            else:
                self._type_mismatch(current, ref)
        elif type(ref) != type(current):
            self._type_mismatch(current, ref)
        elif ref != current:
            self.print(f"   values don't match")

    def _compare_array(self, current, ref):
        if current.shape != ref.shape:
            self.print(f"   Current shape {current.shape} does not match reference {ref.shape}")
        if ref.dtype != current.dtype:
            self.print(f"   Current dtype {current.dtype} does not match reference {ref.dtype}")
        try:
            # Don't need to compare empty arrays
            if 0 not in ref.shape:
                if len(ref.shape) > 0 and self.unique:
                    axis = self.unique_dim if self.unique_dim < len(current.shape) else len(current.shape) - 1
                    u_current = np.unique(current, axis=axis)
                    u_ref = np.unique(ref, axis=axis)
                    if u_current.shape != current.shape:
                        self.print(f"   expected unique current tensor, but not unique")
                    if u_ref.shape != ref.shape:
                        self.print(f"   expected unique reference tensor, but not unique")
                    if u_current.shape == current.shape and u_ref.shape == ref.shape:
                        current, ref = u_current, u_ref
                if ref.dtype == np.dtype('bool'):
                    if not (ref == current).all():
                        self.print(f"   tensors have different values")
                elif np.abs(current - ref).max() > self.tol:
                    self.print(f"   tensors have different values")
                    if ref.std() > current.std() * 10:
                        self.print("    ==> current low std")
        except ValueError:
            pass

    def _compare_dict(self, current, ref):
        extra_d = ref.keys() - current.keys()
        if len(extra_d) > 0:
            self.print("  stored data object [data] has keys not in current data object [obj]:")
            for key in extra_d:
                self.print("   " + key)
        extra_o = current.keys() - ref.keys()
        if len(extra_o) > 0:
            self.print("  current data object [obj] has keys not in stored data object [data]:")
            for key in extra_o:
                self.print("   " + key)
        for key, value in ref.items():
            self.push(f"[{key}]")
            try:
                self._compare(current[key], value)
            except KeyError:
                pass
            self.pop()

    def _compare_list(self, current, ref):
        if len(current) != len(ref):
            self.print(f"  current length {len(current)} but reference length {len(ref)}")
        else:
            for i in range(len(current)):
                self.push(f"[{i}]")
                self._compare(current[i], ref[i])
                self.pop()

    def _compare_special(self, special_class, obj, ref):
        self_unique = self.unique
        self.unique = False
        some_error = self.compare_error
        if special_class == "torchrf.rt.paths.Paths":
            # Fields that should match exactly
            for attribute in ('sources', 'targets', 'reverse_direction', 'normalize_delays'):
                self.compare_error = False
                self.push(attribute)
                self._compare(self.prepickle(obj.__getattribute__(attribute)), ref[f'_{attribute}'])
                if not self.compare_error:
                    self.print()
                self.pop()
                some_error |= self.compare_error

            # Align the ordering of the paths between the two objects
            self.compare_error = False
            self.push('a')
            obj_a, obj_indices = np.unique(obj.a.detach().numpy(), axis=5, return_index=True)
            ref_a, ref_indices = np.unique(ref['_a'], axis=5, return_index=True)
            if obj_a.shape != obj.a.shape:
                self.print("   paths in Paths.a tensor not unique!")
            self._compare_array(obj_a, ref_a)
            if not self.compare_error:
                self.print()
            self.pop()

            # Fields that have paths in the final dimension
            for attribute in ('tau', 'theta_t', 'theta_r', 'phi_t', 'phi_r', 'mask', 'objects', 'types'):
                self.compare_error = False
                self.push(attribute)
                obj_array = obj.__getattribute__(attribute)[..., obj_indices]
                ref_array = ref[f'_{attribute}'][..., ref_indices]
                self._compare(self.prepickle(obj_array), ref_array)
                if not self.compare_error:
                    self.print()
                self.pop()
                some_error |= self.compare_error
            self.compare_error = some_error
        else:
            raise ValueError(f"Unknown Special Class {special_class}")
        self.unique = self_unique
