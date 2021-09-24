from typing import Dict, Mapping, Optional, Tuple, Union, Set
from enum import Enum, auto
import os
import warnings

import numpy as np
from numpy.random.mtrand import seed
import torch as th
from traitlets.traitlets import default

from ail.common.type_alias import GymEnv
from ail.common.math import normalize


class Buffer:
    __slots__ = [
        "_capacity",
        "_sample_shapes",
        "_arrays",
        "_stored_keys",
        "_n_data",
        "_idx",
        "device",
        "seed",
        "rng"
    ]
    """
    A FIFO ring buffer for NumPy arrays of a fixed shape and dtype.
    Supports random sampling with replacement.

    :param capacity: The number of data samples that can be stored in this buffer.

    :param sample_shapes: A dictionary mapping string keys to the shape of each data
                samples associated with that key.

    :param dtypes:  A dictionary mapping string keys to the dtype of  each data
                of samples associated with that key.

    :param device: PyTorch device to which the values will be converted.
    """

    def __init__(
        self,
        capacity: int,
        sample_shapes: Mapping[str, Tuple[int, ...]],
        dtypes: Mapping[str, np.dtype],
        device: Union[th.device, str],
        seed: int
    ):
        assert isinstance(capacity, int), "capacity must be integer."

        if sample_shapes.keys() != dtypes.keys():
            raise KeyError("sample_shape and dtypes keys don't match.")
        self._capacity = capacity
        self._sample_shapes = {k: tuple(shape) for k, shape in sample_shapes.items()}

        # The underlying NumPy arrays (which actually store the data).
        self._arrays = {
            k: np.zeros((capacity,) + shape, dtype=dtypes[k])
            for k, shape in self._sample_shapes.items()
        }

        self._stored_keys = set(self._sample_shapes.keys())

        # An integer in `range(0, self.capacity + 1)`.
        # This attribute is the return value of `self.size()`.
        self._n_data = 0
        # An integer in `range(0, self.capacity)`.
        self._idx = 0
        self.device = device
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def sample_shapes(self) -> Mapping[str, Tuple[int, ...]]:
        return self._sample_shapes

    @property
    def stored_keys(self) -> Set[str]:
        return self._stored_keys

    def size(self) -> int:
        """Returns the number of samples currently stored in the buffer."""
        # _ndata: integer in `range(0, self.capacity + 1)`.
        assert (
            0 <= self._n_data <= self._capacity
        ), "_ndata: integer in range(0, self.capacity + 1)."
        return self._n_data

    def full(self) -> bool:
        """Returns True if the buffer is full, False otherwise."""
        return self.size() == self._capacity

    @classmethod
    def from_data(
        cls,
        data: Dict[str, np.ndarray],
        device: Union[th.device, str],
        capacity: Optional[int] = None,
        truncate_ok: bool = False,
    ) -> "Buffer":
        """
        Constructs and return a Buffer containing the provided data.
        Shapes and dtypes are automatically inferred.

        :param data: A dictionary mapping keys to data arrays. The arrays may differ
                in their shape, but should agree in the first axis.

        :param device: PyTorch device to which the values will be converted.

        :param capacity: The Buffer capacity. If not provided, then this is automatically
                set to the size of the data, so that the returned Buffer is at full
                capacity.
        :param truncate_ok: Whether to error if `capacity` < the number of samples in
                `data`. If False, then only store the last `capacity` samples from
                `data` when overcapacity.
        Examples:
            In the follow examples, suppose the arrays in `data` are length-1000.
            `Buffer` with same capacity as arrays in `data`::
                Buffer.from_data(data)
            `Buffer` with larger capacity than arrays in `data`::
                Buffer.from_data(data, 10000)
            `Buffer with smaller capacity than arrays in `data`. Without
            `truncate_ok=True`, `from_data` will error::
                Buffer.from_data(data, 5, truncate_ok=True)
        """
        data_capacities = [arr.shape[0] for arr in data.values()]
        data_capacities = np.unique(data_capacities)
        if len(data) == 0:
            raise ValueError("No keys in data.")
        if len(data_capacities) > 1:
            raise ValueError("Keys map to different length values.")
        if capacity is None:
            capacity = data_capacities[0]

        sample_shapes = {k: arr.shape[1:] for k, arr in data.items()}
        dtypes = {k: arr.dtype for k, arr in data.items()}
        buf = cls(capacity, sample_shapes, dtypes, device=device)
        buf.store(data, truncate_ok=truncate_ok)
        return buf

    def store(
        self,
        data: Dict[str, np.ndarray],
        truncate_ok: bool = False,
        missing_ok: bool = True,
    ) -> None:
        """
        Stores new data samples, replacing old samples with FIFO priority.

        :param data: A dictionary mapping keys `k` to arrays with shape
            `(n_samples,) + self.sample_shapes[k]`,
            where `n_samples` is less than or equal to `self.capacity`.
        :param truncate_ok: If False, then error if the length of `transitions` is
            greater than `self.capacity`.
            Otherwise, store only the final `self.capacity` transitions.
        :param missing_ok: If False, then error if attempt to store a subset of
            sample's key store in buffer
        """
        data_keys = set(data.keys())
        expected_keys = set(self._sample_shapes.keys())
        missing_keys = expected_keys - data_keys
        unexpected_keys = data_keys - expected_keys

        if missing_keys and not missing_ok:
            raise ValueError(f"Missing keys {missing_keys}")
        if unexpected_keys:
            raise ValueError(f"Unexpected keys {unexpected_keys}")

        n_samples = np.unique([arr.shape[0] for arr in data.values()])
        if len(n_samples) > 1:
            raise ValueError("Keys map to different length values.")

        n_samples = n_samples[0]
        if n_samples == 0:
            raise ValueError("Trying to store empty data.")

        if n_samples > self._capacity:
            if not truncate_ok:
                raise ValueError("Not enough capacity to store data.")
            else:
                data = {k: data[k][-self._capacity :] for k in data.keys()}

        for k in data.keys():
            if data[k].shape[1:] != self._sample_shapes[k]:
                raise ValueError(f"Wrong data shape for {k}.")

        new_idx = self._idx + n_samples
        if new_idx > self._capacity:
            n_remain = self._capacity - self._idx
            # Need to loop around the buffer. Break into two "easy" calls.
            self._store_easy({k: data[k][:n_remain] for k in data.keys()}, truncate_ok)
            assert self._idx == 0
            self._store_easy({k: data[k][n_remain:] for k in data.keys()}, truncate_ok)
        else:
            self._store_easy(data)

    def _store_easy(self, data: Dict[str, np.ndarray], truncate_ok=False) -> None:
        """
        Stores new data samples, replacing old samples with FIFO priority.
        Requires that `size(data) <= self.capacity - self._idx`,
        where `size(data)` is the number of rows in every array in `data.values()`.

        Updates `self._idx` to be the insertion point of the next call to `_store_easy` call,
        looping back to `self._idx = 0` if necessary.
        Also updates `self._n_data`.

        :param data: Same as in `self.store`'s docstring, except with the additional
            constraint `size(data) <= self.capacity - self._idx`.
        :param truncate_ok: If False, then error if the length of `transitions` is
            greater than `self.capacity`.
            Otherwise, store only the final `self.capacity` transitions.
        Note: serve as singe pair store
        """
        assert isinstance(data, dict), "data must be a dictionary"
        # shape (1, n): 1 is the number of samples, n is the dimension of that sample
        n_samples = np.unique([arr.shape[0] for arr in data.values()])
        assert len(n_samples) == 1

        n_samples = n_samples[0]
        assert n_samples <= self._capacity - self._idx
        idx_hi = self._idx + n_samples

        for k in data.keys():
            if not truncate_ok:
                if self._n_data + n_samples > self._capacity:
                    raise ValueError("exceed buffer capacity")
            self._arrays[k][self._idx : idx_hi] = data[k]
        self._idx = idx_hi % self._capacity
        self._n_data = int(min(self._n_data + n_samples, self._capacity))

    def sample(self, n_samples: int) -> Dict[str, th.Tensor]:
        """
        Uniformly sample `n_samples` samples from the buffer with replacement.
        :param n_samples: The number of samples to randomly sample.
        :return: A dictionary of samples (np.ndarray)
            with shape `(n_samples) + self.sample_shape`.
        """
        # TODO: ERE (https://arxiv.org/pdf/1906.04009.pdf)
        assert isinstance(n_samples, int), "n_samples must be int"
        assert self.size() != 0, "Buffer is empty"
        # Uniform sampling
        ind = self.rng.integers(self.size(), size=n_samples)
        return self._get_batch_from_index(ind)
        
        # res_idx = []
        # res2={}
        # for idx, (k, buffer) in enumerate(self._arrays.items()):
        #     if idx == 0:
        #         enum = self.rng.choice(list(enumerate(buffer[:self.size()])), size=n_samples, replace=True, p=None, axis=0, shuffle=True)
                
        #         # ic(k, enum)
        #         for i in enum:
        #             res_idx.append(i[0])
        #         # ic(res_idx)
        #     break
        # res2 = self._get_batch_from_index(np.asarray(res_idx))
        # ic(res_idx)
        # ic(res2)
        # return res2

    def get(
        self,
        n_samples: Optional[int] = None,
        last_n: bool = True,
        shuffle: bool = False,
    ) -> Dict[str, th.Tensor]:
        """
        Returns samples in the buffer.
        :param: n_samples: The number of samples to return.
            By default, return all samples in the buffer, if n_samples is None.
        :param last_n: If True, then return the last `n_samples` samples.
        :param shuffle: If True, then return the samples in a random order.
        return: Tensor Dict
        """
        if n_samples is None:
            assert self.full(), "Buffer is not full"
            # Obatain all data in buffer.
            if shuffle:
                # Same as uniform sampling whole buffer.
                return self.sample(n_samples=self._capacity)
            else:
                # Get all buffer data with order preserved.
                return self._get_batch_from_index(batch_idxes=slice(0, self._capacity))
        else:
            # Obtain a slice of data in buffer
            assert isinstance(n_samples, int), "n_samples must be integer."
            n_data = self.size()
            if n_samples > n_data:
                raise ValueError(
                    f"Cannot get {n_samples} of samples, "
                    f"which exceeds {n_data} samples currrently store in buffer."
                )
            if last_n:
                # Obtain `last n_samples` data with index in range [n_data - n_samples, n_data)
                start, end = (n_data - n_samples), n_data
            else:
                # Obtain data with index in range [0, n_samples)
                start, end = 0, n_samples

            batch_idxes = (
                np.random.randint(low=start, high=end, size=n_samples)
                if shuffle
                else slice(start, end)
            )
            return self._get_batch_from_index(batch_idxes)

    def _get_batch_from_index(
        self,
        batch_idxes: Union[np.ndarray, slice],
    ) -> Dict[str, th.Tensor]:
        """
        Get a batch data based on index.
        :param batch_idxes: Index of batch.
        :param shuffle: If True, then return the samples in a random order.
        """
        assert isinstance(batch_idxes, (slice, np.ndarray))
        return {
            k: self.to_torch(buffer[batch_idxes]) for k, buffer in self._arrays.items()
        }

    def to_torch(self, array: np.ndarray, copy: bool = True, **kwargs) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default.
        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        """
        if copy:
            return th.tensor(array, dtype=th.float32, device=self.device, **kwargs)
        elif isinstance(array, np.ndarray):
            return th.from_numpy(array).float().to(self.device)
        else:
            return th.as_tensor(array, dtype=th.float32, device=self.device)

    @staticmethod
    def to_numpy(tensor: th.Tensor) -> np.ndarray:
        """Convert torch tensor to numpy array and send to CPU."""
        return tensor.detach().cpu().numpy()

    def save(self, save_dir: str) -> None:
        """
        Saving the data in buffer as .npz archive to a directory.
        see: https://numpy.org/doc/stable/reference/generated/numpy.savez.html#numpy.savez
        """
        dir_name = os.path.dirname(save_dir)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        print("Saving buffer _arrays into a .npz archive.")
        print(f"data key: {self._arrays.keys()}")
        np.savez(
            save_dir,
            obs=self._arrays["obs"],
            acts=self._arrays["acts"],
            dones=self._arrays["dones"],
            next_obs=self._arrays["next_obs"],
        )


class BaseBuffer:
    __slots__ = [
        "capacity",
        "sample_shapes",
        "dtypes",
        "device",
        "_buffer",
        "seed"
        "abs_counter",
    ]

    """
    Base class that represent a buffer (rollout or replay).

    :param capacity: The number of samples that can be stored.
    :param device: PyTorch device to which the values will be converted.
    :param env: The environment whose action and observation
        spaces can be used to determine the data shapes of
        the underlying buffers.
        Overrides all the following arguments.
    :param obs_shape: The shape of the observation space.
    :param act_shape: The shape of the action space.
    :param obs_dtype: The dtype of the observation space.
    :param act_dtype: The dtype of the action space.
    """

    def __init__(
        self,
        capacity: int,
        device: Union[th.device, str],
        seed: int,
        env: Optional[GymEnv] = None,
        obs_shape: Optional[Tuple[int, ...]] = None,
        act_shape: Optional[Tuple[int, ...]] = None,
        obs_dtype: np.dtype = np.float32,
        act_dtype: np.dtype = np.float32,
        with_reward=True,
    ):
        if isinstance(capacity, float):
            self.capacity = int(capacity)
        elif isinstance(capacity, int):
            self.capacity = capacity
        else:
            raise ValueError("capacity must be integer number.")

        params = [obs_shape, act_shape, obs_dtype, act_dtype]
        self.sample_shapes = {}
        self.dtypes = {}
        if env is not None:
            if np.any([x is not None for x in params]):
                print("Specified shape and dtype and environment.", flush=True)
                print("Shape and dtypes will be refer to env.", flush=True)
            self.sample_shapes.update(
                {
                    "obs": tuple(env.observation_space.shape),
                    "acts": tuple(env.action_space.shape),
                    "next_obs": tuple(env.observation_space.shape),
                    "dones": (1,),
                }
            )

            self.dtypes.update(
                {
                    "obs": env.observation_space.dtype,
                    "acts": env.action_space.dtype,
                    "next_obs": env.observation_space.dtype,
                    "dones": np.float32,
                }
            )
        else:
            if np.any([x is None for x in params]):
                raise ValueError("Shape or dtype missing and no environment specified.")

            self.sample_shapes = {
                "obs": tuple(obs_shape),
                "acts": tuple(act_shape),
                "next_obs": tuple(obs_shape),
                "dones": (1,),
            }
            self.dtypes = {
                "obs": obs_dtype,
                "acts": act_dtype,
                "next_obs": obs_dtype,
                "dones": np.float32,
            }

        if with_reward:
            self.sample_shapes["rews"] = (1,)
            self.dtypes["rews"] = np.float32

        self.device = device
        self._buffer = None
        self.seed = seed
        self.abs_counter=0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def _init_buffer(self) -> None:
        """Initiate Buffer"""
        if len(self.sample_shapes) == 0:
            raise ValueError("sample shape not define.")
        if len(self.dtypes) == 0:
            raise ValueError("dtypes not define.")
        self.reset()

    def reset(self) -> None:
        """Reset equivalent to re-initiate a new Buffer."""
        self._buffer = Buffer(
            capacity=self.capacity,
            sample_shapes=self.sample_shapes,
            dtypes=self.dtypes,
            device=self.device,
            seed=self.seed,
        )

    def stored_keys(self) -> Set[str]:
        return self._buffer.stored_keys

    def size(self) -> int:
        """Returns the number of samples stored in the buffer."""
        return self._buffer.size()

    def full(self) -> bool:
        """Returns whether the buffer is full."""
        return self._buffer.full()

    def store(
        self,
        transitions: Dict[str, np.ndarray],
        truncate_ok: bool = False,
    ) -> None:
        """Store obs-act-obs triples and additional info in transitions.
        Args:
          transitions: Transitions to store.
          truncate_ok: If False, then error if the length of `transitions` is
            greater than `self.capacity`. Otherwise, store only the final
            `self.capacity` transitions.
        Raises:
            ValueError: The arguments didn't have the same length.
        """
        if not isinstance(transitions, dict):
            try:
                transitions = dict(transitions)
            except TypeError:
                raise TypeError(
                    "Prefer transitions to be a dict or a dictionary-like object"
                )
        keys = set(transitions.keys())
        intersect = self._buffer._stored_keys & keys
        difference = self._buffer._stored_keys - keys
        ignore = keys - self._buffer._stored_keys

        if difference:
            warnings.warn(f"Unfulfill keys: {difference}.")
        if ignore:
            warnings.warn(f"Ignore keys: {ignore}.")

        # Remove unnecessary fields
        trans_dict = {k: transitions[k] for k in intersect}
        self._buffer._store_easy(trans_dict, truncate_ok=truncate_ok)  # noqa

    def store_path(
        self, transitions: Dict[str, np.ndarray], truncate_ok: bool = True
    ) -> None:
        """Store a path of obs-act-obs triples and additional info in transitions.
        Args:
          transitions: Transitions to store.
          truncate_ok: If False, then error if the length of `transitions` is
            greater than `self.capacity`. Otherwise, store only the final
            `self.capacity` transitions.
        Raises:
            ValueError: The arguments didn't have the same length.
        """
        if not isinstance(transitions, dict):
            try:
                transitions = dict(transitions)
            except TypeError:
                raise TypeError(
                    "Prefer transitions to be a dict or a dictionary-like object."
                )
        keys = set(transitions.keys())
        intersect = self._buffer._stored_keys & keys
        difference = self._buffer._stored_keys - keys
        ignore = keys - self._buffer._stored_keys

        if difference:
            warnings.warn(f"Unfulfill keys: {difference}.")
        if ignore:
            warnings.warn(f"Ignore keys: {ignore}.")

        # Remove unnecessary fields
        trans_dict = {k: transitions[k] for k in intersect}
        self._buffer.store(trans_dict, truncate_ok=truncate_ok)

    def sample(self, n_samples: int) -> Dict[str, th.Tensor]:
        """
        Sample obs-act-obs triples.
        :param n_samples: The number of samples.
        :return:A Transitions named tuple containing n_samples transitions.
        """
        return self._buffer.sample(n_samples)

    def get(
        self,
        n_samples: Optional[int] = None,
        last_n: bool = True,
        shuffle: bool = False,
    ) -> Dict[str, th.Tensor]:
        """
        Obtain a batch of samples with size = n_samples. (order preserved)
            By default, return all samples in the buffer, if n_samples is None.
        """
        return self._buffer.get(n_samples, last_n, shuffle)

    @classmethod
    def from_data(
        cls,
        transitions: Dict[str, np.ndarray],
        device: Union[th.device, str],
        seed: int,
        capacity: Optional[int] = None,
        truncate_ok: bool = False,
        with_reward: bool = True,
    ) -> "BaseBuffer":
        """
        Construct and return a ReplayBuffer/RolloutBuffer containing the provided data.
        Shapes and dtypes are automatically inferred, and the returned ReplayBuffer is
        ready for sampling.
        Args:
            transitions: Transitions to store.
            device: PyTorch device to which the values will be converted.
            capacity: The ReplayBuffer capacity. If not provided, then this is
                automatically set to the size of the data, so that the returned Buffer
                is at full capacity.
            truncate_ok: Whether to error if `capacity` < the number of samples in
                `data`. If False, then only store the last `capacity` samples from
                `data` when overcapacity.
        Examples:
            `ReplayBuffer` with same capacity as arrays in `data`::
                ReplayBuffer.from_data(data)
            `ReplayBuffer` with larger capacity than arrays in `data`::
                ReplayBuffer.from_data(data, 10000)
            `ReplayBuffer with smaller capacity than arrays in `data`. Without
            `truncate_ok=True`, `from_data` will error::
                ReplayBuffer.from_data(data, 5, truncate_ok=True)
        Returns:
            A new ReplayBuffer.
        """
        obs_shape = transitions["obs"].shape[1:]
        act_shape = transitions["acts"].shape[1:]
        if capacity is None:
            capacity = transitions["obs"].shape[0]
        instance = cls(
            capacity=capacity,
            obs_shape=obs_shape,
            act_shape=act_shape,
            obs_dtype=transitions["obs"].dtype,
            act_dtype=transitions["acts"].dtype,
            device=device,
            seed=seed,
            with_reward=with_reward,
        )
        instance._init_buffer()
        instance.store_path(transitions, truncate_ok=truncate_ok)
        return instance

    def save(self, save_dir) -> None:
        """Save trainsitions to save_dir."""
        self._buffer.save(save_dir)


class BufferTag(Enum):
    REPLAY = auto()
    ROLLOUT = auto()


class ReplayBuffer(BaseBuffer):
    """Replay Buffer for Transitions."""

    def __init__(
        self,
        capacity: int,
        device: Union[th.device, str],
        seed: int,
        env: Optional[GymEnv] = None,
        obs_shape: Optional[Tuple[int, ...]] = None,
        act_shape: Optional[Tuple[int, ...]] = None,
        obs_dtype: np.dtype = np.float32,
        act_dtype: np.dtype = np.float32,
        with_reward: bool = True,
        extra_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
        extra_dtypes: Optional[Dict[str, np.dtype]] = None,
    ):
        """
        Constructs a ReplayBuffer.

        :param capacity: The number of samples that can be stored.
        :param device: PyTorch device to which the values will be converted.
        :param env: The environment whose action and observation
            spaces can be used to determine the data shapes of
            the underlying buffers.
            Overrides all the following arguments.
        :param obs_shape: The shape of the observation space.
        :param act_shape: The shape of the action space.
        :param obs_dtype: The dtype of the observation space.
        :param act_dtype: The dtype of the action space.
        """

        super(ReplayBuffer, self).__init__(
            capacity,
            device,
            seed,
            env,
            obs_shape,
            act_shape,
            obs_dtype,
            act_dtype,
            with_reward,
        )

        if extra_shapes is not None:
            if isinstance(extra_shapes, dict):
                self.sample_shapes.update(extra_shapes)
            else:
                raise ValueError("extra_shapes should be Dict[str, Tuple[int, ...]]")
        if extra_dtypes is not None:
            if isinstance(extra_dtypes, dict):
                self.dtypes.update(extra_dtypes)
            else:
                raise ValueError("extra_dtypes should be Dict[str, np.dtype]")

        self._init_buffer()
        self._tag = BufferTag.REPLAY

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} (capacity={self.capacity}, data={self.stored_keys()}, size={self.size()})"

    @property
    def tag(self) -> BufferTag:
        return self._tag


class RolloutBuffer(BaseBuffer):
    """Rollout Buffer for Transitions."""

    def __init__(
        self,
        capacity: int,
        device: Union[th.device, str],
        seed: int,
        env: Optional[GymEnv] = None,
        obs_shape: Optional[Tuple[int, ...]] = None,
        act_shape: Optional[Tuple[int, ...]] = None,
        obs_dtype: np.dtype = np.float32,
        act_dtype: np.dtype = np.float32,
        with_reward: bool = True,
        extra_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
        extra_dtypes: Optional[Dict[str, np.dtype]] = None,
    ):
        """
        Constructs a ReplayBuffer.

        :param capacity: The number of samples that can be stored.
        :param device: PyTorch device to which the values will be converted.
        :param env: The environment whose action and observation
            spaces can be used to determine the data shapes of
            the underlying buffers.
            Overrides all the following arguments.
        :param obs_shape: The shape of the observation space.
        :param act_shape: The shape of the action space.
        :param obs_dtype: The dtype of the observation space.
        :param act_dtype: The dtype of the action space.
        """

        super(RolloutBuffer, self).__init__(
            capacity,
            device,
            seed,
            env,
            obs_shape,
            act_shape,
            obs_dtype,
            act_dtype,
            with_reward,
        )

        if extra_shapes is not None:
            if isinstance(extra_shapes, dict):
                self.sample_shapes.update(extra_shapes)
            else:
                raise ValueError("extra_shapes should be Dict[str, Tuple[int, ...]]")
        if extra_dtypes is not None:
            if isinstance(extra_dtypes, dict):
                self.dtypes.update(extra_dtypes)
            else:
                raise ValueError("extra_dtypes should be Dict[str, np.dtype]")

        self._init_buffer()
        self._tag = BufferTag.ROLLOUT

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} (capacity={self.capacity}, data={self.stored_keys()}, size={self.size()})"

    @property
    def tag(self) -> BufferTag:
        return self._tag


class BufferType(Enum):
    rollout = RolloutBuffer
    replay = ReplayBuffer
    rolloutbuffer = RolloutBuffer
    replaybuffer = ReplayBuffer
    rollout_buffer = RolloutBuffer
    replay_buffer = ReplayBuffer
    ROLLOUT_BUFFER = RolloutBuffer
    REPLAY_BUFFER = ReplayBuffer
