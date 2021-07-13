from typing import Dict, Mapping, Optional, Tuple, Union

import numpy as np
import torch as th

from icecream import ic

from ail.common.type_alias import GymEnv


class Buffer(object):
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

    __slots__ = ["capacity", "sample_shapes", "_arrays", "stored_keys", "_n_data", "_idx", "device"]

    def __init__(
        self,
        capacity: int,
        sample_shapes: Mapping[str, Tuple[int, ...]],
        dtypes: Mapping[str, np.dtype],
        device: Union[th.device, str],
    ):
        if sample_shapes.keys() != dtypes.keys():
            raise KeyError("sample_shape and dtypes keys don't match")
        self.capacity = capacity
        self.sample_shapes = {k: tuple(shape) for k, shape in sample_shapes.items()}

        # The underlying NumPy arrays (which actually store the data).
        self._arrays = {
            k: np.zeros((capacity,) + shape, dtype=dtypes[k])
            for k, shape in self.sample_shapes.items()
        }

        self.stored_keys = set(self.sample_shapes.keys())

        # An integer in `range(0, self.capacity + 1)`.
        # This attribute is the return value of `self.size()`.
        self._n_data = 0
        # An integer in `range(0, self.capacity)`.
        self._idx = 0
        self.device = device

    def size(self) -> int:
        """Returns the number of samples currently stored in the buffer."""
        # _ndata: integer in `range(0, self.capacity + 1)`.
        assert 0 <= self._n_data <= self.capacity
        return self._n_data

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
            raise ValueError("Keys map to different length values")
        if capacity is None:
            capacity = data_capacities[0]

        sample_shapes = {k: arr.shape[1:] for k, arr in data.items()}
        dtypes = {k: arr.dtype for k, arr in data.items()}
        buf = cls(capacity, sample_shapes, dtypes, device=device)
        buf.store(data, truncate_ok=truncate_ok)
        return buf

    def store(self, data: Dict[str, np.ndarray], truncate_ok: bool = False, missing_ok=True) -> None:
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
        expected_keys = set(self.sample_shapes.keys())
        missing_keys = expected_keys - data_keys
        unexpected_keys = data_keys - expected_keys

        if len(missing_keys) and not missing_ok > 0:
            raise ValueError(f"Missing keys {missing_keys}")
        if len(unexpected_keys) > 0:
            raise ValueError(f"Unexpected keys {unexpected_keys}")

        n_samples = np.unique([arr.shape[0] for arr in data.values()])
        if len(n_samples) > 1:
            raise ValueError("Keys map to different length values.")

        n_samples = n_samples[0]
        if n_samples == 0:
            raise ValueError("Trying to store empty data.")

        if n_samples > self.capacity:
            if not truncate_ok:
                raise ValueError("Not enough capacity to store data.")
            else:
                data = {k: data[k][-self.capacity:] for k in data.keys()}

        for k in data.keys():
            if data[k].shape[1:] != self.sample_shapes[k]:
                ic(data[k].shape[1:])
                ic(self.sample_shapes[k])
                raise ValueError(f"Wrong data shape for {k}")

        new_idx = self._idx + n_samples
        if new_idx > self.capacity:
            n_remain = self.capacity - self._idx
            # Need to loop around the buffer. Break into two "easy" calls.
            self._store_easy({k: data[k][:n_remain] for k in data.keys()}, truncate_ok)
            assert self._idx == 0
            self._store_easy({k: data[k][n_remain:] for k in data.keys()}, truncate_ok)
        else:
            self._store_easy(data)

    def _store_easy(self, data: Dict[str, np.ndarray], truncate_ok=False) -> None:
        """
        Stores new data samples, replacing old samples with FIFO priority.
        Requires that `size(data) <= self.capacity - self._idx`, where `size(data)` is
        the number of rows in every array in `data.values()`.
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
        n_samples = np.unique([arr.shape[0] for arr in data.values()])
        assert len(n_samples) == 1
        n_samples = n_samples[0]
        assert n_samples <= self.capacity - self._idx
        idx_hi = self._idx + n_samples
        for k in data.keys():
            if not truncate_ok:

                if self._n_data + n_samples > self.capacity:
                    ic(self._n_data, n_samples)
                    raise ValueError("exceed buffer capacity")

            self._arrays[k][self._idx: idx_hi] = data[k]
        self._idx = idx_hi % self.capacity
        self._n_data = int(min(self._n_data + n_samples, self.capacity))

    def sample(self, n_samples: int) -> Dict[str, th.Tensor]:
        """
        Uniformly sample `n_samples` samples from the buffer with replacement.
        :param: n_samples: The number of samples to randomly sample.
            samples (np.ndarray): An array with shape `(n_samples) + self.sample_shape`.
        """
        # TODO: ERE (https://arxiv.org/pdf/1906.04009.pdf)
        assert self.size() != 0, "Buffer is empty"
        # uniform sampling
        ind = np.random.randint(self.size(), size=n_samples)
        return {k: self.to_torch(buffer[ind]) for k, buffer in self._arrays.items()}

    def get(self, n_samples: Optional[int] = None) -> Dict[str, th.Tensor]:
        if n_samples is None:
            assert self.size() == self.capacity, "Buffer is not full"    
            return self._get_samples()
        else:
            path_slice = slice(0, n_samples)
            return self._get_samples(path_slice)

    def _get_samples(self, batch_idxes: Union[np.ndarray, slice] = None):
        """Get a batch size or whole buffer size with order preserved."""
        batch_idxes = slice(0, self.capacity) if batch_idxes is None else batch_idxes
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


class BaseBuffer(object):
    """
    Base class that represent a buffer (rollout or replay)

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
        env: Optional[GymEnv] = None,
        obs_shape: Optional[Tuple[int, ...]] = None,
        act_shape: Optional[Tuple[int, ...]] = None,
        obs_dtype: Optional[np.dtype] = None,
        act_dtype: Optional[np.dtype] = None,
        with_reward=True,
    ):
        params = [obs_shape, act_shape, obs_dtype, act_dtype]
        self.sample_shapes = {}
        self.dtypes = {}
        if env is not None:
            if np.any([x is not None for x in params]):
                raise ValueError("Specified shape or dtype and environment.")

            self.sample_shapes.update(
                {
                    'obs': tuple(env.observation_space.shape),
                    'acts': tuple(env.action_space.shape),
                    'next_obs': tuple(env.observation_space.shape),
                    "dones": (1,),
                }
            )

            self.dtypes.update(
                {
                    'obs': env.observation_space.dtype,
                    'acts': env.action_space.dtype,
                    'next_obs': env.observation_space.dtype,
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
            self.sample_shapes['rews'] = (1,)
            self.dtypes['rews'] = np.float32

        self.capacity = capacity
        self.device = device
        self._buffer = None


    def _init_buffer(self) -> None:
        """Initiate Buffer"""
        assert len(self.sample_shapes) > 0, "sample shape not define"
        assert len(self.dtypes) > 0, "dtypes not define"
        self.reset()

    def reset(self):
        """ Reset equivalent to re-initiate a new Buffer"""
        self._buffer = Buffer(
            capacity=self.capacity,
            sample_shapes=self.sample_shapes,
            dtypes=self.dtypes,
            device=self.device,
        )

    def size(self) -> int:
        """Returns the number of samples stored in the buffer."""
        return self._buffer.size()

    def store(
        self, transitions: Dict[str, np.ndarray], truncate_ok: bool = False,
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
        intersect = self._buffer.stored_keys.intersection(transitions.keys())
        # Remove unnecessary fields
        trans_dict = {k: transitions[k] for k in intersect}
        self._buffer._store_easy(trans_dict, truncate_ok=truncate_ok)

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
        intersect = self._buffer.stored_keys.intersection(transitions.keys())
        # Remove unnecessary fields
        trans_dict = {k: transitions[k] for k in intersect}
        self._buffer.store(trans_dict, truncate_ok=truncate_ok)


    def sample(self, n_samples: int) -> Dict[str, th.Tensor]:
        """Sample obs-act-obs triples.
        Args:
            n_samples: The number of samples.
        Returns:
            A Transitions named tuple containing n_samples transitions.
        """
        return self._buffer.sample(n_samples)

    def get(self, n_samples: Optional[int] = None):
        return self._buffer.get(n_samples)

    @classmethod
    def from_data(
        cls,
        transitions: Dict[str, np.ndarray],
        device: Union[th.device, str],
        capacity: Optional[int] = None,
        truncate_ok: bool = False,
    ):
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
        # TODO: load from npz
        obs_shape = transitions["obs"].shape[1:]
        act_shape = transitions["act"].shape[1:]
        if capacity is None:
            capacity = transitions["obs"].shape[0]
        instance = cls(
            capacity=capacity,
            obs_shape=obs_shape,
            act_shape=act_shape,
            obs_dtype=transitions["obs"].dtype,
            act_dtype=transitions["act"].dtype,
            device=device,
        )
        instance._init_buffer()
        instance.store(transitions, truncate_ok=truncate_ok)
        return instance


class ReplayBuffer(BaseBuffer):
    """Replay Buffer for Transitions."""

    def __init__(
        self,
        capacity: int,
        device: Union[th.device, str],
        env: Optional[GymEnv] = None,
        obs_shape: Optional[Tuple[int, ...]] = None,
        act_shape: Optional[Tuple[int, ...]] = None,
        obs_dtype: Optional[np.dtype] = None,
        act_dtype: Optional[np.dtype] = None,
        with_reward=True,
        buf_kwargs=None,
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
            env,
            obs_shape,
            act_shape,
            obs_dtype,
            act_dtype,
            with_reward
        )

        if buf_kwargs is None:
            buf_kwargs = {}

        extra_sample_shapes = buf_kwargs.get("extra_sample_shapes", {})
        extra_sample_dtypes = buf_kwargs.get("extra_sample_dtypes", {})
        self.sample_shapes.update(extra_sample_shapes)
        self.dtypes.update(extra_sample_dtypes)


class RolloutBuffer(BaseBuffer):
    """Rollout Buffer for Transitions."""

    def __init__(
        self,
        capacity: int,
        device: Union[th.device, str],
        env: Optional[GymEnv] = None,
        obs_shape: Optional[Tuple[int, ...]] = None,
        act_shape: Optional[Tuple[int, ...]] = None,
        obs_dtype: Optional[np.dtype] = None,
        act_dtype: Optional[np.dtype] = None,
        with_reward=True,
        buf_kwargs=None,
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
            env,
            obs_shape,
            act_shape,
            obs_dtype,
            act_dtype,
            with_reward
        )
        if buf_kwargs is None:
            buf_kwargs = {}

        # log_pis, advs, rets, vals
        extra_shapes = buf_kwargs.get(
            "extra_shapes",
            {
                "advs": (1,),
                "rets": (1,),
                "vals": (1,),
                "log_pis": (1,),
            }
        )
        extra_dtypes = buf_kwargs.get(
            "extra_dtypes",
            {
                "advs": np.float32,
                "rets": np.float32,
                "vals": np.float32,
                "log_pis": np.float32,
            }
        )
        self.sample_shapes.update(extra_shapes)
        self.dtypes.update(extra_dtypes)
        self._init_buffer()

