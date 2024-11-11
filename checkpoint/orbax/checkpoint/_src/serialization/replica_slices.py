# Copyright 2024 The Orbax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import dataclasses

from absl import logging
import jax
import jax.numpy as jnp
import math
import numpy as np
from orbax.checkpoint._src.arrays import numpy_utils
from orbax.checkpoint._src.arrays import types
from orbax.checkpoint._src.arrays.fragments import Fragment, Fragments, validate_fragments_can_be_stacked
from orbax.checkpoint._src.multihost import multihost
from typing import Callable, Optional, Sequence


Shape = types.Shape
Index = types.Index


@dataclasses.dataclass(frozen=True)
class ReplicaSliceOnDevice:
  """
  ReplicaSliceOnDevice represents the part of a jax.Shard that a replica is
  uniquely responsible for.

  With single-replica checkpointing the entirety of each jax.Shard is owned by
  exactly one replica. (Currently the only option.)
  """

  replica_id: int
  index: Index
  data: jax.Array


@dataclasses.dataclass
class ReplicaSlices:
  """
  ReplicaSlices groups all the sliced data of one jax.Array that a replica is
  uniquely responsible for. Slices may be either on-device (as a list of
  ReplicaSliceOnDevice) or on-host (as a list of indices and numpy arrays).
  """

  global_shape: Shape
  local_shape: Shape
  sharding: jax.sharding.Sharding
  dtype: np.dtype
  # Whether the replica slices have been transferred and are ready as ndarrays
  transferred: bool
  replica_slices: list[ReplicaSliceOnDevice] | list[tuple[Index, np.ndarray]]

  @property
  def nbytes(self) -> int:
    slice_nbytes = math.prod(self.local_shape) * self.dtype.itemsize
    return slice_nbytes * len(self.replica_slices)

  def to_fragments(self) -> Fragments:
    assert self.transferred
    fragments = Fragments(
        shape=self.global_shape,
        dtype=self.dtype,
        fragments=[
            Fragment(
                index=numpy_utils.resolve_slice(index, self.global_shape),
                value=data,
            )
            for index, data in self.replica_slices
        ],
    )
    if fragments.fragments:
      validate_fragments_can_be_stacked(fragments)
    if not fragments.is_degenerate():
      assert self.local_shape == fragments.fragments[0].shape
    return fragments


def get_replica_slices(
    arr: jax.Array,
    replica_id: Optional[int],
) -> ReplicaSlices:
  """Returns the replica slices a given replica is responsible for.
  Does not transfer allocate or transfer any data."""
  Result = tuple[list[ReplicaSliceOnDevice], Shape]
  shard0 = arr.addressable_shards[0]

  # single-replica: a single replica saves an entire shard.
  def pick_single_replica() -> Result:
    # Omitting the replica id just picks the first addressable shard's replica
    # id so that the process writes each of its addressable shards exactly
    # once. (This is the desired behavior for local checkpointing.)
    target_replica_id = replica_id or shard0.replica_id
    rslices = [
        ReplicaSliceOnDevice(
            replica_id=shard.replica_id,
            index=shard.index,
            data=shard.data,
        )
        for shard in arr.addressable_shards
        if shard.replica_id == target_replica_id
    ]
    local_shape = shard0.data.shape
    return rslices, local_shape

  shards_info = ', '.join(
      [
          f'Shard(index={shard.index}, replica_id={shard.replica_id})'
          for shard in arr.addressable_shards
      ]
  )
  logging.vlog(
      1,
      '[process=%d] get_replica_slices: replica_id=%d, shards=[%s]',
      multihost.process_index(),
      replica_id,
      shards_info,
  )

  # In order for all processes to agree on the right serialization metadata
  # we want to compute the correct local shape regardless of whether there
  # are any replica slices to save locally.
  rslices, local_shape = pick_single_replica()
  return ReplicaSlices(
      global_shape=arr.shape,
      local_shape=local_shape,
      sharding=arr.sharding,
      dtype=arr.dtype,
      transferred=False,
      replica_slices=rslices,
  )


def transfer_arrays_to_host(
    arrays: Sequence[jax.Array],
    replica_id: Optional[int],
    *,
    enable_pinned_host_transfer: bool = True,
) -> Sequence[ReplicaSlices]:
  """
  Transfers jax.Arrays to host memory and returns all the fragments to be
  serialized by the given replica, along with local shape. Blocks until
  completion.
  """

  def use_pinned_host_transfer(device):
    has_pinned_host = any(
        m.kind == 'pinned_host' for m in device.addressable_memories()
    )
    return (
        enable_pinned_host_transfer
        and has_pinned_host
        and jax._src.config.enable_memories.value  # pylint: disable=protected-access
    )

  def async_transfer_slice(rslice: ReplicaSliceOnDevice) -> tuple[Index, jax.Array]:
    index = rslice.index
    data = rslice.data
    device = data.device
    # Start the asynchronous device-to-host copy
    if use_pinned_host_transfer(device):
      # If available, transfer to pinned host memory
      data = jax.device_put(
          data,
          jax.sharding.SingleDeviceSharding(device, memory_kind='pinned_host'),
      )
    else:
      data.copy_to_host_async()
    return rslice.index, data

  # Gather the replica slices to be saved for each array.
  rslices_per_array = [get_replica_slices(arr, replica_id) for arr in arrays]
  # Kick off transfers for all replica slices to be saved.
  transfers_per_array = [
      [async_transfer_slice(rslice) for rslice in rslices.replica_slices]
      for rslices in rslices_per_array
  ]
  # Wait for all the transferred data to be ready.
  return [
      dataclasses.replace(
          rslices,
          transferred=True,
          # Conversion to numpy arrays forces block_until_ready.
          replica_slices=[(index, np.asarray(data)) for index, data in transfers],
      )
      for rslices, transfers in zip(rslices_per_array, transfers_per_array)
  ]
