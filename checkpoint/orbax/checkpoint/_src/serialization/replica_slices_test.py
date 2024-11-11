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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from orbax.checkpoint._src.serialization.replica_slices import get_replica_slices, transfer_arrays_to_host


def make_multi_device_array(shape, spec):
  key = jax.random.PRNGKey(0)
  x = jax.random.normal(jax.random.PRNGKey(0), shape)
  mesh = jax.sharding.Mesh(jax.devices(), ('x',))
  sharding = jax.sharding.NamedSharding(mesh, spec)
  return jax.device_put(x, sharding)


def is_pow_of_two(n):
  while n > 1:
    n, rem = divmod(n, 2)
    if rem == 1:
      return False
  return True


class ReplicaSlicesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    num_devices = len(jax.devices())
    assert num_devices >= 2
    assert is_pow_of_two(num_devices)

  def test_get_replica_slices_single_replica(self):
    replicated_spec = jax.sharding.PartitionSpec()
    arr = make_multi_device_array((64, 64), replicated_spec)
    num_replicas = len(arr.devices())

    # Using an addressable replica_id yields that replica.
    for replica_id in range(num_replicas):
      rslices = get_replica_slices(arr, replica_id=replica_id).replica_slices
      self.assertEqual(len(rslices), 1)
      self.assertEqual(rslices[0].replica_id, replica_id)

    # Omitting replica_id yields _some_ replica.
    rslices = get_replica_slices(arr, replica_id=None).replica_slices
    self.assertEqual(len(rslices), 1)

    # Using an unaddressable replica_id yields nothing.
    rslices = get_replica_slices(arr, replica_id=-1).replica_slices
    self.assertEqual(len(rslices), 0)

  def test_transfer(self):
    replicated_spec = jax.sharding.PartitionSpec()
    arr = make_multi_device_array((32, 32), replicated_spec)
    shard0 = next(shard for shard in arr.addressable_shards if shard.replica_id == 0)

    rslices = transfer_arrays_to_host([arr], replica_id=0)[0]
    self.assertEqual(len(rslices.replica_slices), 1)
    index, shard_data_on_host = rslices.replica_slices[0]
    self.assertEqual(index, shard0.index)
    self.assertIsInstance(shard_data_on_host, np.ndarray)
    np.testing.assert_array_equal(shard0.data, shard_data_on_host)


if __name__ == '__main__':
  absltest.main()
