import os

import pytest
import torch
import torch.distributed as dist

from rl4llm.core.distributed import DistributedManager


# A fixture to clear environment variables between tests
@pytest.fixture(autouse=True)
def reset_env(monkeypatch):
    for var in [
        'RANK',
        'WORLD_SIZE',
        'MASTER_ADDR',
        'MASTER_PORT',
        'LOCAL_RANK',
    ]:
        monkeypatch.delenv(var, raising=False)
    yield


# A fixture to monkey-patch the torch.distributed API with fake implementations
@pytest.fixture
def fake_dist(monkeypatch):
    # Create a state dictionary to simulate the distributed process group state.
    fake_state = {'initialized': False, 'calls': []}

    # Ensure distributed is available.
    monkeypatch.setattr(dist, 'is_available', lambda: True)

    # Patch is_initialized to use our fake state.
    monkeypatch.setattr(
        dist, 'is_initialized', lambda: fake_state['initialized']
    )

    # Patch init_process_group to update our state.
    def fake_init_process_group(*args, **kwargs):
        fake_state['initialized'] = True
        fake_state['calls'].append('init_process_group')

    monkeypatch.setattr(dist, 'init_process_group', fake_init_process_group)

    # Patch destroy_process_group to update our state.
    def fake_destroy_process_group():
        fake_state['initialized'] = False
        fake_state['calls'].append('destroy_process_group')

    monkeypatch.setattr(
        dist, 'destroy_process_group', fake_destroy_process_group
    )

    # Return the fake state for inspection in tests.
    monkeypatch.setattr(
        dist, 'get_rank', lambda: int(os.environ.get('RANK', 0))
    )
    monkeypatch.setattr(
        dist, 'get_world_size', lambda: int(os.environ.get('WORLD_SIZE', 1))
    )
    monkeypatch.setattr(dist, 'get_backend', lambda: 'gloo')
    monkeypatch.setattr(
        dist, 'barrier', lambda: fake_state['calls'].append('barrier')
    )

    # Patch tensor communication functions
    monkeypatch.setattr(
        dist,
        'gather',
        lambda tensor, gather_list, dst: [
            gather_list.__setitem__(i, tensor.clone())
            for i in range(len(gather_list))
        ],
    )
    monkeypatch.setattr(
        dist,
        'all_gather',
        lambda gathered_tensors, tensor: [
            gathered_tensors.__setitem__(i, tensor.clone())
            for i in range(len(gathered_tensors))
        ],
    )
    monkeypatch.setattr(dist, 'reduce', lambda tensor, dst, op: None)
    monkeypatch.setattr(dist, 'all_reduce', lambda tensor, op: None)
    monkeypatch.setattr(dist, 'broadcast', lambda tensor, src: None)

    # Object-based communication functions.
    monkeypatch.setattr(
        dist,
        'gather_object',
        lambda obj, object_gather_list, dst: (
            object_gather_list.__setitem__(0, obj)
            if object_gather_list is not None
            else None
        ),
    )
    monkeypatch.setattr(
        dist,
        'all_gather_object',
        lambda output_objects, obj: [
            output_objects.__setitem__(i, obj)
            for i in range(len(output_objects))
        ],
    )
    monkeypatch.setattr(
        dist, 'broadcast_object_list', lambda obj_list, src: None
    )
    monkeypatch.setattr(
        dist,
        'scatter_object_list',
        lambda scatter_object_output_list, scatter_object_input_list, src: scatter_object_output_list.__setitem__(
            0,
            scatter_object_input_list[0] if scatter_object_input_list else None,
        ),
    )

    return fake_state


# Fixture to simulate a CPU-only environment.
@pytest.fixture(autouse=True)
def fake_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 0)
    return


def test_init_single_process(monkeypatch, fake_dist):
    # Simulate single process (WORLD_SIZE=1); no need for MASTER_ADDR/MASTER_PORT.
    os.environ['WORLD_SIZE'] = '1'
    dm = DistributedManager()
    assert dm.world_size == 1
    assert dm.global_rank == 0
    # Should use CPU device.
    assert dm.device.type == 'cpu'

    tensor = torch.tensor([1, 2, 3])
    # If concat_dim is None, gather_tensor returns a list.
    gathered = dm.gather_tensor(tensor, concat_dim=None)
    assert isinstance(gathered, list)
    assert gathered == [tensor]
    # With concat_dim specified, it returns the tensor itself.
    gathered_cat = dm.gather_tensor(tensor, concat_dim=0)
    assert torch.equal(gathered_cat, tensor)


def test_is_master(monkeypatch, fake_dist):
    # For multi-process tests, set MASTER_ADDR and MASTER_PORT.
    os.environ['WORLD_SIZE'] = '2'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = '0'
    dm_master = DistributedManager()
    assert dm_master.is_master

    os.environ['RANK'] = '1'
    dm_nonmaster = DistributedManager()
    assert not dm_nonmaster.is_master


def test_gather_tensor(monkeypatch, fake_dist):
    # Simulate multi-process: WORLD_SIZE=2 on rank 0.
    os.environ['WORLD_SIZE'] = '2'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = '0'
    dm = DistributedManager()
    tensor = torch.tensor([1])

    # Patch gather so that every slot gets a copy of the input tensor.
    def fake_gather(tensor, gather_list, dst):
        for i in range(len(gather_list)):
            gather_list[i].copy_(tensor)

    monkeypatch.setattr(dist, 'gather', fake_gather)
    # With concat_dim provided, the gathered tensors should be concatenated.
    gathered_cat = dm.gather_tensor(tensor, dst=0, concat_dim=0)
    expected = torch.cat([tensor, tensor], dim=0)
    assert torch.equal(gathered_cat, expected)


def test_all_reduce_tensor(monkeypatch, fake_dist):
    # WORLD_SIZE=2; testing all_reduce_tensor.
    os.environ['WORLD_SIZE'] = '2'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = '0'
    dm = DistributedManager()
    tensor = torch.tensor([1.0])
    ret = dm.all_reduce_tensor(tensor, op=dist.ReduceOp.SUM)
    # Our fake all_reduce is a no-op so the tensor remains unchanged.
    assert torch.equal(ret, tensor)


def test_broadcast_tensor(monkeypatch, fake_dist):
    # Simulate broadcast: WORLD_SIZE=2, testing on a non-source rank.
    os.environ['WORLD_SIZE'] = '2'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = '1'
    dm = DistributedManager()
    tensor = torch.tensor([0])

    # Fake broadcast that sets the tensor value to 42.
    def fake_broadcast(tensor, src):
        tensor.fill_(42)

    monkeypatch.setattr(dist, 'broadcast', fake_broadcast)
    ret = dm.broadcast_tensor(tensor, src=0)
    assert torch.equal(ret, torch.tensor([42]))


def test_gather_object(monkeypatch, fake_dist):
    # Test gather_object for WORLD_SIZE=2 on rank 0.
    os.environ['WORLD_SIZE'] = '2'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = '0'
    dm = DistributedManager()
    obj = {'key': 'value'}
    gathered = dm.gather_object(obj, dst=0)
    # Our fake gather_object simply places the object in index 0.
    assert gathered[0] == obj


def test_all_gather_object(monkeypatch, fake_dist):
    # Test all_gather_object for WORLD_SIZE=2.
    os.environ['WORLD_SIZE'] = '2'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = '0'
    dm = DistributedManager()
    obj = 'test'
    gathered = dm.all_gather_object(obj)
    # Fake all_gather_object fills each index with the object.
    assert gathered == ['test', 'test']


def test_broadcast_object(monkeypatch, fake_dist):
    # Test broadcast_object for WORLD_SIZE=2.
    os.environ['WORLD_SIZE'] = '2'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    # Test from source rank.
    os.environ['RANK'] = '0'
    dm_src = DistributedManager()
    obj = 123
    broadcasted = dm_src.broadcast_object(obj, src=0)
    assert broadcasted == 123

    # Test on non-source rank.
    os.environ['RANK'] = '1'
    dm_non = DistributedManager()
    # Our fake broadcast_object_list does nothing so the placeholder remains None.
    broadcasted_non = dm_non.broadcast_object(456, src=0)
    assert broadcasted_non is None


def test_scatter_object(monkeypatch, fake_dist):
    # Simulate multi-process scatter (WORLD_SIZE=2).
    os.environ['WORLD_SIZE'] = '2'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    # For non-source rank, scatter_list is ignored.
    os.environ['RANK'] = '1'
    dm_non = DistributedManager()
    scattered_non = dm_non.scatter_object(None, src=0)
    # Our fake scatter sets the output to None.
    assert scattered_non is None

    # For source rank in a single-process case.
    os.environ['WORLD_SIZE'] = '1'
    os.environ['RANK'] = '0'
    dm_single = DistributedManager()
    # If scatter_list is not a list with one element, ValueError is raised.
    with pytest.raises(ValueError):
        dm_single.scatter_object([], src=0)
    ret = dm_single.scatter_object([123], src=0)
    assert ret == 123


def test_teardown_and_singleton(monkeypatch, fake_dist):
    os.environ['WORLD_SIZE'] = '1'
    os.environ['RANK'] = '0'
    dm1 = DistributedManager.get_instance()
    dm2 = DistributedManager.get_instance()
    assert dm1 is dm2

    dm1.teardown()
    assert 'destroy_process_group' in fake_dist['calls']

    dm3 = DistributedManager.get_instance()
    assert dm1 is not dm3
