import os

import pytest
import torch
import torch.distributed as dist

from rl4llm.core.distributed import DistributedOps


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


@pytest.fixture
def fake_dist(monkeypatch):
    fake_state = {'initialized': False, 'calls': []}
    monkeypatch.setattr(dist, 'is_available', lambda: True)
    monkeypatch.setattr(
        dist, 'is_initialized', lambda: fake_state['initialized']
    )

    def fake_init_process_group(*args, **kwargs):
        fake_state['initialized'] = True
        fake_state['calls'].append('init_process_group')

    monkeypatch.setattr(dist, 'init_process_group', fake_init_process_group)

    def fake_destroy_process_group():
        fake_state['initialized'] = False
        fake_state['calls'].append('destroy_process_group')

    monkeypatch.setattr(
        dist, 'destroy_process_group', fake_destroy_process_group
    )
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


@pytest.fixture(autouse=True)
def fake_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 0)


def test_init_single_process(fake_dist):
    """Tests DistributedOps initialization in a single-process environment."""
    os.environ['WORLD_SIZE'] = '1'
    dm = DistributedOps()
    assert dm.world_size == 1
    assert dm.global_rank == 0
    assert dm.device.type == 'cpu'
    tensor = torch.tensor([1, 2, 3])
    gathered = dm.gather_tensor(tensor, concat_dim=None)
    assert isinstance(gathered, list)
    assert gathered == [tensor]
    gathered_cat = dm.gather_tensor(tensor, concat_dim=0)
    assert torch.equal(gathered_cat, tensor)


@pytest.mark.parametrize('rank, expected', [('0', True), ('1', False)])
def test_is_master(fake_dist, rank, expected):
    """Tests if DistributedOps correctly identifies the master process."""
    os.environ['WORLD_SIZE'] = '2'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = rank
    dm = DistributedOps()
    assert dm.is_master == expected


def test_gather_tensor(fake_dist, monkeypatch):
    """Tests gathering tensors across processes with concatenation."""
    os.environ['WORLD_SIZE'] = '2'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = '0'
    dm = DistributedOps()
    tensor = torch.tensor([1])

    def fake_gather(tensor, gather_list, dst):
        for i in range(len(gather_list)):
            gather_list[i].copy_(tensor)

    monkeypatch.setattr(dist, 'gather', fake_gather)
    gathered_cat = dm.gather_tensor(tensor, dst=0, concat_dim=0)
    expected = torch.cat([tensor, tensor], dim=0)
    assert torch.equal(gathered_cat, expected)


def test_all_reduce_tensor(fake_dist):
    """Tests all_reduce operation on a tensor across processes."""
    os.environ['WORLD_SIZE'] = '2'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = '0'
    dm = DistributedOps()
    tensor = torch.tensor([1.0])
    ret = dm.all_reduce_tensor(tensor, op=dist.ReduceOp.SUM)
    assert torch.equal(ret, tensor)


def test_broadcast_tensor(fake_dist, monkeypatch):
    """Tests broadcasting a tensor from a source rank."""
    os.environ['WORLD_SIZE'] = '2'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = '1'
    dm = DistributedOps()
    tensor = torch.tensor([0])

    def fake_broadcast(tensor, src):
        tensor.fill_(42)

    monkeypatch.setattr(dist, 'broadcast', fake_broadcast)
    ret = dm.broadcast_tensor(tensor, src=0)
    assert torch.equal(ret, torch.tensor([42]))


def test_gather_object(fake_dist):
    """Tests gathering objects across processes."""
    os.environ['WORLD_SIZE'] = '2'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = '0'
    dm = DistributedOps()
    obj = {'key': 'value'}
    gathered = dm.gather_object(obj, dst=0)
    assert gathered[0] == obj


def test_all_gather_object(fake_dist):
    """Tests all_gather operation for objects across processes."""
    os.environ['WORLD_SIZE'] = '2'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = '0'
    dm = DistributedOps()
    obj = 'test'
    gathered = dm.all_gather_object(obj)
    assert gathered == ['test', 'test']


@pytest.mark.parametrize(
    'rank, input_obj, expected', [('0', 123, 123), ('1', 456, None)]
)
def test_broadcast_object(fake_dist, rank, input_obj, expected):
    """Tests broadcasting an object from a source rank."""
    os.environ['WORLD_SIZE'] = '2'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = rank
    dm = DistributedOps()
    broadcasted = dm.broadcast_object(input_obj, src=0)
    assert broadcasted == expected


@pytest.mark.parametrize(
    'world_size, rank, scatter_list, expected',
    [
        ('2', '1', None, None),
        ('1', '0', [123], 123),
    ],
)
def test_scatter_object(fake_dist, world_size, rank, scatter_list, expected):
    """Tests scattering objects across processes."""
    os.environ['WORLD_SIZE'] = world_size
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = rank
    dm = DistributedOps()
    if world_size == '1' and not scatter_list:
        with pytest.raises(ValueError):
            dm.scatter_object([], src=0)
    else:
        scattered = dm.scatter_object(scatter_list, src=0)
        assert scattered == expected


def test_teardown_and_singleton(fake_dist):
    """Tests DistributedOps singleton behavior and teardown."""
    os.environ['WORLD_SIZE'] = '1'
    os.environ['RANK'] = '0'
    dm1 = DistributedOps.get_instance()
    dm2 = DistributedOps.get_instance()
    assert dm1 is dm2
    dm1.teardown()
    assert 'destroy_process_group' in fake_dist['calls']
    dm3 = DistributedOps.get_instance()
    assert dm1 is not dm3
