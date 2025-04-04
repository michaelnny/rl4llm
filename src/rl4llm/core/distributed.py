import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import torch
import torch.distributed as dist

from rl4llm.constants import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

# Generic Type Variable for object methods
T = TypeVar('T')


class DistributedManager:
    """
    Handles distributed training setup and communication utilities for both
    tensors and arbitrary Python objects.

    Initializes the default process group using environment variables
    (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT, LOCAL_RANK) unless
    explicit arguments are provided.
    """

    _instance = None  # Optional: For Singleton pattern if desired

    def __init__(
        self, backend: str = 'nccl', init_method: Optional[str] = None
    ):
        """
        Initializes the distributed environment.

        Args:
            backend (str): The distributed backend to use (e.g., 'nccl', 'gloo').
            init_method (Optional[str]): Optional URL specifying how to initialize
                the process group (e.g., 'env://'). If None, defaults to 'env://'.
        """
        if not dist.is_available():
            raise RuntimeError('Distributed training is not available.')

        if not dist.is_initialized():
            # Default to environment variable initialization if not specified
            if init_method is None:
                init_method = 'env://'
                # Ensure necessary env vars are set for 'env://'
                required_env = [
                    'RANK',
                    'WORLD_SIZE',
                    'MASTER_ADDR',
                    'MASTER_PORT',
                ]
                if not all(env in os.environ for env in required_env):
                    # Fallback for single-node, single-process execution
                    if (
                        'WORLD_SIZE' not in os.environ
                        or int(os.environ.get('WORLD_SIZE', 1)) == 1
                    ):
                        os.environ['MASTER_ADDR'] = '127.0.0.1'
                        os.environ['MASTER_PORT'] = (
                            '29500'  # Default port, can be randomized
                        )
                        os.environ['RANK'] = '0'
                        os.environ['WORLD_SIZE'] = '1'
                        logger.warning(
                            'Distributed environment variables not fully set. '
                            'Assuming single-process execution (rank 0, world size 1).'
                        )
                    else:
                        raise ValueError(
                            'Required environment variables for distributed training '
                            f"(RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT) are not set, "
                            f"and WORLD_SIZE ({os.environ.get('WORLD_SIZE')}) > 1."
                        )

            # Get rank and world size *before* init_process_group if possible
            # (some init methods might need them set externally)
            self.world_size = int(os.environ.get('WORLD_SIZE', 1))
            self.global_rank = int(os.environ.get('RANK', 0))
            self.local_rank = int(
                os.environ.get('LOCAL_RANK', self.global_rank)
            )  # Default local = global if not set

            # Add rank info to logger adapter for context
            self.logger = logging.LoggerAdapter(
                logger, {'rank': self.global_rank}
            )

            self.logger.info(
                f"Initializing process group with backend='{backend}', init_method='{init_method}'"
            )
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=self.world_size,
                rank=self.global_rank,
            )
            self.logger.info('Process group initialized.')

        else:
            # Already initialized, just retrieve info
            self.world_size = dist.get_world_size()
            self.global_rank = dist.get_rank()
            self.local_rank = int(
                os.environ.get('LOCAL_RANK', self.global_rank)
            )  # Assume local = global if not set
            self.logger = logging.LoggerAdapter(
                logger, {'rank': self.global_rank}
            )
            self.logger.info(
                'Process group already initialized. Attaching manager.'
            )

        # Set device based on local rank
        if (
            torch.cuda.is_available()
            and torch.cuda.device_count()
            >= self.world_size / int(os.environ.get('NODE_COUNT', 1))
        ):
            # Check if CUDA is available and there are enough GPUs
            if torch.cuda.device_count() > self.local_rank:
                self.device = torch.device('cuda', self.local_rank)
                torch.cuda.set_device(self.device)
                self.logger.info(f"Set device to {self.device}")
            else:
                self.logger.warning(
                    f"CUDA available but local_rank {self.local_rank} >= device_count {torch.cuda.device_count()}. Using CPU."
                )
                self.device = torch.device('cpu')
        else:
            self.logger.warning(
                'CUDA not available or not enough GPUs. Using CPU.'
            )
            self.device = torch.device('cpu')

        self.backend = dist.get_backend()
        self.logger.info(
            f"Distributed setup: WorldSize={self.world_size}, GlobalRank={self.global_rank}, "
            f"LocalRank={self.local_rank}, Device={self.device}, Backend='{self.backend}'"
        )

    # --- Process Group Info ---
    @property
    def is_master(self) -> bool:
        """Returns True if the current process is the master (rank 0)."""
        return self.global_rank == 0

    # --- Synchronization ---
    def barrier(self) -> None:
        """Synchronizes all processes. Blocks until all processes reach this point."""
        if self.world_size > 1:
            dist.barrier()
            self.logger.debug('Barrier synchronization complete.')

    # --- Tensor Communication ---
    def gather_tensor(
        self, tensor: torch.Tensor, dst: int = 0, concat_dim: Optional[int] = 0
    ) -> Optional[List[torch.Tensor]]:
        """
        Gathers tensors from all processes to a destination process (default rank 0).

        Args:
            tensor (torch.Tensor): Tensor to be sent from the current process.
                                   Must have the same shape across all processes.
            dst (int): Destination rank (default is 0).
            concat_dim (Optional[int]): If specified, concatenates the gathered tensors
                                       along this dimension on the destination rank.
                                       If None, returns a list of tensors.

        Returns:
            Optional[Union[torch.Tensor, List[torch.Tensor]]]:
                - On dst rank: A list of gathered tensors or a single concatenated tensor.
                - On other ranks: None.
                - If world_size is 1: The input tensor itself (or list containing it if concat_dim is None).
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(tensor)}")
        if self.world_size == 1:
            return (
                [tensor] if concat_dim is None else tensor
            )  # Match return type expectation

        tensor = tensor.contiguous().to(
            self.device if tensor.device.type != 'cpu' else 'cpu'
        )  # Ensure contiguous and correct device for comms

        gather_list = (
            [torch.zeros_like(tensor) for _ in range(self.world_size)]
            if self.global_rank == dst
            else None
        )

        dist.gather(tensor, gather_list=gather_list, dst=dst)

        if self.global_rank == dst:
            if concat_dim is not None:
                try:
                    return torch.cat(gather_list, dim=concat_dim)
                except Exception as e:
                    self.logger.error(
                        f"Failed to concatenate tensors with shapes {[t.shape for t in gather_list]} along dim {concat_dim}: {e}"
                    )
                    # Fallback to returning list on error
                    return gather_list
            else:
                return gather_list
        return None

    def all_gather_tensor(
        self, tensor: torch.Tensor, concat_dim: Optional[int] = 0
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Gathers tensors from all processes to all processes.

        Args:
            tensor (torch.Tensor): Tensor to be sent from the current process.
                                   Must have the same shape across all processes.
            concat_dim (Optional[int]): If specified, concatenates the gathered tensors
                                       along this dimension on all ranks.
                                       If None, returns a list of tensors on all ranks.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]:
                - A list of gathered tensors or a single concatenated tensor, available on all ranks.
                - If world_size is 1: The input tensor itself (or list containing it if concat_dim is None).
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(tensor)}")
        if self.world_size == 1:
            return (
                [tensor] if concat_dim is None else tensor
            )  # Match return type expectation

        tensor = tensor.contiguous().to(
            self.device if tensor.device.type != 'cpu' else 'cpu'
        )  # Ensure contiguous and correct device

        gathered_tensors = [
            torch.zeros_like(tensor) for _ in range(self.world_size)
        ]
        dist.all_gather(gathered_tensors, tensor)

        if concat_dim is not None:
            try:
                return torch.cat(gathered_tensors, dim=concat_dim)
            except Exception as e:
                self.logger.error(
                    f"Failed to concatenate tensors with shapes {[t.shape for t in gathered_tensors]} along dim {concat_dim}: {e}"
                )
                # Fallback to returning list on error
                return gathered_tensors
        else:
            return gathered_tensors

    def reduce_tensor(
        self, tensor: torch.Tensor, dst: int = 0, op=dist.ReduceOp.SUM
    ) -> Optional[torch.Tensor]:
        """
        Reduces tensor data across all machines (result only on dst rank).

        Args:
            tensor (torch.Tensor): Tensor to be reduced. Must have same shape/type
                                   across processes. Input tensor is modified in-place
                                   on the dst rank.
            dst (int): Destination rank (default is 0).
            op (dist.ReduceOp): Reduction operation (e.g., SUM, AVG, MAX, MIN).

        Returns:
            Optional[torch.Tensor]: The reduced tensor on the dst rank, None otherwise.
                                    If world_size is 1, returns the input tensor.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(tensor)}")
        if self.world_size == 1:
            return tensor

        tensor = tensor.contiguous().to(
            self.device if tensor.device.type != 'cpu' else 'cpu'
        )  # Ensure correct device

        dist.reduce(tensor, dst=dst, op=op)
        self.logger.debug(f"Tensor reduced to rank {dst} with op {op}.")
        return tensor if self.global_rank == dst else None

    def all_reduce_tensor(
        self, tensor: torch.Tensor, op=dist.ReduceOp.SUM
    ) -> torch.Tensor:
        """
        Reduces tensor data across all machines (result available on all ranks).

        Args:
            tensor (torch.Tensor): Tensor to be reduced. Must have same shape/type
                                   across processes. Tensor is modified in-place.
            op (dist.ReduceOp): Reduction operation (e.g., SUM, AVG, MAX, MIN).

        Returns:
            torch.Tensor: The reduced tensor, available on all ranks. Tensor is
                          modified in-place.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(tensor)}")
        if self.world_size == 1:
            return tensor

        tensor = tensor.contiguous().to(
            self.device if tensor.device.type != 'cpu' else 'cpu'
        )  # Ensure correct device

        dist.all_reduce(tensor, op=op)
        self.logger.debug(f"Tensor all-reduced with op {op}.")
        return tensor

    def broadcast_tensor(
        self, tensor: torch.Tensor, src: int = 0
    ) -> torch.Tensor:
        """
        Broadcasts a tensor from the source rank to all other processes.

        Args:
            tensor (torch.Tensor): Tensor to be broadcasted. On src rank, this is the
                                   tensor to send. On other ranks, its shape/type/device
                                   determine the received tensor properties. The tensor
                                   will be modified in-place on non-src ranks.
            src (int): Source rank from which to broadcast (default is 0).

        Returns:
            torch.Tensor: The broadcasted tensor, available on all ranks.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(tensor)}")
        if self.world_size == 1:
            return tensor

        # Ensure tensor is on the correct device, especially for non-src ranks
        # where it acts as a buffer
        tensor = tensor.contiguous().to(
            self.device if tensor.device.type != 'cpu' else 'cpu'
        )

        dist.broadcast(tensor, src=src)
        self.logger.debug(f"Tensor broadcasted from rank {src}.")
        return tensor

    def gather_object(self, obj: T, dst: int = 0) -> Optional[List[T]]:
        """
        Gathers picklable Python objects from all processes to the destination rank.

        Args:
            obj (T): The picklable object to send from the current process.
            dst (int): Destination rank (default is 0).

        Returns:
            Optional[List[T]]: A list containing the objects from all ranks in rank
                               order, available only on the `dst` rank. Returns
                               `None` on non-destination ranks. Returns `[obj]`
                               if world_size is 1.
        """
        if self.world_size == 1:
            return [obj]

        # Use pickle implicitly via dist.gather_object
        output_objects = (
            [None for _ in range(self.world_size)]
            if self.global_rank == dst
            else None
        )
        dist.gather_object(obj, object_gather_list=output_objects, dst=dst)
        self.logger.debug(f"Objects gathered at rank {dst}.")
        return output_objects  # type: ignore # Correctly typed based on runtime check

    def all_gather_object(self, obj: T) -> List[T]:
        """
        Gathers picklable Python objects from all processes to all processes.

        Args:
            obj (T): The picklable object to send from the current process.

        Returns:
            List[T]: A list containing the objects from all ranks in rank order,
                     available on all ranks. Returns `[obj]` if world_size is 1.
        """
        if self.world_size == 1:
            return [obj]

        output_objects = [None for _ in range(self.world_size)]
        dist.all_gather_object(output_objects, obj)
        self.logger.debug('Objects all-gathered.')
        return output_objects  # type: ignore # Correctly typed based on runtime check

    def broadcast_object(self, obj: T, src: int = 0) -> T:
        """
        Broadcasts a picklable Python object from the source rank to all other processes.

        Args:
            obj (T): The picklable object to broadcast *on the src rank*. On non-src
                     ranks, this argument is ignored, but the function must still be called.
            src (int): Source rank from which to broadcast (default is 0).

        Returns:
            T: The broadcasted object, available on all ranks.
        """
        if self.world_size == 1:
            return obj

        # broadcast_object_list expects a list. Wrap the object in a list.
        if self.global_rank == src:
            obj_list = [obj]
        else:
            # Non-source ranks need a placeholder list of the correct size
            # (will be overwritten)
            obj_list = [None]

        dist.broadcast_object_list(obj_list, src=src)
        self.logger.debug(f"Object broadcasted from rank {src}.")
        return obj_list[0]  # type: ignore # Correctly typed based on runtime check

    def scatter_object(
        self, scatter_list: Optional[List[T]], src: int = 0
    ) -> T:
        """
        Scatters a list of picklable Python objects from the source rank to all processes.
        Each process receives one object from the list based on its rank.

        Args:
            scatter_list (Optional[List[T]]): A list of picklable objects *on the src rank*.
                                            The length of the list must equal the world size.
                                            On non-src ranks, this argument should be None or [].
            src (int): Source rank from which to scatter (default is 0).

        Returns:
            T: The specific object from the `scatter_list` intended for the current rank.

        Raises:
            ValueError: If `scatter_list` is not provided on the src rank or its
                        length does not match the world size.
            TypeError: If elements in scatter_list are not picklable.
        """
        if self.world_size == 1:
            if self.is_master:
                if scatter_list and len(scatter_list) == 1:
                    return scatter_list[0]
                else:
                    raise ValueError(
                        'On rank 0 with world_size=1, scatter_list must be a list with one element.'
                    )
            else:
                # This case shouldn't happen in correct setup but handle defensively
                raise RuntimeError(
                    'scatter_object called on non-master rank with world_size=1'
                )

        # Prepare arguments for dist.scatter_object_list
        output_obj_list = [
            None
        ]  # Placeholder for the received object on this rank
        input_list = scatter_list if self.global_rank == src else None

        # Validation on source rank
        if self.global_rank == src:
            if not isinstance(input_list, list):
                raise ValueError(
                    f"scatter_list must be a list on the source rank {src}, got {type(input_list)}"
                )
            if len(input_list) != self.world_size:
                raise ValueError(
                    f"scatter_list length ({len(input_list)}) must equal world_size ({self.world_size}) on source rank {src}"
                )

        # Perform the scatter operation
        dist.scatter_object_list(
            scatter_object_output_list=output_obj_list,
            scatter_object_input_list=input_list,  # type: ignore # PyTorch expects Optional[List] here
            src=src,
        )
        self.logger.debug(f"Object scattered from rank {src}.")

        # Return the received object (it's placed in the first element of output_obj_list)
        return output_obj_list[0]  # type: ignore # Correctly typed based on runtime check

    def synchronize(self) -> None:
        """Calls torch.cuda.synchronize."""
        torch.cuda.synchronize(self.device)

    @classmethod
    def get_instance(cls, **kwargs) -> 'DistributedManager':
        """Gets the singleton instance of the DistributedManager."""
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    def teardown(self) -> None:
        """Cleans up the distributed process group."""
        if dist.is_initialized():
            self.logger.info('Destroying process group.')
            dist.destroy_process_group()
            DistributedManager._instance = (
                None  # Clear singleton instance if used
            )


# --- Example Usage ---
if __name__ == '__main__':
    # This example assumes you run it using torchrun or similar:
    # torchrun --nproc_per_node=2 your_script_name.py

    # Initialize the manager (it handles init_process_group)
    try:
        # Using default 'env://' init method
        manager = DistributedManager(
            backend='nccl' if torch.cuda.is_available() else 'gloo'
        )

        # === Tensor Example ===
        # Create a tensor unique to each rank but on the correct device
        my_tensor = torch.ones(2, 2, device=manager.device) * (
            manager.global_rank + 1
        )
        manager.logger.info(f"Initial tensor: \n{my_tensor}")
        manager.barrier()

        # All-gather tensors (concatenate along dim 0)
        all_tensors = manager.all_gather_tensor(my_tensor, concat_dim=0)
        manager.logger.info(
            f"All-gathered tensor (concatenated): \n{all_tensors}"
        )
        manager.barrier()

        # All-gather tensors (as a list)
        all_tensors_list = manager.all_gather_tensor(my_tensor, concat_dim=None)
        if (
            all_tensors_list
        ):  # Check as it might return tensor directly if concat_dim=0 and exception occurs
            manager.logger.info(
                f"All-gathered tensor (list): {[t.shape for t in all_tensors_list]}"
            )
        manager.barrier()

        # Broadcast a tensor from rank 0
        if manager.is_master:
            broadcast_data = torch.tensor([10.0, 20.0], device=manager.device)
        else:
            # Non-src ranks need a buffer tensor with the correct shape/dtype/device
            broadcast_data = torch.empty(
                2, dtype=torch.float32, device=manager.device
            )

        received_broadcast_tensor = manager.broadcast_tensor(
            broadcast_data, src=0
        )
        manager.logger.info(
            f"Received broadcast tensor: {received_broadcast_tensor}"
        )
        manager.barrier()

        # === Object Example (Model Weights Dictionary) ===

        # 1. Create different "weights" on each rank
        my_weights = {
            'layer1.weight': torch.rand(4, 4) * (manager.global_rank + 1),
            'layer1.bias': torch.zeros(4) + manager.global_rank,
            'metadata': f"Data from rank {manager.global_rank}",
        }
        manager.logger.info(
            f"Initial weights keys: {list(my_weights.keys())}, metadata: {my_weights['metadata']}"
        )
        manager.barrier()

        # 2. Broadcast weights from rank 0 to all others
        if manager.is_master:
            weights_to_broadcast = my_weights  # Use rank 0's weights
        else:
            weights_to_broadcast = None  # Only needed on src rank

        broadcasted_weights = manager.broadcast_object(
            weights_to_broadcast, src=0
        )
        # Verify received weights
        manager.logger.info(
            f"Received broadcasted weights. Metadata: '{broadcasted_weights['metadata']}', "
            f"layer1.weight mean: {broadcasted_weights['layer1.weight'].mean():.4f}"
        )
        manager.barrier()

        # 3. Gather objects (e.g., metrics) to rank 0
        my_metric = {
            'loss': torch.rand(1).item() * (manager.global_rank + 1),
            'rank': manager.global_rank,
        }
        gathered_metrics = manager.gather_object(my_metric, dst=0)

        if manager.is_master:
            manager.logger.info(
                f"Gathered metrics on rank 0: {gathered_metrics}"
            )
        else:
            manager.logger.info(
                f"Gathered metrics on rank {manager.global_rank}: {gathered_metrics} (should be None)"
            )
        manager.barrier()

        # 4. All-gather objects
        my_info = f"Info from {manager.global_rank}"
        all_gathered_info = manager.all_gather_object(my_info)
        manager.logger.info(f"All gathered info: {all_gathered_info}")
        manager.barrier()

        # 5. Scatter objects (e.g., configurations) from rank 0
        scatter_data = None
        if manager.is_master:
            scatter_data = [
                f"Config for rank {r}" for r in range(manager.world_size)
            ]

        my_config = manager.scatter_object(scatter_data, src=0)
        manager.logger.info(f"Received scattered object: {my_config}")
        manager.barrier()

        manager.logger.info('Example finished successfully.')

    except Exception as e:
        # Use root logger here as manager logger might not be initialized if __init__ failed
        logging.exception(
            f"An error occurred during distributed execution: {e}"
        )  # Log traceback

    finally:
        # Ensure cleanup happens even if errors occur
        if 'manager' in locals() and isinstance(manager, DistributedManager):
            manager.teardown()
