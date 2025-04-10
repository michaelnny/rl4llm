# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
"""
The entry point of inference server. (SRT = SGLang Runtime)

This file implements HTTP APIs for the inference engine via fastapi.

We adapted the code to only keep essential features related to a RL training setup.
"""

import asyncio
import dataclasses
import logging
import multiprocessing as multiprocessing
import os
import threading
import time
from http import HTTPStatus
from typing import Callable, Dict, Optional

# Fix a bug of Python threading
setattr(threading, '_register_atexit', lambda *args, **kwargs: None)

from contextlib import asynccontextmanager

import numpy as np
import uvicorn
import uvloop
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, Response, StreamingResponse
from sglang.srt.entrypoints.engine import _launch_subprocesses
from sglang.srt.managers.io_struct import (
    EmbeddingReqInput,
    GenerateReqInput,
    InitWeightsUpdateGroupReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightFromDiskReqInput,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.metrics.func_timer import enable_func_timer
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    add_api_key_middleware,
    add_prometheus_middleware,
    set_uvicorn_logging_configs,
)
from sglang.srt.warmup import execute_warmups

logger = logging.getLogger(__name__)
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


# Store global states
@dataclasses.dataclass
class _GlobalState:
    tokenizer_manager: TokenizerManager
    scheduler_info: Dict


_global_state: Optional[_GlobalState] = None


def set_global_state(global_state: _GlobalState):
    global _global_state
    _global_state = global_state


@asynccontextmanager
async def lifespan(fast_api_app: FastAPI):
    server_args: ServerArgs = fast_api_app.server_args
    if server_args.warmups is not None:
        await execute_warmups(
            server_args.warmups.split(','), _global_state.tokenizer_manager
        )
        logger.info('Warmup ended')

    warmup_thread = getattr(fast_api_app, 'warmup_thread', None)
    if warmup_thread is not None:
        warmup_thread.start()
    yield


# Fast API
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

HEALTH_CHECK_TIMEOUT = int(os.getenv('SGLANG_HEALTH_CHECK_TIMEOUT', 20))


# Native APIs
@app.get('/health')
async def health() -> Response:
    """Check the health of the http server."""
    return Response(status_code=200)


@app.get('/health_generate')
async def health_generate(request: Request) -> Response:
    """Check the health of the inference server by generating one token."""

    sampling_params = {'max_new_tokens': 1, 'temperature': 0.0}
    rid = f"HEALTH_CHECK_{time.time()}"

    if _global_state.tokenizer_manager.is_image_gen:
        raise NotImplementedError()
    elif _global_state.tokenizer_manager.is_generation:
        gri = GenerateReqInput(
            rid=rid,
            input_ids=[0],
            sampling_params=sampling_params,
            log_metrics=False,
        )
    else:
        gri = EmbeddingReqInput(
            rid=rid,
            input_ids=[0],
            sampling_params=sampling_params,
            log_metrics=False,
        )

    async def gen():
        async for _ in _global_state.tokenizer_manager.generate_request(
            gri, request
        ):
            break

    tic = time.time()
    task = asyncio.create_task(gen())
    while time.time() < tic + HEALTH_CHECK_TIMEOUT:
        await asyncio.sleep(1)
        if _global_state.tokenizer_manager.last_receive_tstamp > tic:
            task.cancel()
            _global_state.tokenizer_manager.rid_to_state.pop(rid, None)
            return Response(status_code=200)

    task.cancel()
    tic_time = time.strftime('%H:%M:%S', time.localtime(tic))
    last_receive_time = time.strftime(
        '%H:%M:%S',
        time.localtime(_global_state.tokenizer_manager.last_receive_tstamp),
    )
    logger.error(
        f"Health check failed. Server couldn't get a response from detokenizer for last "
        f"{HEALTH_CHECK_TIMEOUT} seconds. tic start time: {tic_time}. "
        f"last_heartbeat time: {last_receive_time}"
    )
    _global_state.tokenizer_manager.rid_to_state.pop(rid, None)
    return Response(status_code=503)


# fastapi implicitly converts json in the request to obj (dataclass)
@app.post('/generate')
async def generate_request(obj: GenerateReqInput, request: Request):
    """Handle a generate request."""
    # remove stream support
    try:
        ret = await _global_state.tokenizer_manager.generate_request(
            obj, request
        ).__anext__()
        return ret
    except ValueError as e:
        logger.error(f"Error: {e}")
        return _create_error_response(e)


@app.post('/flush_cache')
async def flush_cache():
    """Flush the radix cache."""
    _global_state.tokenizer_manager.flush_cache()
    return Response(
        content='Cache flushed.\nPlease check backend logs for more details. '
        '(When there are running or waiting requests, the operation will not be performed.)\n',
        status_code=200,
    )


@app.post('/update_weights_from_disk')
async def update_weights_from_disk(
    obj: UpdateWeightFromDiskReqInput, request: Request
):
    """Update the weights from disk inplace without re-launching the server."""
    success, message, num_paused_requests = (
        await _global_state.tokenizer_manager.update_weights_from_disk(
            obj, request
        )
    )
    content = {
        'success': success,
        'message': message,
        'num_paused_requests': num_paused_requests,
    }
    if success:
        return ORJSONResponse(
            content,
            status_code=HTTPStatus.OK,
        )
    else:
        return ORJSONResponse(
            content,
            status_code=HTTPStatus.BAD_REQUEST,
        )


@app.post('/init_weights_update_group')
async def init_weights_update_group(
    obj: InitWeightsUpdateGroupReqInput, request: Request
):
    """Initialize the parameter update group."""
    success, message = (
        await _global_state.tokenizer_manager.init_weights_update_group(
            obj, request
        )
    )
    content = {'success': success, 'message': message}
    if success:
        return ORJSONResponse(content, status_code=200)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


# @app.post("/update_weights_from_tensor")
# async def update_weights_from_tensor(
#     obj: UpdateWeightsFromTensorReqInput, request: Request
# ):
#     """Update model parameter from tensors online."""
#     success, message = (
#         await _global_state.tokenizer_manager.update_weights_from_tensor(
#             obj, request
#         )
#     )
#     content = {"success": success, "message": message}
#     if success:
#         return ORJSONResponse(content, status_code=200)
#     else:
#         return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


@app.post('/release_memory_occupation')
async def release_memory_occupation(
    obj: ReleaseMemoryOccupationReqInput, request: Request
):
    """Release GPU memory occupation temporarily."""
    try:
        await _global_state.tokenizer_manager.release_memory_occupation(
            obj, request
        )
    except Exception as e:
        return _create_error_response(e)


@app.post('/resume_memory_occupation')
async def resume_memory_occupation(
    obj: ResumeMemoryOccupationReqInput, request: Request
):
    """Resume GPU memory occupation."""
    try:
        await _global_state.tokenizer_manager.resume_memory_occupation(
            obj, request
        )
    except Exception as e:
        return _create_error_response(e)


def _create_error_response(e):
    return ORJSONResponse(
        {'error': {'message': str(e)}}, status_code=HTTPStatus.BAD_REQUEST
    )


def launch_server(
    server_args: ServerArgs,
    pipe_finish_writer: Optional[multiprocessing.connection.Connection] = None,
    launch_callback: Optional[Callable[[], None]] = None,
):
    """
    Launch SRT (SGLang Runtime) Server.

    The SRT server consists of an HTTP server and an SRT engine.

    - HTTP server: A FastAPI server that routes requests to the engine.
    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server, Engine, and TokenizerManager both run in the main process.
    2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
    """
    tokenizer_manager, scheduler_info = _launch_subprocesses(
        server_args=server_args
    )
    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            scheduler_info=scheduler_info,
        )
    )

    # Add api key authorization
    if server_args.api_key:
        add_api_key_middleware(app, server_args.api_key)

    # Add prometheus middleware
    if server_args.enable_metrics:
        add_prometheus_middleware(app)
        enable_func_timer()

    try:
        # Update logging configs
        set_uvicorn_logging_configs()
        app.server_args = server_args
        # Listen for HTTP requests
        uvicorn.run(
            app,
            host=server_args.host,
            port=server_args.port,
            log_level=server_args.log_level_http or server_args.log_level,
            timeout_keep_alive=5,
            loop='uvloop',
        )
    finally:
        pass
