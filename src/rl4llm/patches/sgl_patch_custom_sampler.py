# --------------  sgl_patch_custom_sampler.py  -----------------
"""
Apply exploration / token‑replacement for SGLang.
Patch is applied only when   apply_patch_explore_sampler(**cfg)  is called.

apply_patch_explore_sampler(replace_prob=0.1, ...)

"""

import json
import os
import random
from importlib import import_module
from typing import List

import torch
import torch.distributed as dist

# ---------- default configuration (may be overridden) ----------------
_DEFAULTS = dict(
    # —— exploration ——————————————————————————————
    explore_steps=0,
    explore_skip=0,
    explore_top_k=100,
    explore_decay=0.8,
    # —— replacement ——————————————————————————————
    replace_prob=0.0,
    replace_max=0,
    replace_source_tokens=[32021],
    replace_target_tokens=[852, 1932],
    replace_prevent_patterns=[],
)


# # Optional correctness fn
# def correctness_callback(text: str) -> float:
#     """Return <1.0 if sequence is 'incorrect'; 1.0 if OK."""
#     return 0.0


# ---------------------------------------------------------------------


# ------ helpers ------------------------------------------------------
def _uniform_topk_sample(row: torch.Tensor, k: int) -> int:
    k = min(k, row.size(0))
    _, idx = torch.topk(row, k)
    return idx[random.randrange(k)].item()


def _patterns_exist(seq: List[int], patterns: List[List[int]]) -> bool:
    return any(
        len(seq) >= len(pat) and seq[-len(pat) :] == pat for pat in patterns
    )


# ------ the real patch ----------------------------------------------
def apply_patch_explore_sampler(**user_cfg):
    """Call once *before* the first request. Accepts any of the *_DEFAULTS keys."""

    # merge precedence: ENV‑VAR  >  kwargs  >  _DEFAULTS
    env_cfg = {}
    if 'SG_CUSTOM_SAMPLER_CFG' in os.environ:
        try:
            env_cfg = json.loads(os.environ['SG_CUSTOM_SAMPLER_CFG'])
        except Exception as e:
            raise ValueError(f"Bad SG_CUSTOM_SAMPLER_CFG: {e}")

    BASE_CFG = {**_DEFAULTS, **user_cfg, **env_cfg}

    smod = import_module('sglang.srt.sampling.sampler')
    orig_forward = smod.Sampler.forward
    if getattr(smod.Sampler, '_patched_by_custom', False):
        # already patched
        return

    def patched_forward(
        self,
        logits_out,
        samp_info,
        return_logprob,
        top_logprobs_nums,
        token_ids_logprobs,
    ):

        next_ids = orig_forward(
            self,
            logits_out,
            samp_info,
            return_logprob,
            top_logprobs_nums,
            token_ids_logprobs,
        )
        batch = next_ids.size(0)
        logits = logits_out.next_token_logits
        temps = samp_info.temperatures

        for i in range(batch):

            # ---------- per‑request toggle & overrides ----------------
            cp = samp_info.custom_params[i] or {}
            if not cp.get('enable_custom', False):
                continue

            cfg = {
                **BASE_CFG,
                **{k: v for k, v in cp.items() if k in _DEFAULTS},
            }

            st = cp.setdefault('_state', {'step': 0, 'hist': [], 'repl_cnt': 0})
            step, hist, repl = st['step'], st['hist'], st['repl_cnt']
            tok_id = next_ids[i].item()

            # —— exploration —————————————————————————
            exp_active = (
                cfg['explore_steps'] > 0
                and step >= cfg['explore_skip']
                and step < cfg['explore_skip'] + cfg['explore_steps']
            )
            if exp_active:
                eff = step - cfg['explore_skip']
                cur_k = max(
                    2, int(cfg['explore_top_k'] * (cfg['explore_decay'] ** eff))
                )
                tok_id = _uniform_topk_sample(logits[i] / temps[i], cur_k)

            # —— replacement ——————————————————————————
            if (
                tok_id in cfg['replace_source_tokens']
                and repl < cfg['replace_max']
                and not _patterns_exist(hist, cfg['replace_prevent_patterns'])
                # and (
                #     correctness_callback is None
                #     or correctness_callback(
                #         samp_info.tokenizer.decode(hist + [tok_id])
                #     )
                #     < 1.0
                # )
                and random.random() < cfg['replace_prob']
            ):

                tok_id = random.choice(cfg['replace_target_tokens'])
                repl += 1
                st['repl_cnt'] = repl

            # —— commit & update state ————————————
            next_ids[i] = tok_id
            hist.append(tok_id)
            st['step'] = step + 1

        # keep upstream TP‑sync
        if smod.SYNC_TOKEN_IDS_ACROSS_TP or samp_info.grammars:
            dist.all_reduce(
                next_ids, op=dist.ReduceOp.MIN, group=self.tp_sync_group
            )
        return next_ids

    smod.Sampler.forward = patched_forward
    smod.Sampler._patched_by_custom = True
    print('[custom‑sampler] Sampler.forward patched with cfg =', BASE_CFG)


# # -------- auto‑apply if env‑var present ------------------------------
# if "SG_CUSTOM_SAMPLER_CFG" in os.environ:
#     apply_patch_explore_sampler()
