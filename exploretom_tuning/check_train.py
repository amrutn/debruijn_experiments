"""
Isolate a non-finite training loss on this machine before the experiment runs.

Runs the first few micro-batches of the real training path -- same data,
model, adapter shape, optimizer and gradient checkpointing as
`train_eval.train_adapter` -- once per attention setting, and reports the loss
of every micro-batch and the gradient norm of every optimizer step. A NaN that
appears under one attention kernel and not another is that kernel's; one that
appears everywhere except in float32 is a precision problem; one that
appears everywhere is the data or the code.

The settings tried, in order: the run's default (sdpa, whichever kernels torch
picks), sdpa restricted to each single kernel (math, efficient, flash, cudnn;
an unsupported one is reported as such), eager attention, and sdpa in
float32 as the reference. Each takes a model load plus a few steps: about a
minute per setting on a GPU. Whatever passes can be selected for the
experiment with ``--attn`` / ``--sdpa-backends`` of `run_experiments`.

    python check_train.py                          # first GPU, state-tracking targets
    python check_train.py --device cuda:0 --steps 2 --cond chain
    python check_train.py --only default,eager     # a subset of the settings
"""

import sys
import math
import time
import random
import argparse
import platform

import torch

from exploretom_data import DataConfig
from traces import TraceConfig, TEACHER
from train_eval import (
    ModelConfig, TrainConfig, encode_example, collate, training_pool,
    sdpa_context, SDPA_BACKENDS, BASE_MODELS,
)

SETTINGS = ['default'] + [f'sdpa:{b}' for b in SDPA_BACKENDS] + ['eager', 'float32']


def environment(device):
    """Versions and the attention kernels torch has enabled, for the report."""
    import transformers
    import peft
    lines = [f'python {platform.python_version()}  torch {torch.__version__}  '
             f'transformers {transformers.__version__}  peft {peft.__version__}']
    if device.startswith('cuda') and torch.cuda.is_available():
        i = torch.device(device).index or 0
        lines.append(f'{torch.cuda.get_device_name(i)}  cuda {torch.version.cuda}  '
                     f'cudnn {torch.backends.cudnn.version()}  capability '
                     f'{".".join(map(str, torch.cuda.get_device_capability(i)))}')
        b = torch.backends.cuda
        flags = {n: getattr(b, f'{n}_sdp_enabled')() for n in ('flash', 'mem_efficient', 'math')
                 if hasattr(b, f'{n}_sdp_enabled')}
        if hasattr(b, 'cudnn_sdp_enabled'):
            flags['cudnn'] = b.cudnn_sdp_enabled()
        lines.append('sdpa kernels enabled: ' + ', '.join(f'{k}={v}' for k, v in flags.items()))
    return lines


def build_batches(cond, dcfg, trcfg, tok, tcfg, n_micro):
    """The first `n_micro` micro-batches of the run at seed `tcfg.seed`."""
    examples = training_pool(cond, dcfg, trcfg)
    data = []
    for prob, target in examples:
        e = encode_example(tok, prob, target, tcfg.max_seq_len)
        if e is not None:
            data.append(e)
    order = list(range(len(data)))
    random.Random(tcfg.seed).shuffle(order)
    out = []
    for m in range(n_micro):
        idx = order[m * tcfg.batch_size:(m + 1) * tcfg.batch_size]
        out.append(collate([data[i] for i in idx], tok.pad_token_id))
    return out


def try_setting(name, batches, mcfg, tcfg, device, log=print):
    """Train `len(batches)` micro-batches under one attention setting; True if all finite."""
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    impl = 'eager' if name == 'eager' else 'sdpa'
    backends = [name.split(':', 1)[1]] if name.startswith('sdpa:') else []
    dtype = torch.float32 if name == 'float32' else getattr(torch, tcfg.dtype)
    torch.manual_seed(tcfg.seed)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        mcfg.base_model, revision=mcfg.base_revision, dtype=dtype, attn_implementation=impl)
    model.to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=mcfg.lora_r, lora_alpha=mcfg.lora_alpha, lora_dropout=mcfg.lora_dropout,
        target_modules=list(mcfg.target_modules), bias='none', task_type='CAUSAL_LM'))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    model.train()
    ok = True
    losses, gnorms = [], []
    try:
        with sdpa_context(backends):
            for m, (ids, labels, attn) in enumerate(batches):
                out = model(input_ids=ids.to(device), attention_mask=attn.to(device),
                            labels=labels.to(device))
                loss = float(out.loss.item())
                losses.append(loss)
                if m == 0:
                    lg = out.logits.detach().float()
                    log(f'    first micro-batch: seq len {ids.shape[1]}, loss {loss:.4f}, '
                        f'|logits| max {lg.abs().max().item():.1f}, '
                        f'non-finite logits {(~torch.isfinite(lg)).sum().item()}')
                (out.loss / tcfg.grad_accum).backward()
                if (m + 1) % tcfg.grad_accum == 0:
                    g = float(torch.nn.utils.clip_grad_norm_(params, tcfg.grad_clip))
                    gnorms.append(g)
                    if not math.isfinite(g):
                        bad = [n for n, p in model.named_parameters()
                               if p.grad is not None and not torch.isfinite(p.grad).all()]
                        log(f'    non-finite gradients in {len(bad)} tensors, e.g. '
                            + ', '.join(bad[:3]))
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    if not all(torch.isfinite(p).all() for p in params):
                        log('    adapter weights became non-finite after the step')
                        ok = False
    except Exception as e:                        # an unsupported kernel, an OOM, ...
        log(f'    not run: {type(e).__name__}: {str(e).splitlines()[0][:160]}')
        del model, opt
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        return None
    ok = ok and all(math.isfinite(x) for x in losses) and all(math.isfinite(x) for x in gnorms)
    log(f'    losses {["%.4f" % x for x in losses]}  grad norms {["%.3f" % x for x in gnorms]}  '
        f'-> {"OK" if ok else "NON-FINITE"}  ({time.time() - t0:.0f}s)')
    del model, opt
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--cond', default='state_tracking',
                    choices=('direct', 'chain', 'state_tracking', 'narration', 'distill'))
    ap.add_argument('--steps', type=int, default=2, help='optimizer steps per setting')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--accum', type=int, default=4)
    ap.add_argument('--n-train', type=int, default=DataConfig.n_train)
    ap.add_argument('--teacher', default=TEACHER)
    ap.add_argument('--only', default=None, help='comma-separated subset of: ' + ', '.join(SETTINGS))
    ap.add_argument('--base-model', default=ModelConfig.base_model)
    ap.add_argument('--base-revision', default=None,
                    help='Hub revision (default: the pinned one for a known model)')
    args = ap.parse_args()
    if args.base_revision is None:
        args.base_revision = BASE_MODELS.get(args.base_model)

    from transformers import AutoTokenizer
    mcfg = ModelConfig(base_model=args.base_model, base_revision=args.base_revision)
    tcfg = TrainConfig(batch_size=args.batch, grad_accum=args.accum)
    dcfg = DataConfig(n_train=args.n_train)
    trcfg = TraceConfig(teacher=args.teacher)
    for ln in environment(args.device):
        print(ln)
    tok = AutoTokenizer.from_pretrained(mcfg.base_model, revision=mcfg.base_revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    batches = build_batches(args.cond, dcfg, trcfg, tok, tcfg, args.steps * args.accum)
    print(f'{args.cond} targets, {len(batches)} micro-batches of {args.batch} '
          f'({args.steps} optimizer steps), sequence lengths {[b[0].shape[1] for b in batches]}')

    settings = [s.strip() for s in args.only.split(',')] if args.only else SETTINGS
    results = {}
    for name in settings:
        print(f'\n[{name}]', flush=True)
        results[name] = try_setting(name, batches, mcfg, tcfg, args.device)
    print('\nsummary:')
    for name, ok in results.items():
        print(f'  {name:16s} {"OK" if ok else "not run" if ok is None else "NON-FINITE"}')
    good = [n for n, ok in results.items() if ok]
    if results.get('default'):
        print('\nthe default setting is fine here; a NaN in the experiment has another cause')
    elif good:
        print('\nuse one of the passing settings for the experiment, e.g.:')
        for n in good[:3]:
            if n == 'eager':
                print('  python run_experiments.py --attn eager')
            elif n.startswith('sdpa:'):
                print(f'  python run_experiments.py --sdpa-backends {n.split(":", 1)[1]}')
            elif n == 'float32':
                print('  (float32 only: a precision problem; report this)')
    sys.exit(0 if results.get('default') else 1)


if __name__ == '__main__':
    main()
