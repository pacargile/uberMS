"""
Timing benchmark for the SVI regression cases.

For each case: NN load, svi.init, the first update (compile), and the
steady-state time per SVI step. Run it on an otherwise idle machine.

    python tests/regression/bench.py [--cases ...] [--steps 200] [--out bench.json]
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run as reg  # noqa: E402  (also puts the repo first on sys.path)
from cases import CASES, resolve  # noqa: E402


def bench_case(name, nsteps):
    import jax
    from jax import random as jrandom

    case = resolve(name)
    arrs, meta = reg.load_frozen(case)
    indict = reg.build_indict(case, arrs, meta, outfile=os.devnull)
    t0 = time.perf_counter()
    driver = reg.make_driver(case['driver'])
    t_load = time.perf_counter() - t0

    svi, _, _, modelkw = driver.run(indict, dryrun=True)
    t0 = time.perf_counter()
    state = svi.init(jrandom.PRNGKey(0), **modelkw)
    jax.block_until_ready(state)
    t_init = time.perf_counter() - t0

    update = jax.jit(lambda s: svi.stable_update(s, **modelkw))
    t0 = time.perf_counter()
    state, loss = update(state)
    jax.block_until_ready(loss)
    t_compile = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(nsteps):
        state, loss = update(state)
    jax.block_until_ready(loss)
    ms_step = (time.perf_counter() - t0) / nsteps * 1e3
    jax.clear_caches()
    return {'load_s': round(t_load, 2), 'init_s': round(t_init, 2),
            'compile_s': round(t_compile, 2), 'ms_per_step': round(ms_step, 2),
            'min_per_30k': round(ms_step * 30000 / 6e4, 1)}


def main():
    svi_cases = [n for n in CASES if resolve(n)['driver'].startswith('svi')]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cases', nargs='+', default=svi_cases)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    import jax
    res = {'provenance': reg.provenance(), 'devices': [str(d) for d in jax.devices()],
           'cases': {}}
    print(f'{"case":24s} {"load s":>7s} {"init s":>7s} {"compile s":>9s} {"ms/step":>8s} {"min/30k":>8s}')
    for name in args.cases:
        r = bench_case(name, args.steps)
        res['cases'][name] = r
        print(f'{name:24s} {r["load_s"]:7.1f} {r["init_s"]:7.1f} {r["compile_s"]:9.1f} '
              f'{r["ms_per_step"]:8.1f} {r["min_per_30k"]:8.1f}', flush=True)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
