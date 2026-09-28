"""Quantum-inspired tensor-network search for a compact mapped walking corridor.

An MPO stores a binary route energy; DMRG variationally optimizes an MPS.
The result is approximate and all displayed routes receive an exact constraint check.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault('NUMBA_CACHE_DIR', str(Path(tempfile.gettempdir()) / 'hackathon_numba_cache'))

from itertools import combinations
import numpy as np
import quimb.operator as qop
import quimb.tensor as qtn

from route_corridor import assess, build_corridor
from route_core import street_conditions


def _chain_conditions(corridor, edge):
    lengths = []
    states = []
    for u, v in zip(edge['nodes'], edge['nodes'][1:]):
        data = corridor.scored[u][v]
        lengths.append(float(data['length']))
        states.append(street_conditions(data)[0])
    run = maximum = 0.0
    for length, state in zip(lengths, states):
        run = run + length if state == 'unlit' else 0.0
        maximum = max(maximum, run)
    prefix = 0.0
    for length, state in zip(lengths, states):
        if state != 'unlit':
            break
        prefix += length
    suffix = 0.0
    for length, state in zip(reversed(lengths), reversed(states)):
        if state != 'unlit':
            break
        suffix += length
    return {
        'distance': sum(lengths),
        'max_unlit': maximum,
        'unlit_prefix': prefix,
        'unlit_suffix': suffix,
        'unknown': sum(length for length, state in zip(lengths, states) if state == 'unknown'),
    }


def build_hamiltonian(corridor, *, max_unlit_m=100, avoid_unknown=False):
    """Make a diagonal 2-local MPO for score and route-flow constraints.

    The complete detour and uninterrupted-unlit constraints are checked after
    decoding because they are path-level rules, not generally 2-local rules.
    """
    n = len(corridor.edges)
    if n < 2:
        raise ValueError('At least two corridor choices are needed for MPS optimization')
    space = qop.HilbertSpace(sites=list(range(n)))
    h = qop.SparseOperatorBuilder(hilbert_space=space)
    costs = np.array([float(edge['score']) for edge in corridor.edges])
    penalty = float(max(100.0, 2 * costs.sum() + 1))
    linear = costs.copy()
    quadratic = {}

    def add_pair(i, j, value):
        a, b = sorted((i, j))
        quadratic[(a, b)] = quadratic.get((a, b), 0.0) + value

    # The selected directed chains must have one unit of net flow from A to B.
    nodes = {corridor.source, corridor.target}
    for edge in corridor.edges:
        nodes.update((edge['start'], edge['end']))
    for node in nodes:
        b = 1 if node == corridor.source else -1 if node == corridor.target else 0
        incident = [(i, (1 if edge['start'] == node else 0) - (1 if edge['end'] == node else 0))
                    for i, edge in enumerate(corridor.edges)]
        incident = [(i, a) for i, a in incident if a]
        for i, a in incident:
            linear[i] += penalty * (a*a - 2*b*a)
        for (i, a), (j, c) in combinations(incident, 2):
            add_pair(i, j, 2*penalty*a*c)

    conditions = [_chain_conditions(corridor, edge) for edge in corridor.edges]
    for i, cond in enumerate(conditions):
        if cond['max_unlit'] > max_unlit_m + 1e-6 or (avoid_unknown and cond['unknown'] > 0):
            linear[i] += penalty
    for i, first in enumerate(corridor.edges):
        for j, second in enumerate(corridor.edges):
            if i != j and first['end'] == second['start']:
                if conditions[i]['unlit_suffix'] + conditions[j]['unlit_prefix'] > max_unlit_m + 1e-6:
                    add_pair(i, j, penalty)

    for i, c in enumerate(linear):
        if abs(c) > 1e-12:
            h.add_term(float(c), ('n', i))
    for (i, j), c in quadratic.items():
        if abs(c) > 1e-12:
            h.add_term(float(c), ('n', i), ('n', j))
    return h.build_mpo(), {'choices': n, 'penalty': penalty, 'mpo_terms': n + len(quadratic)}


def solve_tensor_network(
    corridor,
    *,
    max_unlit_m=100,
    max_detour_pct=20,
    avoid_unknown=False,
    bond_dimension=16,
    samples=128,
    seed=7,
    return_candidates=False,
):
    """Optimize an MPS and return valid routes, or the best one by default."""
    mpo, diagnostics = build_hamiltonian(
        corridor, max_unlit_m=max_unlit_m, avoid_unknown=avoid_unknown
    )
    dmrg = qtn.DMRG2(mpo, bond_dims=[bond_dimension, bond_dimension], cutoffs=1e-9)
    converged = dmrg.solve(tol=1e-5, max_sweeps=8, verbosity=0)
    state = dmrg.state
    rng = np.random.default_rng(seed)
    seen = set()
    best = None
    best_source = None
    valid_mps_candidates = 0
    valid_routes = {}
    # Try MPS samples first. Seed routes are a fallback when the approximate
    # optimizer misses a feasible part of the compact corridor.
    proposals = []
    for _ in range(samples):
        bits, _probability = state.sample_configuration(seed=int(rng.integers(0, 2**32 - 1)))
        proposals.append(tuple(bits))
    proposals.extend(corridor.seeds)
    for index, bits in enumerate(proposals):
        bits = tuple(int(bit) for bit in bits)
        if bits in seen:
            continue
        seen.add(bits)
        candidate = assess(
            corridor, bits, max_unlit_m=max_unlit_m,
            max_detour_pct=max_detour_pct, avoid_unknown=avoid_unknown,
        )
        if candidate is not None and index < samples:
            valid_mps_candidates += 1
        if candidate is not None:
            route_key = tuple(candidate[0])
            valid_routes.setdefault(route_key, candidate)
        if candidate is not None and (best is None or candidate[1]['score'] < best[1]['score']):
            best = candidate
            best_source = 'MPS sample' if index < samples else 'mapped seed route'
    diagnostics.update({
        'converged': bool(converged),
        'mps_bond_dimension': int(state.max_bond()),
        'unique_candidates_checked': len(seen),
        'sampled_candidates': samples,
        'valid_mps_candidates': valid_mps_candidates,
        'valid_routes_found': len(valid_routes),
        'selected_from': best_source,
    })
    if return_candidates:
        routes = sorted(valid_routes.values(), key=lambda item: item[1]['score'])
        return routes, diagnostics
    return best, diagnostics
