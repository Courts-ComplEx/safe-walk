"""Build compact route corridors and check exact walking-route limits."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice

import networkx as nx

from route_core import route_profile, scored_digraph


@dataclass
class Corridor:
    scored: nx.DiGraph
    source: int
    target: int
    edges: list[dict]
    seeds: list[tuple[int, ...]]
    shortest_m: float


def build_corridor(graph, source, target, *, max_edges=14, search_paths=100, weights=None):
    """Compress a few distinct mapped paths into binary street-chain choices."""
    scored = scored_digraph(graph, weights=weights)
    shortest_m = nx.shortest_path_length(graph, source, target, weight="length")
    generator = nx.shortest_simple_paths(scored, source, target, weight="score")
    paths = []
    edge_sets = []
    for path in islice(generator, search_paths):
        selected = set(zip(path, path[1:]))
        if not edge_sets or all(selected != old for old in edge_sets):
            paths.append(path)
            edge_sets.append(selected)
        if len(paths) == 3:
            break
    if not paths:
        raise nx.NetworkXNoPath("No walking route found")

    # Keep the largest path set that fits the small binary experiment.
    while paths:
        union = nx.DiGraph()
        for path in paths:
            union.add_edges_from(zip(path, path[1:]))
        junctions = {
            node for node in union
            if node in {source, target} or union.in_degree(node) != 1 or union.out_degree(node) != 1
        }
        chains = []
        elementary_to_chain = {}
        for start in junctions:
            for successor in union.successors(start):
                chain = [start, successor]
                while chain[-1] not in junctions:
                    chain.append(next(iter(union.successors(chain[-1]))))
                chain_id = len(chains)
                for pair in zip(chain, chain[1:]):
                    elementary_to_chain[pair] = chain_id
                chains.append({
                    "start": start,
                    "end": chain[-1],
                    "nodes": chain,
                    "score": sum(scored[u][v]["score"] for u, v in zip(chain, chain[1:])),
                })
        if len(chains) <= max_edges:
            seeds = []
            for path in paths:
                bits = [0] * len(chains)
                for pair in zip(path, path[1:]):
                    bits[elementary_to_chain[pair]] = 1
                seeds.append(tuple(bits))
            return Corridor(scored, source, target, chains, seeds, shortest_m)
        paths.pop()
        edge_sets.pop()
    raise ValueError("Could not build a compact corridor")


def decode_route(corridor, bits):
    """Expand a selected set of macro-edges; reject branches and disconnected loops."""
    selected = [i for i, bit in enumerate(bits) if bit]
    outgoing = {}
    for i in selected:
        start = corridor.edges[i]["start"]
        if start in outgoing:
            return None
        outgoing[start] = i
    route = [corridor.source]
    used = set()
    current = corridor.source
    while current != corridor.target:
        if current not in outgoing:
            return None
        edge_id = outgoing[current]
        if edge_id in used:
            return None
        used.add(edge_id)
        chain = corridor.edges[edge_id]["nodes"]
        route.extend(chain[1:])
        current = chain[-1]
        if len(route) > len(corridor.scored):
            return None
    if len(used) != len(selected) or len(route) != len(set(route)):
        return None
    return route


def assess(corridor, bits, *, max_unlit_m, max_detour_pct, avoid_unknown):
    route = decode_route(corridor, bits)
    if route is None:
        return None
    profile = route_profile(corridor.scored, route)
    allowed_m = corridor.shortest_m * (1 + max_detour_pct / 100)
    if profile["distance_m"] > allowed_m + 0.1:
        return None
    if profile["longest_unlit_m"] > max_unlit_m + 0.1:
        return None
    if avoid_unknown and profile["unknown_lighting_m"] > 0:
        return None
    return route, profile
