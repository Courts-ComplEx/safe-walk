"""Starter routing logic for the Wits walking-route hackathon.

Scores describe mapped walking conditions, not the probability of crime or GBV.
Unknown lighting is treated as unlit for route scoring.
"""

from __future__ import annotations

import ast
from itertools import islice
from math import cos, radians

import networkx as nx


def nearest_node(graph, latitude, longitude):
    """Find a nearby node without OSMnx's optional scikit-learn dependency."""
    lon_scale = cos(radians(latitude))
    return min(
        graph.nodes,
        key=lambda node: (
            (float(graph.nodes[node]["y"]) - latitude) ** 2
            + ((float(graph.nodes[node]["x"]) - longitude) * lon_scale) ** 2
        ),
    )


def _tag(value):
    """Normalise common OSMnx/GraphML tag shapes."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return str(value[0]).lower() if value else ""
    value = str(value)
    if value.startswith("["):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list) and parsed:
                return str(parsed[0]).lower()
        except (ValueError, SyntaxError):
            pass
    return value.lower()


def street_conditions(data):
    """Return lighting, sidewalk status, and a road-class traffic proxy."""
    lit = _tag(data.get("lit"))
    if lit in {"yes", "24/7", "automatic"}:
        lighting = "lit"
    elif lit in {"no", "disused"}:
        lighting = "unlit"
    else:
        lighting = "unlit"

    sidewalk = _tag(data.get("sidewalk"))
    sides = [_tag(data.get(f"sidewalk:{side}")) for side in ("left", "right")]
    if sidewalk in {"both", "yes", "left", "right", "separate"} or any(
        s in {"yes", "separate"} for s in sides
    ):
        pavement = "mapped"
    elif sidewalk in {"no", "none"} or sides == ["no", "no"]:
        pavement = "none"
    else:
        pavement = "unknown"

    road_type = _tag(data.get("highway"))
    # Road class is a rough proxy, not measured traffic volume.
    traffic_proxy = road_type in {"primary", "secondary", "tertiary", "trunk"}
    return lighting, pavement, traffic_proxy


def edge_score(data, *, weights=None):
    """Additive route cost; lower is preferred and weights set user priorities."""
    w = weights or {
        "distance": 0.2, "unlit": 2.0, "unknown_lighting": 1.0,
        "no_sidewalk": 2.0, "unknown_sidewalk": 1.0, "traffic": 0.0,
    }
    length = float(data.get("length", 0.0))
    lighting, pavement, traffic = street_conditions(data)
    per_100m = (
        w["distance"]
        + w["unlit"] * (lighting == "unlit")
        + w["unknown_lighting"] * (lighting == "unknown")
        + w["no_sidewalk"] * (pavement == "none")
        + w["unknown_sidewalk"] * (pavement == "unknown")
        + w["traffic"] * traffic
    )
    return max(length, 0.0) / 100.0 * per_100m


def scored_digraph(graph, *, weights=None):
    """Keep the lowest-scoring version of each directed street connection."""
    simple = nx.DiGraph()
    simple.add_nodes_from(graph.nodes(data=True))
    for u, v, _key, data in graph.edges(keys=True, data=True):
        candidate = dict(data)
        candidate["score"] = edge_score(candidate, weights=weights)
        # Only penalise a crossing explicitly mapped as unmarked/absent.
        if _tag(graph.nodes[v].get("crossing")) in {"unmarked", "no"}:
            candidate["score"] += 4.0
        candidate["length"] = float(candidate.get("length", 0.0))
        if not simple.has_edge(u, v) or candidate["score"] < simple[u][v]["score"]:
            simple.add_edge(u, v, **candidate)
    return simple


def route_profile(graph, route):
    """Calculate route cost, length, and mapped street-condition distances."""
    distance = score = unlit_run = longest_unlit = unlit_m = unknown_lighting = 0.0
    missing_sidewalk_m = unknown_sidewalk_m = traffic_m = 0.0
    for u, v in zip(route, route[1:]):
        data = graph[u][v]
        length = data["length"]
        lighting, pavement, traffic = street_conditions(data)
        distance += length
        score += data["score"]
        if lighting == "unlit":
            unlit_m += length
            unlit_run += length
            longest_unlit = max(longest_unlit, unlit_run)
        else:
            unlit_run = 0.0
        if lighting == "unknown":
            unknown_lighting += length
        if pavement == "none":
            missing_sidewalk_m += length
        elif pavement == "unknown":
            unknown_sidewalk_m += length
        if traffic:
            traffic_m += length
    return {
        "distance_m": round(distance, 1),
        "score": round(score, 2),
        "unlit_m": round(unlit_m, 1),
        "longest_unlit_m": round(longest_unlit, 1),
        "unknown_lighting_m": round(unknown_lighting, 1),
        "missing_sidewalk_m": round(missing_sidewalk_m, 1),
        "unknown_sidewalk_m": round(unknown_sidewalk_m, 1),
        "traffic_proxy_m": round(traffic_m, 1),
    }


def find_routes(
    graph,
    source,
    target,
    *,
    max_unlit_m=100.0,
    max_detour_pct=20.0,
    avoid_unknown_lighting=False,
    candidates_to_check=100,
    return_count=3,
):
    """Starter candidate search. An empty result is NOT proof no feasible route exists."""
    scored = scored_digraph(graph)
    shortest_m = nx.shortest_path_length(graph, source, target, weight="length")
    allowed_m = shortest_m * (1 + max_detour_pct / 100.0)
    feasible = []
    candidate_paths = nx.shortest_simple_paths(scored, source, target, weight="score")
    for route in islice(candidate_paths, candidates_to_check):
        profile = route_profile(scored, route)
        if profile["distance_m"] > allowed_m:
            continue
        if profile["longest_unlit_m"] > max_unlit_m:
            continue
        if avoid_unknown_lighting and profile["unknown_lighting_m"] > 0:
            continue
        feasible.append((route, profile))
        if len(feasible) >= return_count:
            break
    return feasible, {"shortest_m": round(shortest_m, 1), "allowed_m": round(allowed_m, 1), "checked_up_to": candidates_to_check}
