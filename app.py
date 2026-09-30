"""Wits safer walking-route prototype. Run: streamlit run app.py"""
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import pydeck as pdk
import streamlit as st

from route_core import nearest_node, scored_digraph
from route_corridor import Corridor
from segment_conditions import load_excel_conditions
from tensor_route_solver import solve_tensor_network

GRAPH_PATH = Path(__file__).with_name("wits_walk_3km.graphml")
ADDRESS_PATH = GRAPH_PATH.with_name("wits_map_data_10km.xlsx")
EXCEL_PATH = GRAPH_PATH.with_name("wits_segment_conditions.xlsx")
MAX_DETOUR_PCT = 50
WITS_MAIN = (-26.1928, 28.0303)
MAP_RADIUS_KM = 3.0

addresses = pd.read_excel(
    ADDRESS_PATH,
    sheet_name="Addresses",
    header=4,
).dropna(subset=["House number", "Street", "Latitude", "Longitude"])

# Only offer addresses covered by the smaller walking map.
lat = np.radians(addresses["Latitude"].astype(float))
lon = np.radians(addresses["Longitude"].astype(float))
center_lat, center_lon = np.radians(WITS_MAIN)
a = (
    np.sin((lat - center_lat) / 2) ** 2
    + np.cos(center_lat) * np.cos(lat) * np.sin((lon - center_lon) / 2) ** 2
)
distance_km = 2 * 6371.0088 * np.arcsin(np.minimum(1, np.sqrt(a)))
addresses = addresses.loc[distance_km <= MAP_RADIUS_KM].reset_index(drop=True)

def address_label(i):
    row = addresses.iloc[i]
    number = str(row["House number"]).removesuffix(".0")
    label = f"{number} {row['Street']}"
    if pd.notna(row["Suburb"]):
        label += f", {row['Suburb']}"
    if pd.notna(row.get("Place name")):
        label += f" — {row['Place name']}"
    return label

st.set_page_config(page_title="Safe Walk", page_icon="🚶", layout="wide")
st.html("""
<style>
    .stApp {
        background: #ffffff;
        color: #30233f;
        --primary-color: #6f4299;
        --background-color: #ffffff;
        --secondary-background-color: #eff7fc;
        --text-color: #30233f;
    }
    [data-testid="stHeader"] { background: #ffffff; }
    [data-testid="stMainBlockContainer"] { padding-top: 2rem; }
    [data-testid="stBaseButton-primary"] {
        background-color: #6f4299;
        border-color: #6f4299;
        color: #ffffff;
    }
    [data-testid="stBaseButton-primary"]:hover {
        background-color: #553178;
        border-color: #553178;
        color: #ffffff;
    }
    [data-baseweb="select"] > div {
        background-color: #eff7fc;
        color: #30233f;
        border-color: #d8cbe8;
    }
    [data-testid="stCheckbox"] input { accent-color: #6f4299; }
    [data-testid="stMetricValue"] { color: #6f4299; }
    .safe-walk-brand {
        position: relative;
        overflow: hidden;
        background: #65418a;
        border-radius: 20px;
        padding: clamp(28px, 5vw, 48px);
        margin: 0 0 26px;
        border-bottom: 5px solid #c9e8f6;
    }
    .safe-walk-brand::after {
        content: "";
        position: absolute;
        right: -35px;
        top: -95px;
        width: 330px;
        height: 330px;
        border: 2px solid rgba(201, 232, 246, 0.18);
        border-radius: 46px;
        transform: rotate(32deg);
        box-shadow: 0 0 0 35px rgba(201, 232, 246, 0.06),
                    0 0 0 70px rgba(201, 232, 246, 0.04);
        pointer-events: none;
    }
    .safe-walk-brand h1 {
        position: relative;
        z-index: 1;
        margin: 0;
        padding: 0;
        color: #ffffff;
        font-family: inherit;
        font-size: clamp(42px, 7vw, 68px);
        font-weight: 800;
        letter-spacing: -1.8px;
        line-height: 1.12;
    }
    .safe-walk-brand h1 span { color: #c9e8f6; }
    .safe-walk-brand p {
        position: relative;
        z-index: 1;
        margin: 14px 0 0;
        color: #ffffff;
        font-size: 16px;
        line-height: 1.5;
    }
    .safe-walk-intro { margin-bottom: 24px; }
    .safe-walk-intro h2 {
        color: #553178;
        padding: 0;
        margin: 0 0 8px;
        font-size: clamp(22px, 4vw, 28px);
        letter-spacing: -0.5px;
    }
    .safe-walk-intro p {
        margin: 0;
        max-width: 650px;
        color: #51455e;
        font-size: 15px;
        line-height: 1.6;
    }
</style>
<header class="safe-walk-brand">
    <h1>Safe <span>Walk</span></h1>
    <p>Your walk. Your priorities.</p>
</header>
<div class="safe-walk-intro">
    <h2>A better route starts with you.</h2>
    <p>Choose where you’re going and the street conditions that matter most.
    Explore walking routes scored around your priorities.</p>
</div>
""")

start_i = st.selectbox(
    "Where from? (current location)",
    range(len(addresses)),
    format_func=address_label,
    index=None,
    placeholder="Type to search for a starting address",
)
end_i = st.selectbox(
    "Where to? (destination)",
    range(len(addresses)),
    format_func=address_label,
    index=None,
    placeholder="Type to search for a destination address",
)

if start_i is None or end_i is None:
    st.info("Choose a starting address and destination.")
    st.stop()

start_lat = float(addresses.iloc[start_i]["Latitude"])
start_lon = float(addresses.iloc[start_i]["Longitude"])
end_lat = float(addresses.iloc[end_i]["Latitude"])
end_lon = float(addresses.iloc[end_i]["Longitude"])

@st.cache_resource(max_entries=1, show_spinner='Preparing walking map and street conditions...')
def load_graph_with_excel_conditions(excel_mtime_ns):
    # The workbook modification time invalidates this cache after a saved edit.
    graph = ox.io.load_graphml(GRAPH_PATH)
    return load_excel_conditions(graph, EXCEL_PATH)

def build_corridor(graph, source, target, *, max_edges=32, search_paths=100, weights=None):
    """Compress a few distinct mapped paths into binary street-chain choices."""
    scored = scored_digraph(graph, weights=weights)
    shortest_m = nx.shortest_path_length(graph, source, target, weight="length")
    # Include distinct distance-based alternatives instead of tiny variations
    # of the three cheapest paths. All routes retain the same detour limit.
    allowed_m = shortest_m * 1.5 + 0.1
    paths = []
    edge_sets = []

    def add_path(path):
        selected = set(zip(path, path[1:]))
        length = sum(scored[u][v]['length'] for u, v in selected)
        if length > allowed_m or not length:
            return False
        for old in edge_sets:
            common = sum(scored[u][v]['length'] for u, v in selected & old)
            old_length = sum(scored[u][v]['length'] for u, v in old)
            if common / min(length, old_length) > 0.75:
                return False
        paths.append(path)
        edge_sets.append(selected)
        return True

    add_path(nx.shortest_path(scored, source, target, weight='score'))
    add_path(nx.shortest_path(scored, source, target, weight='length'))
    for _ in range(3):
        if len(paths) >= 3:
            break
        used = set().union(*edge_sets) if edge_sets else set()
        found = False
        for factor in (1.5, 2, 3, 5, 10, 30):
            path = nx.shortest_path(
                scored, source, target,
                weight=lambda u, v, data: data['length'] * (factor if (u, v) in used else 1),
            )
            if add_path(path):
                found = True
                break
        if not found:
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
            if len(chains) == 1 and len(chains[0]['nodes']) > 2:
                nodes = chains[0]['nodes']
                middle = len(nodes) // 2
                chains = []
                elementary_to_chain = {}
                for chain in (nodes[:middle + 1], nodes[middle:]):
                    chain_id = len(chains)
                    for pair in zip(chain, chain[1:]):
                        elementary_to_chain[pair] = chain_id
                    chains.append({'start': chain[0], 'end': chain[-1], 'nodes': chain,
                                   'score': sum(scored[u][v]['score'] for u, v in zip(chain, chain[1:]))})
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



def line_data(graph, route):
    return pd.DataFrame([
        {'start': [float(graph.nodes[u]['x']), float(graph.nodes[u]['y'])],
         'end': [float(graph.nodes[v]['x']), float(graph.nodes[v]['y'])]}
        for u, v in zip(route, route[1:])
    ])

if not GRAPH_PATH.exists():
    st.error(f'Saved walking map not found: {GRAPH_PATH}')
    st.stop()
if not EXCEL_PATH.exists():
    st.error(f'Editable segment workbook not found: {EXCEL_PATH}')
    st.stop()

st.header('What matters on your walk?')
prefer_lighting = st.checkbox('Minimise unlit walking', value=True)
prefer_sidewalks = st.checkbox('Minimise walking without sidewalks', value=True)
st.caption('These are route preferences. A route may still include unlit streets or streets without mapped sidewalks.')

st.caption('Demo workbook: lighting and sidewalk values are synthetic, not verified street conditions.')
run = st.button('Find your route', type='primary')

weights = {
    'distance': 0.5,
    'unlit': 6.0 if prefer_lighting else 0.0,
    'unknown_lighting': 3.0 if prefer_lighting else 0.0,
    'no_sidewalk': 6.0 if prefer_sidewalks else 0.0,
    'unknown_sidewalk': 3.0 if prefer_sidewalks else 0.0,
    'traffic': 1.0,
}

def score_out_of_10(profile):
    distance = profile["distance_m"]
    if distance <= 0:
        return 0.0

    # Keep the displayed safety rubric fixed when someone changes route preferences.
    condition_cost = (
        6 * profile['unlit_m']
        + 6 * profile['missing_sidewalk_m']
        + 2 * profile['unknown_sidewalk_m']
        + profile['traffic_proxy_m']
    )
    rating = 10 * (1 - condition_cost / (13 * distance))
    return max(0.0, min(10.0, rating))

excel_mtime_ns = EXCEL_PATH.stat().st_mtime_ns
request_key = (start_i, end_i, prefer_lighting, prefer_sidewalks, excel_mtime_ns)

if run:
    st.session_state.pop('route_results', None)
    try:
        graph = load_graph_with_excel_conditions(excel_mtime_ns)
    except (ValueError, FileNotFoundError) as error:
        st.error(f'Could not read segment conditions: {error}')
        st.stop()
    source = nearest_node(graph, start_lat, start_lon)
    target = nearest_node(graph, end_lat, end_lon)
    if source == target:
        st.warning('The points snapped to the same mapped walking node. Choose points farther apart.')
        st.stop()
    try:
        with st.spinner('Finding candidate routes...'):
            corridor = build_corridor(graph, source, target, search_paths=30, weights=weights)
        with st.spinner('Optimising the tensor network...'):
            candidates, stats = solve_tensor_network(
                corridor,
                max_unlit_m=float('inf'),
                max_detour_pct=MAX_DETOUR_PCT,
                return_candidates=True,
            )
    except (nx.NetworkXNoPath, nx.NodeNotFound, ValueError, RuntimeError) as error:
        st.error(f'Could not find a route for these points: {error}')
        st.stop()
    if not candidates:
        st.warning('No route was found within the app’s detour limit for these addresses. Try other addresses.')
        st.stop()

    ranked_candidates = sorted(
        ((route, profile, score_out_of_10(profile)) for route, profile in candidates),
        key=lambda item: (-item[2], item[1]["distance_m"]),
    )

    def shared_fraction(route_a, route_b):
        edges_a = set(zip(route_a, route_a[1:]))
        edges_b = set(zip(route_b, route_b[1:]))
        total = sum(corridor.scored[u][v]["length"] for u, v in edges_a)
        shared = sum(corridor.scored[u][v]["length"] for u, v in edges_a & edges_b)
        return shared / total if total else 1.0

    ranked = []
    for candidate in ranked_candidates:
        if all(shared_fraction(candidate[0], chosen[0]) <= 0.75
            for chosen in ranked):
            ranked.append(candidate)
        if len(ranked) == 3:
            break

    ranked.sort(key=lambda item: (-item[2], item[1]['distance_m']))

    st.session_state['route_results'] = {
        'request_key': request_key,
        'routes': ranked,
        'stats': stats,
    }
    st.session_state['route_choice'] = 0

saved = st.session_state.get('route_results')
if saved is not None and saved['request_key'] == request_key:
    routes = saved['routes']
    stats = saved['stats']

    def route_label(index):
        _route, profile, rating = routes[index]
        name = 'Recommended route' if index == 0 else f'Alternative {index}'
        return f"{name} · {rating:.1f}/10 · {profile['distance_m'] / 1000:.2f} km"

    selected = st.selectbox(
        'View a route', range(len(routes)), format_func=route_label, key='route_choice'
    )
    route, profile, rating = routes[selected]
    graph = load_graph_with_excel_conditions(excel_mtime_ns)

    a, b, c, d = st.columns(4)
    a.metric("Mapped route score", f"{rating:.1f}/10")
    b.metric('Walking distance', f"{profile['distance_m']:.0f} m")
    c.metric('Unlit walking', f"{profile['unlit_m']:.0f} m")
    d.metric('Explicitly no sidewalk', f"{profile['missing_sidewalk_m']:.0f} m")
    st.caption('Unknown lighting is counted as unlit. Sidewalk values marked unknown remain unknown.')

    layer = pdk.Layer(
        'LineLayer', line_data(graph, route), get_source_position='start',
        get_target_position='end', get_color=[111, 66, 153],
        get_width=5, width_min_pixels=3,
    )
    middle_node = graph.nodes[route[len(route) // 2]]

    score_layer = pdk.Layer(
        "TextLayer",
        data=[{
            "position": [
                float(middle_node["x"]),
                float(middle_node["y"]),
            ],
            "label": f"Safety score: {rating:.1f}/10",
        }],
        get_position="position",
        get_text="label",
        get_size=20,
        get_color=[255, 255, 255, 255],
        background=True,
        get_background_color=[85, 49, 120, 245],
        background_padding=[12, 8],
        background_border_radius=8,
        pickable=False,
    )    
    
    view = pdk.ViewState(latitude=start_lat, longitude=start_lon, zoom=14, pitch=0)
    st.pydeck_chart(
        pdk.Deck(layers=[layer, score_layer], initial_view_state=view, map_style="light"),
        width="stretch",
    )

    with st.expander('How the tensor network chose this route'):
        st.write(
            'Each binary variable selects a chain of mapped walking streets. An energy function adds '
            'street condition scores based on your selected priorities and penalises broken route connections. '
            'Unknown lighting and sidewalks are treated as "unlit" and "no sidewalk", resepctively. '
            'The energy is stored as a matrix product operator (MPO). DMRG optimises '
            'a matrix product state (MPS) toward low energy routes. Sampled routes are decoded, '
            'checked against a fixed detour limit, and ranked by mapped route score.'
        )
        st.write(
            f"This run used **{stats['choices']} binary street-chain choices**, "
            f"an MPS bond dimension of **{stats['mps_bond_dimension']}**, and checked "
            f"**{stats['unique_candidates_checked']} distinct route selections**. "
            f"It found **{stats['valid_routes_found']} distinct valid routes**."
        )
        st.caption('This is an approximate classical tensor-network method. It does not use a quantum computer or guarantee a globally optimal route.')

st.write('Scores describe mapped street conditions, not the likelihood of crime or GBV. Use your own judgement and local knowledge when walking.')
st.caption('© OpenStreetMap contributors. Segment lighting and sidewalk edits come from saved street data. Road class is a traffic proxy, not a traffic count.')
