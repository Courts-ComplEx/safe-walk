"""Wits safer walking-route prototype. Run: streamlit run app.py"""
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import pydeck as pdk
import streamlit as st

from route_core import nearest_node
from segment_conditions import load_excel_conditions
from tensor_route_solver import build_corridor, solve_tensor_network

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
    return f"{label}"

st.set_page_config(page_title='Stay Safe', layout='wide')
st.html("""
<style>
    .stay-safe-brand {
        display: flex;
        align-items: center;
        gap: 16px;
        margin: 8px 0 24px;
        color: var(--text-color, #173c32);
    }
    .stay-safe-mark {
        display: flex;
        align-items: center;
        justify-content: center;
        flex: 0 0 64px;
        height: 64px;
        border-radius: 20px;
        background: #173c32;
        box-shadow: 0 6px 18px rgba(23, 60, 50, 0.15);
    }
    .stay-safe-wordmark {
        margin: 0;
        padding: 0;
        font-family: inherit;
        font-size: clamp(32px, 6vw, 46px);
        font-weight: 800;
        letter-spacing: -1.8px;
        line-height: 1.1;
    }
    .stay-safe-wordmark span { color: #369b73; }
    .stay-safe-tagline {
        margin: 7px 0 0;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 2.4px;
        text-transform: uppercase;
    }
    .stay-safe-intro {
        border-top: 1px solid rgba(128, 128, 128, 0.25);
        padding-top: 22px;
        margin-bottom: 24px;
        color: var(--text-color, #173c32);
    }
    .stay-safe-intro h2 {
        margin: 0 0 8px;
        padding: 0;
        font-size: clamp(22px, 4vw, 30px);
        font-weight: 650;
        letter-spacing: -0.6px;
        line-height: 1.25;
    }
    .stay-safe-intro p {
        margin: 0;
        max-width: 620px;
        font-size: 15px;
        line-height: 1.6;
    }
</style>
<header class="stay-safe-brand">
    <div class="stay-safe-mark" aria-hidden="true">
        <svg width="40" height="40" viewBox="0 0 40 40" fill="none"
             xmlns="http://www.w3.org/2000/svg">
            <path d="M10 30V22C10 17 30 23 30 16V10"
                  stroke="#c5edda" stroke-width="3.5" stroke-linecap="round"/>
            <circle cx="10" cy="30" r="4" fill="#c5edda"/>
            <circle cx="30" cy="10" r="5" fill="#ef82aa"/>
            <circle cx="30" cy="10" r="2" fill="#173c32"/>
        </svg>
    </div>
    <div>
        <h1 class="stay-safe-wordmark">Stay <span>Safe</span></h1>
        <p class="stay-safe-tagline">Your walk. Your priorities.</p>
    </div>
</header>
<div class="stay-safe-intro">
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

    # Prefer substantially different routes, then fill remaining slots with
    # routes that still differ by at least one mapped street edge.
    for candidate in ranked_candidates:
        if len(ranked) == 3:
            break
        candidate_edges = set(zip(candidate[0], candidate[0][1:]))
        if all(candidate_edges != set(zip(chosen[0], chosen[0][1:])) for chosen in ranked):
            ranked.append(candidate)
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

    # ── Appearance settings ──────────────────────────────

    MAP_STYLE = "light"       # "light", "dark", "light_no_labels", "dark_no_labels"

    # Layer colours use [red, green, blue, opacity], each from 0–255.
    ROUTE_COLOR = [194, 24, 91, 255]       # Pink
    ROUTE_WIDTH = 5                       # Pixels
    ROUTE_OUTLINE_COLOR = [255, 255, 255, 230]
    ROUTE_OUTLINE_WIDTH = 9               # Wider than the route

    LABEL_TEXT_COLOR = [255, 255, 255, 255]
    LABEL_BACKGROUND_COLOR = [22, 135, 65, 245]
    LABEL_FONT_SIZE = 16
    LABEL_PADDING = [12, 8]               # Horizontal, vertical
    LABEL_CORNER_RADIUS = 8

    MAP_ZOOM = 14
    MAP_PITCH = 0                         # 0 = flat; try 30 for tilt
    MAP_BEARING = 0                       # Rotation in degrees

    # ── Route ────────────────────────────────────────────

    route_data = line_data(graph, route)

    # Draw a wider line underneath to give the route an outline.
    route_outline = pdk.Layer(
        "LineLayer",
        data=route_data,
        get_source_position="start",
        get_target_position="end",
        get_color=ROUTE_OUTLINE_COLOR,
        get_width=ROUTE_OUTLINE_WIDTH,
        width_units="pixels",
        pickable=False,
    )

    layer = pdk.Layer(
        "LineLayer",
        data=route_data,
        get_source_position="start",
        get_target_position="end",
        get_color=ROUTE_COLOR,
        get_width=ROUTE_WIDTH,
        width_units="pixels",
        pickable=False,
    )

    # ── Safety label ─────────────────────────────────────

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
        get_size=LABEL_FONT_SIZE,
        get_color=LABEL_TEXT_COLOR,
        get_pixel_offset=[0, -30],         # Move label above the route
        font_family="Arial",
        font_weight="bold",
        background=True,
        get_background_color=LABEL_BACKGROUND_COLOR,
        background_padding=LABEL_PADDING,
        background_border_radius=LABEL_CORNER_RADIUS,
        pickable=False,
    )

    # ── Map ──────────────────────────────────────────────

    view = pdk.ViewState(
        latitude=start_lat,
        longitude=start_lon,
        zoom=MAP_ZOOM,
        pitch=MAP_PITCH,
        bearing=MAP_BEARING,
    )

    st.pydeck_chart(
        pdk.Deck(
            map_provider="carto",
            map_style=MAP_STYLE,
            layers=[route_outline, layer, score_layer],
            initial_view_state=view,
        ),
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
