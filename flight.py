from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import islice

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load CSV and build graph
file_path = "C:/Users/HP/Downloads/Flight-Route-Planner/indian_flights.csv"
data = pd.read_csv(file_path)

# Ensure Distance is numeric and remove duplicates
data['Distance'] = pd.to_numeric(data['Distance'], errors='coerce')
data = data.drop_duplicates(subset=['Origin_airport', 'Destination_airport'])

# Create a directed graph for flights
graph = nx.DiGraph()

# Add edges to the graph
for _, row in data.iterrows():
    graph.add_edge(row['Origin_airport'], row['Destination_airport'], weight=row['Distance'])

@app.route('/')
def index():
    return render_template('index.html')

# Find shortest and second shortest paths
@app.route('/find_alternative_paths', methods=['GET'])
def find_alternative_paths():
    start = request.args.get('start', '').upper()
    end = request.args.get('end', '').upper()

    if start not in graph.nodes or end not in graph.nodes:
        return jsonify({'error': 'Invalid airport code(s)'}), 400

    try:
        # Get all possible paths
        paths = list(islice(nx.shortest_simple_paths(graph, start, end, weight='weight'), 5))

        if not paths:
            return jsonify({'error': 'No path found'}), 404

        # Compute distances for each path
        path_distances = [(p, sum(graph[u][v]['weight'] for u, v in zip(p, p[1:]))) for p in paths]
        
        # Sort by distance
        path_distances = sorted(path_distances, key=lambda x: x[1])

        # Extract shortest and second shortest paths
        shortest_path, shortest_distance = path_distances[0]
        second_shortest = path_distances[1] if len(path_distances) > 1 else None

        return jsonify({
            'shortest_path': {'path': shortest_path, 'distance': shortest_distance},
            'alternative_route': {'path': second_shortest[0], 'distance': second_shortest[1]} if second_shortest else None
        })

    except nx.NetworkXNoPath:
        return jsonify({'error': 'No alternative path found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Generate and serve the graph image with shortest & alternative paths
@app.route('/graph', methods=['GET'])
def generate_graph():
    start = request.args.get('start', '').upper()
    end = request.args.get('end', '').upper()

    if start not in graph.nodes or end not in graph.nodes:
        return jsonify({'error': 'Invalid airport code(s)'}), 400

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, seed=42)

    # Draw all nodes
    nx.draw(graph, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=10, font_weight='bold', arrows=True)

    # Draw all edges in gray
    nx.draw_networkx_edges(graph, pos, edge_color='gray', alpha=0.5)

    # Display edge weights (distances)
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, font_size=8)

    try:
        # Find the shortest path
        paths = list(islice(nx.shortest_simple_paths(graph, start, end, weight='weight'), 2))
        if paths:
            shortest_edges = list(zip(paths[0], paths[0][1:]))
            nx.draw_networkx_edges(graph, pos, edgelist=shortest_edges, edge_color='red', width=3, style="dashed")

        # Find alternative path (if available)
        if len(paths) > 1:
            alternative_edges = list(zip(paths[1], paths[1][1:]))
            nx.draw_networkx_edges(graph, pos, edgelist=alternative_edges, edge_color='green', width=2, style="dotted")

    except nx.NetworkXNoPath:
        return jsonify({'error': 'No path found'}), 404

    file_path = "graph.png"
    plt.savefig(file_path, format="png", dpi=300)
    plt.close()

    return send_file(file_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import islice

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load CSV and build graph
file_path = "C:/Users/RazaqHaidari/OneDrive/Desktop/2nd Semester/Design and analysisof algorithm/project/Flight-Route-Planner/indian_flights.csv"
data = pd.read_csv(file_path)

# Ensure Distance is numeric and remove duplicates
data['Distance'] = pd.to_numeric(data['Distance'], errors='coerce')
data = data.drop_duplicates(subset=['Origin_airport', 'Destination_airport'])

# Create a directed graph for flights
graph = nx.DiGraph()

# Add edges to the graph
for _, row in data.iterrows():
    graph.add_edge(row['Origin_airport'], row['Destination_airport'], weight=row['Distance'])

@app.route('/')
def index():
    return render_template('index.html')

# Find shortest and second shortest paths
@app.route('/find_alternative_paths', methods=['GET'])
def find_alternative_paths():
    start = request.args.get('start', '').upper()
    end = request.args.get('end', '').upper()

    if start not in graph.nodes or end not in graph.nodes:
        return jsonify({'error': 'Invalid airport code(s)'}), 400

    try:
        # Get all possible paths
        paths = list(islice(nx.shortest_simple_paths(graph, start, end, weight='weight'), 5))

        if not paths:
            return jsonify({'error': 'No path found'}), 404

        # Compute distances for each path
        path_distances = [(p, sum(graph[u][v]['weight'] for u, v in zip(p, p[1:]))) for p in paths]
        
        # Sort by distance
        path_distances = sorted(path_distances, key=lambda x: x[1])

        # Extract shortest and second shortest paths
        shortest_path, shortest_distance = path_distances[0]
        second_shortest = path_distances[1] if len(path_distances) > 1 else None

        return jsonify({
            'shortest_path': {'path': shortest_path, 'distance': shortest_distance},
            'alternative_route': {'path': second_shortest[0], 'distance': second_shortest[1]} if second_shortest else None
        })

    except nx.NetworkXNoPath:
        return jsonify({'error': 'No alternative path found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Generate and serve the graph image with shortest & alternative paths
@app.route('/graph', methods=['GET'])
def generate_graph():
    start = request.args.get('start', '').upper()
    end = request.args.get('end', '').upper()

    if start not in graph.nodes or end not in graph.nodes:
        return jsonify({'error': 'Invalid airport code(s)'}), 400

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, seed=42)

    # Draw all nodes
    nx.draw(graph, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=10, font_weight='bold', arrows=True)

    # Draw all edges in gray
    nx.draw_networkx_edges(graph, pos, edge_color='gray', alpha=0.5)

    # Display edge weights (distances)
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, font_size=8)

    try:
        # Find the shortest path
        paths = list(islice(nx.shortest_simple_paths(graph, start, end, weight='weight'), 2))
        if paths:
            shortest_edges = list(zip(paths[0], paths[0][1:]))
            nx.draw_networkx_edges(graph, pos, edgelist=shortest_edges, edge_color='red', width=3, style="dashed")

        # Find alternative path (if available)
        if len(paths) > 1:
            alternative_edges = list(zip(paths[1], paths[1][1:]))
            nx.draw_networkx_edges(graph, pos, edgelist=alternative_edges, edge_color='green', width=2, style="dotted")

    except nx.NetworkXNoPath:
        return jsonify({'error': 'No path found'}), 404

    file_path = "graph.png"
    plt.savefig(file_path, format="png", dpi=300)
    plt.close()

    return send_file(file_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
