import math
import sys
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import scipy
import folium

# Fórmula del Haversine para calcular la distancia entre dos puntos geográficos
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radio de la Tierra en kilómetros
    dLat = math.radians(lat2 - lat1)  # Diferencia de latitud en radianes
    dLon = math.radians(lon2 - lon1)  # Diferencia de longitud en radianes
    a = (math.sin(dLat / 2) ** 2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dLon / 2) ** 2)  # Fórmula de Haversine
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))  # Se calcula el angulo
    return R * c  # Resultado en km
# Definimos el grafo con los aeropuertos y sus distancias

def build_graph(df):
    graph = {}
    airport_data = {}  # Inicializamos los datos del Aeropuerto

    for index, row in df.iterrows():
        src_code = row['Source Airport Code']
        src_name = row['Source Airport Name']
        src_city = row['Source Airport City']
        src_country = row['Source Airport Country']
        src_lat = float(row['Source Airport Latitude'])
        src_lon = float(row['Source Airport Longitude'])

        dest_code = row['Destination Airport Code']
        dest_name = row['Destination Airport Name']
        dest_city = row['Destination Airport City']
        dest_country = row['Destination Airport Country']
        dest_lat = float(row['Destination Airport Latitude'])
        dest_lon = float(row['Destination Airport Longitude'])

        # Agregamos los datos del aeropuerto
        airport_data[src_code] = {
            'name': src_name,
            'city': src_city,
            'country': src_country,
            'latitude': src_lat,  # Changed from 'lat' to 'latitude'
            'longitude': src_lon  # Changed from 'lon' to 'longitude'
        }
        airport_data[dest_code] = {
            'name': dest_name,
            'city': dest_city,
            'country': dest_country,
            'latitude': dest_lat,  # Changed from 'lat' to 'latitude'
            'longitude': dest_lon   # Changed from 'lon' to 'longitude'
        }

        # Calculamos la distancia entre los dos aerpuertos
        distance = haversine(src_lat, src_lon, dest_lat, dest_lon)

        # Add to the graph
        if src_code not in graph:
            graph[src_code] = {'connections': {}}
        if dest_code not in graph:
            graph[dest_code] = {'connections': {}}

        graph[src_code]['connections'][dest_code] = distance
        graph[dest_code]['connections'][src_code] = distance  # Cuando el grafo no es dirigidp

    return graph, airport_data  # Return both graph and airport data


#  Algoritmo de Dijkstra
def dijkstra(graph, start):
    dist = {node: sys.maxsize for node in graph}  # Inicializar distancias
    dist[start] = 0
    visited = set()
    predecessors = {node: None for node in graph}  # Predecesores para reconstruir el camino

    while len(visited) < len(graph):
        current_node = None
        for node in dist:
            if node not in visited:
                if current_node is None or dist[node] < dist[current_node]:
                    current_node = node

        if current_node is None:
            break

        if current_node not in graph:
            print(f"Warning: {current_node} not found in graph.")
            continue  # Skip if current_node is not in the graph

        visited.add(current_node)

        for neighbor, weight in graph[current_node]['connections'].items():
            if neighbor not in visited:
                new_distance = dist[current_node] + weight
                if new_distance < dist[neighbor]:
                    dist[neighbor] = new_distance
                    predecessors[neighbor] = current_node  # Guardar el predecesor

    return dist, predecessors


# Funciones complementarias para el algoritmo de Kruskal (Union-Find)
def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])


def union(parent, rank, x, y):
    root_x = find(parent, x)
    root_y = find(parent, y)

    if rank[root_x] < rank[root_y]:
        parent[root_x] = root_y
    elif rank[root_x] > rank[root_y]:
        parent[root_y] = root_x
    else:
        parent[root_y] = root_x
        rank[root_x] += 1

#punto 1
#  (DFS) para encontrar los componentes de un grafo
def dfs(graph, node, visited, component):
    visited.add(node)  # Marcar el nodo como visitado
    component.append(node)  # Agregarlo a la componente actual
    for neighbor in graph[node]['connections']:
        if neighbor not in visited:
            dfs(graph, neighbor, visited, component)

# Función para verificar si el grafo es conexo y cuenta con las componentes
def find_components(graph):
    visited = set()
    components = []

    # Ejecutamos el DFS desde cada vértice no visitado
    for node in graph:
        if node not in visited:
            component = []
            dfs(graph, node, visited, component)
            components.append(component)

    # Determinar si el grafo es conexo
    is_connected = len(components) == 1
    return is_connected, components


# Función para imprimir si el grafo es conexo y mostrar las componentes conexas
def print_components_info(graph):
    is_connected, components = find_components(graph)

    if is_connected:
        print("El grafo es conexo.")
    else:
        print(f"El grafo no es conexo. Tiene {len(components)} componentes.")
        for i, component in enumerate(components):
            print(f"Componente {i + 1}: {len(component)} vértices -> {component}")


#punto 2
# Clase para representar una arista entre dos nodos
class Edge:
    def __init__(self, u, v, weight):
        self.u = u  # Nodo origen
        self.v = v  # Nodo destino
        self.weight = weight  # Peso de la arista (distancia)


# Función para encontrar el representante del conjunto de un nodo
def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])


# Función para unir dos conjuntos (Union-Find)
def union(parent, rank, x, y):
    root_x = find(parent, x)
    root_y = find(parent, y)

    if rank[root_x] < rank[root_y]:
        parent[root_x] = root_y
    elif rank[root_x] > rank[root_y]:
        parent[root_y] = root_x
    else:
        parent[root_y] = root_x
        rank[root_x] += 1


# Algoritmo de Kruskal para encontrar el MST de una componente
def kruskal(component, edges):
    parent = {}
    rank = {}

    # Inicializamos los conjuntos disjuntos
    for node in component:
        parent[node] = node
        rank[node] = 0

    # Ordenamos las aristas por peso (distancia)
    edges = sorted(edges, key=lambda edge: edge.weight)

    mst_weight = 0  # Peso total del árbol de expansión mínima
    mst_edges = []  # Aristas en el MST

    # Iterar sobre las aristas y agregarlas al MST si no forman ciclos
    for edge in edges:
        u, v, weight = edge.u, edge.v, edge.weight
        root_u = find(parent, u)
        root_v = find(parent, v)

        if root_u != root_v:
            mst_edges.append(edge)
            mst_weight += weight
            union(parent, rank, root_u, root_v)

    return mst_weight, mst_edges


# Función para encontrar el MST para todas las componentes conexas del grafo
def find_mst_of_components(graph, components, edges):
    for i, component in enumerate(components):
        # Filtrar las aristas que están dentro de la componente actual
        component_edges = [edge for edge in edges if edge.u in component and edge.v in component]

        # Aplicamos Kruskal para esta componente
        mst_weight, mst_edges = kruskal(component, component_edges)

        print(f"Componente {i + 1}:")
        print(f"Peso del Árbol de Expansión Mínima: {mst_weight}")
        print(f"Aristas del MST: {[(edge.u, edge.v, edge.weight) for edge in mst_edges]}")
        print("")

#punto 3
# Función para mostrar la información del aeropuerto
def show_airport_info(airport_code, airport_data):
    data = airport_data.get(airport_code)
    if data:
        print(f"Código: {airport_code}")
        print(f"Nombre: {data['name']}")
        print(f"Ciudad: {data['city']}")
        print(f"País: {data['country']}")
        print(f"Latitud: {data['latitude']}")
        print(f"Longitud: {data['longitude']}")
        print("-" * 40)
    else:
        print(f"Aeropuerto {airport_code} no encontrado en los datos.")


# Función principal para resolver el problema
def longest_paths_from_airport(start_airport_code, graph, airport_data):
    # Mostrar la información del aeropuerto de origen
    print("Información del aeropuerto de origen:")
    show_airport_info(start_airport_code, airport_data)

    # Ejecutar Dijkstra desde el aeropuerto de origen y obtener distancias y predecesores
    distances, predecessors = dijkstra(graph, start_airport_code)

    # Ordenamos los aeropuertos por distancia (de mayor a menor)
    sorted_distances = sorted(distances.items(), key=lambda x: x[1], reverse=True)

    # Mostramos la información de los 10 aeropuertos más lejanos
    print("Los 10 aeropuertos con caminos más largos:")
    count = 0
    for airport_code, distance in sorted_distances:
        if airport_code != start_airport_code and distance < sys.maxsize:
            count += 1
            print(f"\nAeropuerto {count}:")
            show_airport_info(airport_code, airport_data)
            print(f"Distancia: {distance} km")
            print("=" * 40)
        if count == 10:
            break

#punto 4
def reconstruct_path(predecessors, start, end):
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = predecessors[current]

    path.reverse()  # El camino se reconstruye desde el destino, así que lo invertimos
    return path


# graficamos
def plot_graph(graph, airport_data):
    G = nx.Graph()

    # Agregamos nodos y aristaa al grafo
    for airport_code, data in airport_data.items():
        G.add_node(airport_code, label=data['name'])

    for src_code, connections in graph.items():
        for dest_code, distance in connections['connections'].items():
            G.add_edge(src_code, dest_code, weight=distance)

    pos = nx.spring_layout(G)  # Layout for the graph

    # Dibujamos el nodo
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue', alpha=0.7)

    # Dibujamos el nodo
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    labels = {airport_code: data['name'] for airport_code, data in airport_data.items()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    plt.title("Graph of Airports and Connections")
    plt.axis('off')
    plt.show()
# Función para graficar la geolocalización de los aeropuertos y rutas
def plot_map(graph, airport_data):
    # Crear un mapa centrado en una ubicación inicial (puedes elegir cualquier aeropuerto para centrar el mapa)
    initial_airport = next(iter(airport_data.values()))  # Toma el primer aeropuerto como referencia
    center_lat = initial_airport['latitude']
    center_lon = initial_airport['longitude']
    m = folium.Map(location=[center_lat, center_lon], zoom_start=3)

    # Añadir marcadores de los aeropuertos al mapa
    for code, data in airport_data.items():
        folium.Marker(
            location=[data['latitude'], data['longitude']],
            popup=f"{data['name']} ({code})\n{data['city']}, {data['country']}",
            icon=folium.Icon(color='blue', icon='plane', prefix='fa')
        ).add_to(m)

    # Añadir las líneas que representan las rutas entre los aeropuertos
    for src_code, connections in graph.items():
        src_airport = airport_data[src_code]
        for dest_code, distance in connections['connections'].items():
            dest_airport = airport_data[dest_code]
            # Dibujar la línea entre los aeropuertos
            folium.PolyLine(
                locations=[
                    [src_airport['latitude'], src_airport['longitude']],
                    [dest_airport['latitude'], dest_airport['longitude']]
                ],
                color='green',
                weight=1.5,
                opacity=0.7
            ).add_to(m)

    # Guardar el mapa en un archivo HTML y abrirlo en el navegador
    m.save("airports_map.html")
    print("Mapa guardado en 'airports_map.html'.")