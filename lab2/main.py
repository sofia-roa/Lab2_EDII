import pandas as pd
import sys
from code import *
import matplotlib.pyplot as plt
import scipy

def main():
    # Cargamos los datos de los aeropuertos desde el archivo CSV
    df = pd.read_csv('flights_final.csv')
    graph, airport_data = build_graph(df)

    # Función para ejecutar el menú
    def main_menu(graph, airport_data):
        while True:
            print("\nMenú:")
            print("1. Determinar si el grafo generado es conexo.")
            print("2. Determinar el peso del árbol de expansión mínima.")
            print("3. Mostrar información de aeropuertos.")
            print("4. Mostrar camino mínimo entre dos aeropuertos.")
            print("5. Mostrar grafo abstracto.")
            print("6. Mostrar mapa interactivo de aeropuertos y rutas.")
            print("7. Salir.")

            option = input("Elija una opción (1-7): ")

            if option == '1':
                print_components_info(graph)


            elif option == '2':
                edges = []
                # Recolectamos todas las aristas del grafo para el algoritmo de Kruskal
                for src in graph:
                    for dest, weight in graph[src]['connections'].items():
                        edges.append(Edge(src, dest, weight))
                components = find_components(graph)[1]
                find_mst_of_components(graph, components, edges)


            elif option == '3':
                airport_code = input("Ingrese el código del aeropuerto: ")
                show_airport_info(airport_code, airport_data)

                start_airport_code = input("Ingrese el código del aeropuerto de inicio para ver los más lejanos: ")
                longest_paths_from_airport(start_airport_code, graph, airport_data)


            elif option == '4':
                start_airport_code = input("Ingrese el código del primer aeropuerto: ")
                end_airport_code = input("Ingrese el código del segundo aeropuerto: ")

                distances, predecessors = dijkstra(graph, start_airport_code)
                if end_airport_code in distances and distances[end_airport_code] < sys.maxsize:
                    path = reconstruct_path(predecessors, start_airport_code, end_airport_code)
                    print("Camino mínimo desde {} a {}: {}".format(start_airport_code, end_airport_code, " -> ".join(path)))
                    for airport_code in path:
                        show_airport_info(airport_code, airport_data)
                else:
                    print(f"No se encontró un camino desde {start_airport_code} a {end_airport_code}.")


            elif option == '5':
                plot_graph(graph, airport_data)

            elif option == '6':
                # Mostrar el mapa interactivo con la geolocalización de los aeropuertos y rutas
                plot_map(graph, airport_data)
                print("Mapa generado y guardado en 'airports_map.html'. Ábrelo en tu navegador.")

            elif option == '7':
                print("Saliendo del programa...")
                break

            else:
                print("Opción no válida. Intente de nuevo.")

    # Ejecutamos el menú principal con los datos del grafo y los aeropuertos
    main_menu(graph, airport_data)

if __name__ == "__main__":
    main()
