import numpy as np
import matplotlib.pyplot as plt

# Parámetros de simulación
N = 4  # Número de partículas
dt = 0.01  # Paso de tiempo
num_pasos = 10000  # Número de iteraciones
epsilon = 1e-10  # Profundidad del pozo en Lennard-Jones
sigma = 4.0  # Distancia a la cual el potencial es cero
lado_cuadrado = 10.0  # Tamaño del cuadrado (10x10)
radio_p = 0.01
# Condiciones iniciales
# Generación de posiciones iniciales aleatorias con restricción de radio
posiciones = np.zeros((N, 2))
for i in range(N):
    while True:
        nueva_posicion = np.random.uniform(radio_p, lado_cuadrado/2-radio_p, 2)
        cumple_distancia = True
        for j in range(i):
            if np.linalg.norm(nueva_posicion - posiciones[j]) < radio_p:
                cumple_distancia = False
                break
        if cumple_distancia:
            posiciones[i] = nueva_posicion
            break
# Mostrar posiciones iniciales
plt.scatter(posiciones[:, 0], posiciones[:, 1], color="blue")
plt.xlim(0, lado_cuadrado)
plt.ylim(0, lado_cuadrado)
plt.title("Posiciones iniciales aleatorias")
plt.xlabel("Posición X")
plt.ylabel("Posición Y")
plt.grid(True)
plt.show()

velocidades = np.zeros((N, 2))  # Velocidades iniciales en cero
masas = 1e-4  # Masa de cada partícula (asumimos masa uniforme)

# Función de cálculo de fuerzas usando Lennard-Jones
def calcular_aceleraciones(posiciones):
    aceleraciones = np.zeros_like(posiciones)
    for i in range(N):
        for j in range(i + 1, N):
            # Calcula la distancia usando condiciones de frontera periódicas
            r_ij = posiciones[j] - posiciones[i]
            r_ij -= lado_cuadrado * np.round(r_ij / lado_cuadrado)  # Condiciones de frontera periódicas
            distancia = np.linalg.norm(r_ij)
            # Calcular fuerza de Lennard-Jones
            if distancia > 0 and distancia>radio_p:  # Fuerzas fuera del rango de colisión
                fuerza_magnitud = 24 * epsilon * ((2 * (sigma / distancia)**12) - ((sigma / distancia)**6)) / distancia**2
                fuerza = fuerza_magnitud * r_ij / distancia
                # Aplica la fuerza de acción y reacción
                aceleraciones[i] += fuerza / masas
                aceleraciones[j] -= fuerza / masas
    return aceleraciones

# Algoritmo de Velocidad de Verlet
for paso in range(num_pasos):
    # Calcula las aceleraciones iniciales
    aceleraciones = calcular_aceleraciones(posiciones)
    
    # Actualiza posiciones
    posiciones += velocidades * dt + 0.5 * aceleraciones * dt**2
    
    # Aplica condiciones de frontera periódicas
    posiciones = posiciones % lado_cuadrado  # Regresa al cuadrado, como en el juego de Pacman

    # Actualiza velocidades
    nuevas_aceleraciones = calcular_aceleraciones(posiciones)
    velocidades += 0.5 * (aceleraciones + nuevas_aceleraciones) * dt
    if paso % 10 == 0:
        plt.scatter(posiciones[:, 0], posiciones[:, 1], color="blue")
        plt.xlim(0, lado_cuadrado)
        plt.ylim(0, lado_cuadrado)
        plt.title(f"Paso {paso}")
        plt.xlabel("Posición X")
        plt.ylabel("Posición Y")
        plt.pause(0.1)
        plt.clf()
plt.show()

#Imprimimos las posiciones finales de las partículas
print("Posiciones finales de las partículas:")
print(posiciones)

