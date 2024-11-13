import numpy as np
import matplotlib.pyplot as plt

# Parámetros del sistema
N = 4  # Número de partículas
lado_cuadrado = 10.0  # Tamaño del cuadrado
radio_p = 0.01  # Distancia mínima entre partículas
masa = 1e-4  # Masa de cada partícula
epsilon = 1e-10  # Profundidad del pozo de Lennard-Jones
sigma = 4.0  # Distancia a la cual el potencial es cero
dt = 0.01  # Paso de tiempo
num_steps = 1000  # Número de pasos de simulación
temperatura = 1000.0  # Temperatura objetivo del sistema

# Generación de posiciones iniciales aleatorias con restricción de radio
posiciones = np.zeros((N, 2))
for i in range(N):
    while True:
        nueva_posicion = np.random.uniform(0, lado_cuadrado, 2)
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

# Generación de velocidades iniciales con distribución de Maxwell-Boltzmann
velocidades = np.random.normal(0, np.sqrt(temperatura / masa), (N, 2))

# Función para calcular fuerzas de Lennard-Jones con condiciones periódicas
def calcular_fuerzas(posiciones):
    fuerzas = np.zeros((N, 2))  # Fuerzas sobre cada partícula
    for i in range(N):
        for j in range(i + 1, N):
            # Vector de distancia con condiciones periódicas
            delta = posiciones[j] - posiciones[i]
            delta = delta - lado_cuadrado * np.round(delta / lado_cuadrado)
            distancia = np.linalg.norm(delta)
            
            # Cálculo de la fuerza de Lennard-Jones
            if distancia < lado_cuadrado / 2 and distancia> radio_p:
                fuerza_magnitud = 24 * epsilon * (2 * (sigma / distancia)**12 - (sigma / distancia)**6) / distancia**2
                fuerza = fuerza_magnitud * delta
                fuerzas[i] += fuerza
                fuerzas[j] -= fuerza
    return fuerzas  # Retornamos las fuerzas en lugar de fuerzas/masa

# Simulación con el método de Verlet de velocidades
for step in range(num_steps):
    # Calcular fuerzas iniciales
    fuerzas = calcular_fuerzas(posiciones)
    
    # Calcular aceleraciones
    aceleraciones = fuerzas / masa  # Aceleraciones directamente

    # Actualización de posiciones
    nuevas_posiciones = posiciones + velocidades * dt + 0.5 * aceleraciones * dt**2
    
    # Condiciones de frontera periódicas
    nuevas_posiciones = nuevas_posiciones % lado_cuadrado
    
    # Calcular nuevas fuerzas después de actualizar posiciones
    nuevas_fuerzas = calcular_fuerzas(nuevas_posiciones)
    
    # Calcular nuevas aceleraciones
    nuevas_aceleraciones = nuevas_fuerzas / masa  # Nuevas aceleraciones

    # Actualización de velocidades
    velocidades += 0.5 * (aceleraciones + nuevas_aceleraciones) * dt
    
    # Actualizar posiciones para el próximo paso
    posiciones = nuevas_posiciones

#Imprimimos las posiciones finales de las partículas
print("Posiciones finales de las partículas:")
print(posiciones)

# Graficar las posiciones finales de las partículas
plt.scatter(posiciones[:, 0], posiciones[:, 1], color="blue", label="Posiciones Finales")
for i in range(N):
    plt.text(posiciones[i, 0], posiciones[i, 1], f"{i+1}", ha='right', color="blue")

plt.xlim(0, lado_cuadrado)
plt.ylim(0, lado_cuadrado)
plt.xlabel("Posición X")
plt.ylabel("Posición Y")
plt.title("Posiciones finales de las partículas")
plt.legend()
plt.grid()
plt.show()
