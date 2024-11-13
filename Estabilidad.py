import numpy as np
import matplotlib.pyplot as plt

# Parámetros de simulación
N = 4  # Número de partículas
dt = 0.001  # Paso de tiempo
num_steps = 100000  # Número de iteraciones
epsilon = 1e-10  # Profundidad del pozo en Lennard-Jones
sigma = 4.0  # Distancia a la cual el potencial es cero
lado_cuadrado = 10.0  # Tamaño del cuadrado
radio_p = 0.01  # Radio de exclusión
masas = 1e-4  # Masa de cada partícula (asumimos masa uniforme)
k_B = 1e-5
T = 300

# Condiciones iniciales
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

std_dev = np.sqrt(k_B * T / masas)  # Desviación estándar para la distribución
velocidades = np.random.normal(0, std_dev, (N, 2))  # Velocidades iniciales aleatorias

# Función de cálculo de aceleraciones y energía potencial usando Lennard-Jones
def calcular_aceleraciones_y_energia(posiciones):
    aceleraciones = np.zeros_like(posiciones)
    energia_potencial = 0
    for i in range(N):
        for j in range(i + 1, N):
            r_ij = posiciones[j] - posiciones[i]
            r_ij -= lado_cuadrado * np.round(r_ij / lado_cuadrado)  # Condiciones de frontera periódicas
            distancia = np.linalg.norm(r_ij)
            if distancia > 0 and distancia > radio_p:  # Evitar colisiones cercanas
                # Calcular la fuerza de Lennard-Jones
                fuerza_magnitud = 24 * epsilon * ((2 * (sigma / distancia)**12) - ((sigma / distancia)**6)) / distancia**2
                fuerza = fuerza_magnitud * r_ij / distancia
                aceleraciones[i] += fuerza / masas
                aceleraciones[j] -= fuerza / masas
                # Sumar la energía potencial entre partículas
                energia_potencial += 4 * epsilon * ((sigma / distancia)**12 - (sigma / distancia)**6)
    return aceleraciones, energia_potencial


# Almacenar temperatura promedio en cada paso
temperaturas_promedio = []

# Algoritmo de Velocidad de Verlet
for paso in range(num_steps):
    # Calcula las aceleraciones y la energía potencial
    aceleraciones, energia_potencial = calcular_aceleraciones_y_energia(posiciones)

    # Actualiza posiciones
    posiciones += velocidades * dt + 0.5 * aceleraciones * dt**2
    
    # Aplica condiciones de frontera periódicas
    posiciones = posiciones % lado_cuadrado

    # Calcula nuevas aceleraciones para el próximo paso
    nuevas_aceleraciones, _ = calcular_aceleraciones_y_energia(posiciones)
    
    # Actualiza velocidades
    velocidades += 0.5 * (aceleraciones + nuevas_aceleraciones) * dt

    # Calcular energía cinética media y temperatura promedio
    energia_kinetica_media = 0.5 * masas * np.mean(np.sum(velocidades**2, axis=1))
    temperatura = 2 * energia_kinetica_media / (k_B * 2)  # Factor 2 por los grados de libertad
    temperaturas_promedio.append(temperatura)

tiempo = np.arange(num_steps) * dt
# Graficar la temperatura promedio en función del tiempo
plt.plot(tiempo, temperaturas_promedio, label="Temperatura Promedio", color="green")
plt.title("Temperatura promedio del sistema en el tiempo")
plt.xlabel("Paso de tiempo")
plt.ylabel("Temperatura")
plt.legend()
plt.grid(True)
plt.show()

# Imprimir posiciones iniciales y finales
print("Posiciones iniciales:")
print(np.array([[0.1, 0.1], [9.9, 0.1], [0.1, 9.9], [9.9, 9.9]]))
print("Posiciones finales de las partículas:")
print(posiciones)
