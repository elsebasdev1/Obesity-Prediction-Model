# 🏥 Predicción de Niveles de Obesidad: Análisis Multivariado y MLOps

Este proyecto desarrolla un pipeline de Machine Learning para predecir niveles de obesidad basándose en hábitos alimenticios y condición física, superando la simple métrica del IMC. Se implementa un enfoque comparativo entre modelos deterministas (Random Forest) y modelos estocásticos con reducción de dimensionalidad (PCA + Redes Neuronales), gestionado todo el ciclo de vida mediante MLflow.

## 📝 Descripción del problema
El Índice de Masa Corporal (IMC) a menudo falla al no considerar factores de estilo de vida. El objetivo de este proyecto es clasificar a los pacientes en 7 niveles de salud (desde "Bajo Peso" hasta "Obesidad Tipo III") utilizando datos demográficos, hábitos de transporte, alimentación y genética.

El desafío técnico principal consiste en aplicar **Análisis Multivariado** para reducir la dimensionalidad de los datos y optimizar un **Modelo Estocástico (Red Neuronal)** para encontrar el equilibrio entre complejidad y precisión.

## 🛠 Stack Tecnológico
* **Lenguaje:** Python 3.x
* **Procesamiento de Datos:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Random Forest, MLPClassifier, PCA)
* **MLOps & Tracking:** MLflow
* **Visualización:** Matplotlib, Seaborn

## 🏗 Arquitectura -> Fases

El proyecto se estructura en un pipeline secuencial:

1.  **Ingesta y ETL:** Carga de datos desde Excel (`.xlsx`), limpieza de nombres de columnas y conversión de variables categóricas nominales (One-Hot Encoding para transporte) y ordinales.
2.  **Modelo Base (Benchmark):** Entrenamiento de un **Random Forest** para establecer una línea base de rendimiento en datos tabulares crudos.
3.  **Pipeline Estocástico (PCA + NN):**
    * **Estandarización:** Scaling de datos (`StandardScaler`).
    * **Análisis Multivariado:** Aplicación de **PCA** (Análisis de Componentes Principales) dinámico para retener el 90-95% de la varianza.
    * **Modelado:** Red Neuronal Perceptrón Multicapa (`MLPClassifier`).
4.  **Optimización (Fine-Tuning):** Búsqueda de hiperparámetros (`GridSearchCV`) probando arquitecturas neuronales y penalizaciones, con registro automático de experimentos en **MLflow**.

## 📸 Capturas

### 1. Rendimiento del Modelo Estocástico (Arquitectura vs PCA)
<img width="715" height="407" alt="image" src="https://github.com/user-attachments/assets/8d5aad0e-9ec5-44bb-a6a9-0e8ecf1624f6" />
<img width="732" height="421" alt="image" src="https://github.com/user-attachments/assets/d988e235-96d6-46ba-8adf-ba99449cb862" />
<img width="845" height="549" alt="image" src="https://github.com/user-attachments/assets/c38ecc28-be3a-464e-b34b-6e991797707d" />


### 2. Estabilidad del PCA
<img width="800" height="600" alt="estabilidad_pca" src="https://github.com/user-attachments/assets/73a3a9e9-7e3a-459a-904e-4ef055edef62" />

### 3. Matriz de Confusión
<img width="783" height="564" alt="image" src="https://github.com/user-attachments/assets/2b7e14e8-66ba-4e37-9ada-86481417c638" />

### 4. Tracking en MLflow
<img width="2308" height="623" alt="image" src="https://github.com/user-attachments/assets/01b05866-55ee-46ee-96ab-d7db471b3e05" />

## 🚀 Qué lograste
* Implementación exitosa de un **Pipeline de Sklearn** que integra preprocesamiento, reducción de dimensiones y predicción en un solo objeto serializable.
* Análisis profundo de la estructura de los datos: Se descubrió una **baja multicolinealidad**, demostrando que el PCA requiere retener casi todas las componentes (15 de 15) para explicar el 95% de la varianza.
* Comparativa técnica: Se evidenció que para este dataset tabular específico, el modelo de ensamblaje (Random Forest) supera en precisión (~90%) a la Red Neuronal con PCA (~76%), debido a la naturaleza categórica de las variables.
* Documentación automática de experimentos utilizando **MLflow**.

## 💻 Código (Snippet del Pipeline)

```python
# Definición del Pipeline Estocástico para MLOps
pipeline = Pipeline([
    ('scaler', StandardScaler()),       # Estandarización obligatoria para PCA
    ('pca', PCA()),                     # Reducción de dimensionalidad
    ('nn', MLPClassifier(max_iter=500)) # Modelo Estocástico
])

# Espacio de búsqueda para Fine-Tuning
param_grid = {
    'pca__n_components': [0.90, 0.95],          # Varianza explicada
    'nn__hidden_layer_sizes': [(50,), (100,)],  # Arquitecturas
    'nn__activation': ['tanh', 'relu']
}
