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
*(Muestra cómo diferentes configuraciones de neuronas y varianza PCA afectan la precisión)*
![Gráfico de Barras Rendimiento]([Aquí va la captura de "grafico_rendimiento_modelos.png" generado en el bloque maestro])

### 2. Estabilidad del PCA
*(Boxplot que muestra la variabilidad de la precisión según la compresión de datos)*
![Boxplot PCA]([Aquí va la captura de "grafico_estabilidad_pca.png"])

### 3. Matriz de Confusión
*(Análisis de errores por clase: Bajo Peso, Normal, Sobrepeso, Obesidad)*
![Matriz Confusión]([Aquí va la captura de tu matriz de confusión])

### 4. Tracking en MLflow
*(Vista del dashboard de MLflow registrando los experimentos)*
![Dashboard MLflow]([Aquí va una captura de pantalla de la interfaz de MLflow o la carpeta de artefactos])

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
