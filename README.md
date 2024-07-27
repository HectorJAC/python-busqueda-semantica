# Hector José Arámboles Castillo. 2019-0821  
## Proyecto Final: Sistema de Búsqueda Semántica de Artículos Científicos  
### Asignatura: Inteligencia Artificial  
### Profesor: Lizandro José Ramírez Difo  

### Estuctura del proyecto  
El proyecto se encuentra estructurado de la siguiente manera:  
1. **data**: Carpeta que contiene el dataset minimizado de artículos científicos en formato CSV que se ha utilizado para la creación del modelo de búsqueda semántica.  
2. **dataset-clean.py**: Archivo .py con el código que se ha utilizado para limpiar el dataset original de artículos científicos.  
3. **script.py**: Archivo .py con el código que se ha utilizado para la creación del modelo de búsqueda semántica.  
4. **README.md**: Archivo que contiene la descripción del proyecto.  

### Descripción del proyecto  
El proyecto consiste en la creación de un sistema de búsqueda semántica de artículos científicos. Para ello, se ha utilizado un dataset de artículos científicos que ha sido limpiado y minimizado. Posteriormente, se ha creado un modelo de búsqueda semántica que permite realizar búsquedas de artículos científicos a partir de una consulta de texto.  

**Librerias utilizadas:**  
``Pandas``: Librería de manipulación y análisis de datos.  
``Numpy``: Librería de cálculo numérico.  
``Matplotlib``: Librería de visualización de datos.  
``Seaborn``: Librería de visualización de datos basada en Matplotlib.  
``Sklearn``: Librería de aprendizaje automático.  
``Transformers``: Librería de modelos de lenguaje de aprendizaje profundo.  
``Torch``: Librería de aprendizaje profundo.  
``Time``: Librería de manipulación de fechas y horas.  
``Threading``: Librería de programación concurrente.

### Instrucciones de instalación  
Para ejecutar el proyecto, se deben seguir los siguientes pasos:  
1. Instalar las librerías necesarias utilizadas en el archivo script.py:  

``pip install pandas numpy matplotlib seaborn sklearn transformers torch``  

### Guía de Uso  
1. Ejecutar el archivo script.py.  
2. Esperar a que cargue el dataset, a que el preprocesamiento se complete y que el modelo y los embeddings se carguen.  
3. Ingresar lo que se desea buscar.  
4. Esperar a que el sistema devuelva los resultados.
5. Se creara una archivo ``embeddings_visualization.png`` con la visualización de **10** de los resultados obtenidos.  

### Enlace al vídeo explicativo
