# Image Classifier API – Backend

# Integrantes del proyecto

* Josué Rojas
* Alejandro Solis

# Descripción detallada del proyecto

* El presente proyecto consiste en el desarrollo de una API basada en Inteligencia Artificial capaz de clasificar imágenes utilizando un modelo de aprendizaje profundo.
* El sistema recibe imágenes enviadas desde una aplicación web, las procesa mediante un modelo entrenado previamente y retorna una predicción junto con el nivel de confianza asociado.
* La API actúa como la capa intermedia entre la interfaz de usuario y el modelo de inteligencia artificial, permitiendo desacoplar la lógica de clasificación del frontend.

# Resumen teórico

* El proyecto se basa en técnicas de Computer Vision y Deep Learning, específicamente redes neuronales convolucionales (CNN), las cuales permiten identificar patrones visuales dentro de imágenes.
* El modelo fue entrenado utilizando TensorFlow/Keras, framework ampliamente utilizado en aplicaciones de inteligencia artificial debido a su eficiencia en procesamiento matricial y aprendizaje supervisado.

# La arquitectura cliente-servidor permite:

* Separación de responsabilidades
* Escalabilidad
* Reutilización del modelo IA mediante API REST
* Tecnologías utilizadas
* Python 3.11+
* FastAPI   
* TensorFlow / Keras
* Uvicorn
* SQLite
* UV (gestor de dependencias)
* REST API

# Diseño del sistema

* El backend sigue una arquitectura modular:
* Frontend → API REST (FastAPI) → Modelo IA → Resultado JSON

# Componentes principales:

* Router API: manejo de endpoints
* Loader IA: carga del modelo entrenado
* Base de datos: almacenamiento básico
* Servicio de clasificación

# Obstáculos encontrados

Durante el desarrollo se presentaron diversos retos técnicos:
Configuración del entorno de TensorFlow.
Manejo de CORS para permitir comunicación con el frontend.
Diferencias entre nombres de campos enviados en formularios multipart.
Gestión del tamaño del modelo entrenado.

# Conclusiones

* El uso de FastAPI permitió crear una API eficiente y rápida para integrar modelos de inteligencia artificial en aplicaciones web. La separación entre frontend y backend facilita futuras mejoras como reentrenamiento del modelo o despliegue en la nube.


# Instrucciones para ejecutar

* Clonar el repositorio / O abrir el zip:

```git clone <repo>```
```cd image-classifier-api```

* Descargar dependencias: * Instalar uv si no lo tienes *

```uv sync```

* Activar entorno: 

```.venv\Scripts\Activate.ps1```

* Ejecutar servidor

```.\dev.ps1```

* Abrir documentación:

# http://localhost:8000/docs
