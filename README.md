# Deep Learning - Predicción de Engagement en POIs Turísticos

Este proyecto corresponde a la práctica de Deep Learning del Bootcamp.  
El objetivo es **predecir el nivel de engagement** de puntos de interés turísticos (POIs) utilizando un enfoque **multimodal**: imágenes y metadatos.

# Deep Learning Project - Tourist POIs

Este proyecto integra datos multimodales de puntos de interés turísticos (POIs):
- Texto: nombre, descripción, categorías, tags
- Datos estructurados: visitas, likes, dislikes, bookmarks, XP, coordenadas
- Imágenes: almacenadas externamente en Google Drive

---

## 📂 Dataset de imágenes

Las imágenes no están en GitHub. Descárgalas desde Google Drive:

- Carpeta completa: [Google Drive Folder](https://drive.google.com/drive/folders/18aZd5ZusAyCIRYMPq4t90014Ybmtam88?usp=drive_link)  
- Archivo comprimido (.zip): [data_main.zip](https://drive.google.com/file/d/1Zccp97gB8WZE15Uo5cbekvB8cv66z8sX/view?usp=drive_link)

---

## 🔧 Uso en Colab

```python
!pip install gdown
!gdown --id 1Zccp97gB8WZE15Uo5cbekvB8cv66z8sX -O data/data_main.zip
!unzip data/data_main.zip -d data/data_main/

El acceso está restringido: solo usuarios autorizados por correo electrónico podrán descargar las imágenes.
---

## 📂 Estructura del repositorio
deep_learning/
├── data/
│   ├── poi_dataset.csv         # Dataset principal
│   └── data_main/              # Carpeta con imágenes
├── notebooks/
│   └── Practica_Deep_Learning_Fabian_Lopez.ipynb
├── scripts/
│   ├── preprocess.py           # Preprocesamiento de dataset
│   └── module_utils.py         # Funciones auxiliares
├── requirements.txt            # Dependencias del proyecto
└── README.md                   # Documentación del proyecto
└── Memoria_Técnica_Detallada_Práctica_Fabian_López.pdf    # Documentación de la memoria técnica
---

## 🚀 Reproducibilidad

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/Fablop84/deep_learning.git
   cd deep_learningcd deep_learning

Instalar dependencias

- pip install -r requirements.txt

Ejecutar el notebook
- Abre notebooks/Practica_Deep_Learning_Fabian_Lopez.ipynb en Google Colab.
- Conéctate a GPU (T4).
- Ejecuta todas las celdas.

---

## Entregables**

- Notebook reproducible con código comentado.
- Memoria técnica en PDF.
- Modelo entrenado final (`final_model.pth`).
- Scripts auxiliares (`preprocess.py`, `module_utils.py`).

**Autor**: Fabian Camilo López  
**Fecha**: Enero 2026  
**Bootcamp**: Deep Learning - Práctica Final

