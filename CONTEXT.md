# Qsimov — Contexto completo del proyecto

> **Propósito de este archivo**: Descripción exhaustiva del proyecto leída archivo por archivo. Sirve para que futuros agentes AI entiendan el proyecto sin necesidad de re-leer el código.

---

## 1. ¿Qué es Qsimov?

Qsimov es una **librería Python para redes neuronales feed-forward con activación ReLU**. Su objetivo es reemplazar un subconjunto de las últimas capas de una red neuronal pre-entrenada por un sistema equivalente más simple que:

- **No sufre de olvido catastrófico** (catastrophic forgetting) al re-entrenar.
- Permite **re-entrenamiento incremental extremadamente rápido**.
- Separa la información **estructural** (almacenada en el `PathSelector`) de la información **cuantitativa** (almacenada como pesos de caminos).

**Contexto académico**: Es un TFG (Trabajo de Fin de Grado). El directorio raíz es `/home/mati/TFG2.0/qsimov/`.

---

## 2. Concepto matemático central

La idea central del algoritmo es la siguiente:

### 2.1 Caminos (Paths)
En una red neuronal con activación ReLU, para una entrada dada, algunos neuronas tienen activación > 0 (activas) y otras = 0 (inactivas). El **camino activo** de una muestra es la combinación de conexiones de neurona a neurona que permanecen activas desde la entrada hasta la salida.

### 2.2 Path Selector
El `PathSelector` divide la red en:
- **Left model**: capas que se aplican normalmente (sin path selection).
- **Right model**: las capas donde se aplica el algoritmo Qsimov.

Para el right model, se enumeran **todos los caminos posibles** entre capas (combinaciones de conexiones activas dado el peso no-nulo entre neuronas). Para una muestra concreta, solo los caminos cuya neurona de entrada es no-nula se activan.

### 2.3 Sistema lineal
El algoritmo convierte el problema de re-entrenamiento en un **sistema de ecuaciones lineales** `Ax = b`:
- `A`: coeficientes generados por `samples_to_coefficients()` — para cada muestra, el valor de la neurona de entrada asociada a cada camino activo.
- `b`: salida deseada.
- `x`: pesos de cada camino.

Se resuelve via QR + back-substitution o least-squares (lstsq).

### 2.4 Gradiente (alternativa)
En lugar del sistema lineal, el algoritmo también puede usar **descenso de gradiente** sobre una capa densa custom (`CustomConnectedDense` / `CustomConnectedLinear`) que tiene exactamente los mismos caminos como conexiones permitidas.

---

## 3. Estructura del repositorio

```
qsimov/
├── qsimov/               ← Librería principal (paquete Python)
│   ├── __init__.py
│   ├── typing.py         ← Type aliases (arrayNd, array1d, etc.)
│   ├── mixins.py         ← LogMixin, NumpyPersistanceMixin
│   ├── linalg.py         ← Álgebra lineal (QR, back-substitution, solve)
│   ├── path_selector.py  ← Clase abstracta PathSelector + BaseLayerTypes
│   ├── keras_path_selector.py    ← KerasPathSelector (TF/Keras)
│   ├── pytorch_path_selector.py  ← PytorchPathSelector (PyTorch)
│   ├── qsimov_linear_system.py   ← Clase abstracta QsimovLinearSystem
│   ├── keras_qsimov_linear_system.py    ← KerasQsimovLinearSystem
│   ├── pytorch_qsimov_linear_system.py  ← PytorchQsimovLinearSystem
│   ├── keras_qsimov_gradient.py         ← KerasQsimovGradient
│   ├── pytorch_qsimov_gradient.py       ← PytorchQsimovGradient
│   ├── c_linalg.pyx      ← Extensión Cython (back-substitution, make_square_system)
│   └── paths/
│       ├── paths.py       ← sort_paths, retrieve_coefficients, non_zero_input_select_paths
│       ├── combine.py     ← combine_paths, compute_combine_paths_output_size
│       ├── dense.py       ← get_all_paths_dense_layer
│       ├── conv.py        ← get_all_paths_conv_layer (1D/2D/3D)
│       ├── maxpooling.py  ← get_all_paths_maxpool_layer
│       ├── c_paths.pyx    ← Cython: c_non_zero_input_select_paths, c_retrieve_coefficients
│       ├── c_combine.pyx  ← Cython: c_combine_paths_left_right_sort_join
│       ├── c_dense.pyx    ← Cython: c_get_all_paths_dense_layer
│       ├── c_conv.pyx     ← Cython: conv1d/2d/3d paths
│       └── c_maxpooling.pyx ← Cython: maxpool paths
├── experiments/          ← Experimentos con la librería
│   ├── cifar10_gradient_by_splits/   ← CIFAR-10 con gradiente, varias divisiones
│   ├── imagenet_subset_by_splits/    ← ImageNet subset, Keras
│   ├── mnist_learning_rate/          ← MNIST, comparativa learning rates
│   └── mnist_speed_loss/             ← MNIST, velocidad vs pérdida (Keras + PyTorch)
├── tests/                ← Suite de tests (pytest)
│   ├── conftest.py
│   ├── nn_mocks_keras.py     ← Modelos Keras de prueba
│   ├── nn_mocks_pytorch.py   ← Modelos PyTorch de prueba
│   ├── paths/                ← Tests unitarios del módulo paths
│   ├── path_selector/        ← Tests del PathSelector (Keras vs PyTorch)
│   ├── qsimov_gradient/      ← Tests del QsimovGradient
│   └── qsimov_linear_system/ ← Tests del QsimovLinearSystem
├── tutorials/
│   └── api/
│       ├── tf_keras/   ← Ejemplos: qsimov_linear_system.py, qsimov_gradient.py
│       └── pytorch/    ← Ejemplos: qsimov_linear_system.py, qsimov_gradient.py
├── setup.py             ← Build Cython extensions
├── pytest.ini
├── requirements.txt     ← numpy, tensorflow, torch, Cython, pytest
└── requirements_dev.txt
```

---

## 4. Jerarquía de clases

### 4.1 PathSelector (abstracto)
**Archivo**: `qsimov/path_selector.py`

```
PathSelector (ABC, LogMixin, NumpyPersistanceMixin)
├── KerasPathSelector       ← usa TensorFlow/Keras
└── PytorchPathSelector     ← usa PyTorch (requiere input_shape extra)
```

**Atributos públicos clave**:
- `output_masks_`: array2d bool — máscara que indica qué coeficientes corresponden a cada output.
- `left_model_`: modelo hasta `initial_layer` (None si initial_layer=0).
- `right_model_`: modelo desde `initial_layer` hasta el final.

**Método clave**:
- `samples_to_coefficients(X)`: mapea muestras X a coeficientes del sistema lineal.
- `as_numpy_iterator(X, Y, batch_size)`: itera por batches.

**Compilación interna** (en `__init__`):
1. Aplana el modelo (flatten nested Sequentials).
2. Divide en left/right model.
3. Recupera tipos de capas (`_layer_types`).
4. Calcula todos los caminos (`_compute_all_paths`).
5. Calcula índices de transformación (`_make_transformation_indices`).

### 4.2 QsimovLinearSystem (abstracto)
**Archivo**: `qsimov/qsimov_linear_system.py`

```
QsimovLinearSystem (ABC, LogMixin, NumpyPersistanceMixin)
├── KerasQsimovLinearSystem
└── PytorchQsimovLinearSystem
```

**API pública**:
- `fit(X, Y, batch_size)`: genera ecuaciones y las resuelve.
- `predict(X, batch_size)`: predice usando `solutions_`.
- `reset_equations()`: resetea el sistema.
- `save(directory_path)` / `load(directory_path)`: persistencia.

**Atributos**:
- `equations_`: lista de sistemas lineales por output (shape R-transformada).
- `solutions_`: lista de soluciones por output.

### 4.3 QsimovGradient
**Archivos**: `keras_qsimov_gradient.py`, `pytorch_qsimov_gradient.py`

```
KerasQsimovGradient (LogMixin, NumpyPersistanceMixin)
PytorchQsimovGradient (LogMixin, NumpyPersistanceMixin)
```

**API pública** (Keras):
- `compile(optimizer, loss, device, ...)`: crea y compila el modelo interno.
- `fit(X, Y, **kwargs)`: mapea X a coeficientes y entrena con Keras API.
- `predict(X, batch_size)`.
- `save` / `load`.

**API pública** (PyTorch):
- `fit(X, Y, X_val, Y_val, batch_size, shuffle, **kwargs)`: acepta `training_loop` customizable.
- `predict(X, batch_size, device)`.
- `save` / `load`.

**Atributo público**:
- `model_`: Sequential de 1 capa (`CustomConnectedDense` / `CustomConnectedLinear`).

---

## 5. Tipos de capas soportadas

### Keras (`KerasLayerTypes`)
- **Parámetros (path)**: Dense, Conv1D, Conv2D, Conv3D
- **MaxPooling**: MaxPooling1D, MaxPooling2D, MaxPooling3D
- **Activación soportada para path selection**: ReLU, Activation(relu/linear)
- **Otras activaciones** (solo en última capa): Softmax, PReLU, ELU, LeakyReLU, ThresholdedReLU
- **Train-only** (se filtran del right model): Dropout, SpatialDropout1D/2D/3D, GaussianDropout, GaussianNoise, AlphaDropout, ActivityRegularization
- **Otras**: Flatten, Identity

### PyTorch (`PytorchLayerTypes`)
- **Parámetros (path)**: Linear (=Dense), Conv1d, Conv2d, Conv3d
- **MaxPooling**: MaxPool1d, MaxPool2d, MaxPool3d
- **Activación soportada**: ReLU, Identity
- **Otras activaciones**: ReLU6, LeakyReLU, ELU, PReLU, Hardshrink, Hardsigmoid, Tanh, Hardtanh, Hardswish, Sigmoid, LogSigmoid, Threshold, Softmin, Softmax, Softmax2d, LogSoftmax
- **Train-only**: Dropout, Dropout1d/2d/3d, AlphaDropout, FeatureAlphaDropout

**Restricciones**:
- Sin convoluciones dilatadas (dilation > 1).
- En PyTorch: solo `padding_mode='zeros'`.
- Las activaciones intermedias deben ser ReLU (o linear).

---

## 6. Módulo `paths/` — Generación de caminos

### paths.py
- `sort_paths(paths)`: ordena caminos por lexicographic order.
- `non_zero_input_select_paths(flat_inputs_with_bias, all_paths)`: selecciona caminos cuya neurona de entrada es no-nula (implementado en Cython).
- `retrieve_coefficients(select_masks, paths_input_neurons, flat_inputs_with_bias)`: extrae coeficientes del sistema lineal (Cython).
- `partial_to_full_idxs(partial, full)`: índices de transformación para expandir selección parcial a espacio combinado.

### combine.py
- `combine_paths(list_paths)`: combina listas de caminos de capas sucesivas mediante `combine_paths_left_right`.
- `combine_paths_left_right(paths_left, paths_right)`: join interno de paths donde `paths_left[-1] == paths_right[0]`. Paths con `paths_right[0] == 0` son "bias paths" y siempre se incluyen.
- `compute_combine_paths_output_size(list_paths)`: calcula el tamaño esperado sin materializar.

### dense.py
- `get_all_paths_dense_layer(weights, biases)`: todos los caminos entre dos capas densas (Cython).

### conv.py
- `get_all_paths_conv_layer(input_shape, weights, biases, strides, padding, groups, data_format)`: caminos para conv 1D/2D/3D. Soporta `channels_last` y `channels_first`.

### maxpooling.py
- `get_all_paths_maxpool_layer(input_shape, pool_size, strides, padding, data_format)`: caminos para maxpool (selecciona el neurón ganador).

---

## 7. Módulo `linalg.py` — Álgebra lineal

- `r_transform(AB)`: transforma `AB` a la matriz R de la factorización QR (`np.linalg.qr` con `mode="r"`).
- `back_substitution(A, b, absolute_cutoff, relative_cutoff)`: back-substitution con cutoffs opcionales (Cython).
- `_make_square_system(A, b)`: convierte sistema `(M, N)` en cuadrado `(N, N)` eliminando ecuaciones incompatibles (Cython).
- `solve(AB, include_last_row, solver, **kwargs)`: resuelve el sistema lineal. Soporta `"lstsq"` y `"back_substitution"`.
- `_qr_update(AB_old, AB_new)`: actualiza incrementalmente una factorización QR.

---

## 8. Mixins (`mixins.py`)

### LogMixin
Permite filtrar logs por nivel de verbosidad.
- `_log(*message, log_level=0)`: imprime si `log_level < verbose`.

### NumpyPersistanceMixin
Gestiona serialización de objetos con grandes arrays numpy.
- `save(directory_path)`: guarda como directorio `.qsi` con `py_objects.pkl` + `numpy_variables.npz`.
- `load(directory_path)`: carga desde directorio.
- `__getstate__` / `__setstate__`: personaliza pickle para guardar arrays numpy con `np.savez_compressed`.

**Importante**: Los arrays numpy listados en `_NUMPY_VARIABLES` se guardan en `numpy_variables.npz` (no en pickle). Las listas de arrays se codifican como `__iter_<idx>_<name>`.

---

## 9. Flujo de uso típico (API)

### Linear System (Keras)
```python
from qsimov.keras_path_selector import KerasPathSelector
from qsimov.keras_qsimov_linear_system import KerasQsimovLinearSystem

# 1. Pre-entrenar modelo Keras
model = kr.Sequential([...])
model.fit(x_train_1, y_train_1, ...)

# 2. Crear PathSelector sobre el modelo entrenado
path_selector = KerasPathSelector(
    neural_network=model,
    initial_layer=-1,   # solo última capa
    verbose=1
)

# 3. Crear y entrenar QsimovLinearSystem
qls = KerasQsimovLinearSystem(
    path_selector=path_selector,
    solver="back_substitution",
    absolute_cutoff=1e-2,
    relative_cutoff=1e6,
    qr_shrinkage_factor=10,
)
qls.fit(x_train, y_train, batch_size=256)

# 4. Predecir
y_pred = qls.predict(x_test)

# 5. Guardar/cargar
qls.save("my_model.qsi")
qls = KerasQsimovLinearSystem.load("my_model.qsi")
```

### Gradient (PyTorch)
```python
from qsimov.pytorch_path_selector import PytorchPathSelector
from qsimov.pytorch_qsimov_gradient import PytorchQsimovGradient

# 1. Pre-entrenar modelo PyTorch
model = nn.Sequential([...])
# ... training loop ...

# 2. Crear QsimovGradient
qg = PytorchQsimovGradient(
    PytorchPathSelector(
        neural_network=model,
        input_shape=(1, 28, 28),   # REQUERIDO en PyTorch
        initial_layer=-4,
    )
)

# 3. Entrenar
history = qg.fit(
    x_train, y_train,
    epochs=5,
    optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
    loss_function=nn.MSELoss(),
)

# 4. Predecir
y_pred = qg.predict(x_test)
```

---

## 10. Diferencias clave entre frameworks

| Aspecto | Keras | PyTorch |
|---------|-------|---------|
| `PathSelector` constructor | No requiere `input_shape` | Requiere `input_shape` explícito |
| Left model device | GPU si disponible | GPU si disponible |
| Right model device | Siempre CPU | Siempre CPU |
| Formato de pesos Dense | `(in, out)` | `(out, in)` → se transpone |
| data_format conv | `channels_last` | `channels_first` |
| Iterador de batches | `as_tensorflow_dataset()` | `as_pytorch_dataloader()` |
| Modelo interno (gradiente) | `CustomConnectedDense` (Keras Dense) | `CustomConnectedLinear` (nn.Linear) |
| Guardado del modelo | `.h5` files | `.pt` files |
| Activaciones detectadas | `layer.activation.__name__` | `layer.__class__.__name__` |

---

## 11. Persistencia (.qsi directories)

Un directorio `.qsi` contiene:
```
my_model.qsi/
├── py_objects.pkl          ← pickle del objeto (sin numpy ni modelos NN)
├── numpy_variables.npz     ← arrays numpy comprimidos
├── left_model.h5/.pt       ← left model (si existe)
├── right_model.h5/.pt      ← right model
└── path_selector.qsi/      ← PathSelector anidado (en QsimovGradient/LinearSystem)
    ├── py_objects.pkl
    ├── numpy_variables.npz
    ├── left_model.h5/.pt
    └── right_model.h5/.pt
```

---

## 12. Cython extensions

Los módulos `.pyx` se compilan con `python setup.py build_ext --inplace`. Son la parte de alto rendimiento:

| Módulo Cython | Función |
|---------------|---------|
| `c_linalg.pyx` | `c_make_square_system`, `c_back_substitution` |
| `paths/c_paths.pyx` | `c_non_zero_input_select_paths`, `c_retrieve_coefficients` |
| `paths/c_combine.pyx` | `c_combine_paths_left_right_sort_join` |
| `paths/c_dense.pyx` | `c_get_all_paths_dense_layer` |
| `paths/c_conv.pyx` | `c_get_all_paths_conv1d/2d/3d_layer` |
| `paths/c_maxpooling.pyx` | caminos de maxpool |

---

## 13. Experimentos

Los experimentos están en `experiments/` y usan MLflow (`experiments/mlflow.py`) y git (`experiments/git.py`) para tracking.

| Experimento | Dataset | Frameworks | Qué demuestra |
|-------------|---------|------------|---------------|
| `mnist_speed_loss` | MNIST | Keras + PyTorch | Velocidad y pérdida: Qsimov vs re-entrenamiento estándar |
| `mnist_learning_rate` | MNIST | Keras + PyTorch | Sensibilidad al learning rate del algoritmo de gradiente |
| `cifar10_gradient_by_splits` | CIFAR-10 | Keras + PyTorch | Impacto del split layer en precisión (gradiente) |
| `imagenet_subset_by_splits` | ImageNet (100 labels) | Keras | Impacto del tamaño del dataset (splits de datos) |
| `imagenet_continual_learning` | ImageNet (100 labels) | Keras | **Nuevo**: No-forgetting, multi-round retraining |

### Notas sobre `imagenet_subset_by_splits`
- Usa los 4 directorios de train → **100 labels** (1300 muestras/label = 130,000 totales).
- El `preprocess_data.py` genera: `data/imagenet_subset/{x_train,y_train,x_test,y_test}.npy`.
- La pérdida es `sparse_categorical_crossentropy` (labels enteros, no one-hot).
- Solo Keras (PyTorch no implementado).

### Experimento `imagenet_continual_learning` (nuevo)

**Objetivo**: demostrar las propiedades fundamentales de Qsimov en el dataset más grande.

**Dataset**: mismos 100 labels de imagenet_subset, divididos en 4 rondas secuenciales de 325 muestras/label cada una.

**Métodos comparados**:
| Método | Descripción |
|--------|-------------|
| `qsimov_linear_accum` | QsimovLinearSystem **sin** `reset_equations()` entre rondas → ecuaciones acumulativas → **no hay olvido** |
| `qsimov_linear_reset` | QsimovLinearSystem **con** `reset_equations()` antes de cada ronda → baseline de olvido |
| `qsimov_gradient` | QsimovGradient fine-tuned por ronda sin replay |
| `standard_finetune` | Fine-tuning estándar por ronda (olvido catastrófico) |
| `standard_cumulative` | Re-entrenamiento con TODOS los datos acumulados (cota superior) |

**Métricas**:
- `overall_accuracy.html`: accuracy en test set tras cada ronda.
- `forgetting_curves.html`: accuracy por ronda en sus val sets propios (muestra olvido).
- `training_time.html`: tiempo acumulado por método.

**Ficheros**:
```
experiments/imagenet_continual_learning/
├── __init__.py
├── main.py                    ← orquestador MLflow
├── preprocess_data.py         ← load + make_round_splits (N_ROUNDS=4)
├── keras_model_factory.py     ← re-exporta modelos de imagenet_subset
├── train_keras.py             ← entrena path_selector + standard en ronda 1
├── train_keras_qsimov.py      ← 3 variantes Qsimov (accum/reset/gradient)
├── train_keras_standard.py    ← fine-tuning secuencial + cumulative oracle
└── plot_continual_learning.py ← genera los 3 plots HTML
```

**Mecanismo clave del no-forgetting** (sistema lineal):
`QsimovLinearSystem.fit()` NO resetea `equations_` entre llamadas. Llamar `fit(ronda_1)` luego `fit(ronda_2)` acumula ecuaciones de ambas rondas antes de resolver. Esto produce soluciones que satisfacen TODOS los datos vistos sin necesidad de replay.

Cada experimento tiene:
- `main.py`: orquestador principal.
- `train_keras_models.py` / `train_pytorch_models.py`: entrenamiento clásico.
- `train_keras_qsimov_models.py` / `train_pytorch_qsimov_models.py`: entrenamiento Qsimov.
- `plot_*.py`: generación de gráficas comparativas.
- `preprocess_data.py`: preprocesado del dataset.

---

## 14. Tests

Tests ubicados en `tests/`, ejecutar con `pytest tests/`. Estructura:

- `tests/paths/`: pruebas unitarias de generación de caminos (dense, conv1d/2d/3d, maxpool1d/2d/3d, combine).
- `tests/path_selector/`: pruebas del PathSelector con redes simples ([2,2], [3,2,2,1], conv2d, maxpool2d). Contiene comparativa `test_keras_vs_pytorch.py`.
- `tests/qsimov_gradient/`: pruebas del gradient algorithm (red [3,6,5,2]).
- `tests/qsimov_linear_system/`: pruebas del linear system algorithm (red [3,6,5,2]).
- `tests/test_linalg.py`: pruebas del módulo linalg.
- `nn_mocks_keras.py` / `nn_mocks_pytorch.py`: fábricas de redes de prueba.

---

## 15. Dependencias principales

```
numpy==1.21.0
tensorflow==2.11.0      (incluye Keras)
torch==2.0.1
Cython==3.0.0a11
pytest==7.2.0
```

Python recomendado: **3.8.10**.

---

## 16. Notas importantes para trabajar en el proyecto

1. **Compilar Cython antes de usar**: `python setup.py build_ext --inplace` — sin esto el código falla con ImportError.

2. **El PathSelector siempre en CPU (right model)**: Por diseño, el right model siempre se ejecuta en CPU para las operaciones de path selection.

3. **`initial_layer` puede ser negativo**: Se interpreta desde el final del modelo. Ejemplo: `-1` = solo la última capa.

4. **Las activaciones intermedias deben ser ReLU (o linear)**: Cualquier otra activación antes de la última capa de pesos lanza `ValueError`.

5. **PyTorch requiere `input_shape`**: A diferencia de Keras, PyTorch no infiere el input shape del modelo automáticamente.

6. **El directorio `.qsi` es el formato de persistencia**: Todos los `save`/`load` operan sobre directorios `.qsi`, no archivos simples.

7. **QR shrinkage**: El parámetro `qr_shrinkage_factor` controla cuándo se aplica QR para comprimir el sistema lineal. Valores más grandes = menos compressions = más memoria pero más rápido.

8. **`output_masks_`**: Máscara boolean de shape `(n_outputs, n_paths)` que indica qué caminos llegan a cada output. Fundamental para filtrar coeficientes por output.

9. **Camino 0 = bias**: El índice 0 en los caminos es siempre el bias (neurona artificial con valor 1 añadida al principio del vector de entrada).
