import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

# -----------------------------
# FUNCIONES AUXILIARES
# -----------------------------
def generar_malla(X, paso=0.05):
    minimo_x, maximo_x = X[:, 0].min(), X[:, 0].max()
    minimo_y, maximo_y = X[:, 1].min(), X[:, 1].max()
    rejilla_x, rejilla_y = np.meshgrid(np.arange(minimo_x, maximo_x, paso),
                                       np.arange(minimo_y, maximo_y, paso))
    puntos = np.c_[rejilla_x.ravel(), rejilla_y.ravel()]
    return rejilla_x, rejilla_y, puntos

def visualizar_region(X, Y, red, titulo):
    rejilla_x, rejilla_y, puntos = generar_malla(X)
    print(f"Predicción de {len(puntos)} puntos para graficar la frontera de decisión...")
    pred = red.predict(puntos, verbose=1)
    Z = np.argmax(pred, axis=1).reshape(rejilla_x.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(rejilla_x, rejilla_y, Z, cmap=plt.cm.tab10, alpha=0.5)
    ax.contour(rejilla_x, rejilla_y, Z, colors='k', linewidths=0.5)

    etiquetas = np.argmax(Y, axis=1)
    colores = plt.cm.tab10.colors
    for clase in range(np.max(etiquetas) + 1):
        puntos = np.where(etiquetas == clase)
        ax.scatter(X[puntos, 0], X[puntos, 1], c=[colores[clase]], edgecolor='k', s=20)

    ax.set_xlim(rejilla_x.min(), rejilla_x.max())
    ax.set_ylim(rejilla_y.min(), rejilla_y.max())
    ax.set_xlabel("Característica 1")
    ax.set_ylabel("Característica 2")
    ax.set_title(titulo)
    ax.set_aspect('auto')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig("frontera_disfrazada.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

# -----------------------------
# CARGA Y PROCESAMIENTO DE DATOS
# -----------------------------
entradas = pd.read_csv('features_400_2features_8classes.csv', header=None, sep=r'\s+')
salidas = pd.read_csv('labels_400_2features_8classes.csv', header=None)

cantidad = min(len(entradas), len(salidas))
x_data = entradas.values[:cantidad].astype(float)
y_clase = salidas.values[:cantidad].flatten()

traductor = LabelEncoder()
y_codificada = traductor.fit_transform(y_clase)
y_final = to_categorical(y_codificada)

x_ent, x_val, y_ent, y_val = train_test_split(x_data, y_final, test_size=0.2, random_state=42, stratify=y_codificada)

# -----------------------------
# CONSTRUCCION DEL MODELO
# -----------------------------
entrada = Input(shape=(2,))
cap1 = Dense(32, activation='tanh')(entrada)
cap2 = Dense(32, activation='tanh')(cap1)
salida = Dense(y_final.shape[1], activation='softmax')(cap2)

red = Model(inputs=entrada, outputs=salida)
optimizador = SGD(learning_rate=0.005)
red.compile(optimizer=optimizador, loss='categorical_crossentropy', metrics=['accuracy'])

# -----------------------------
# ENTRENAMIENTO PERSONALIZADO
# -----------------------------
epocas = 300
bloque = 32
bitacora = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

print("Entrenamiento en curso:\n")
for epoca in range(epocas):
    resultado = red.fit(x_ent, y_ent, validation_data=(x_val, y_val), epochs=1, batch_size=bloque, verbose=0)

    for clave in bitacora:
        bitacora[clave].append(resultado.history[clave][0])

    print(f"Iteración {epoca+1:03}/{epocas} → Pérdida: {bitacora['loss'][-1]:.4f}, Prec: {bitacora['accuracy'][-1]:.4f}, "+
          f"Val_Pérdida: {bitacora['val_loss'][-1]:.4f}, Val_Prec: {bitacora['val_accuracy'][-1]:.4f}")

# -----------------------------
# GRAFICOS DE RENDIMIENTO
# -----------------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(bitacora['loss'], label='Entrenamiento')
plt.plot(bitacora['val_loss'], label='Validación')
plt.title("Pérdida")
plt.xlabel("Iteración")
plt.ylabel("Pérdida")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(bitacora['accuracy'], label='Entrenamiento')
plt.plot(bitacora['val_accuracy'], label='Validación')
plt.title("Precisión")
plt.xlabel("Iteración")
plt.ylabel("Precisión")
plt.legend()

plt.tight_layout()
plt.savefig("grafico_rendimiento.png", dpi=300)
plt.show()

# -----------------------------
# RESULTADOS Y MATRIZ DE CONFUSION
# -----------------------------
final_train_acc = bitacora['accuracy'][-1] * 100
final_val_acc = bitacora['val_accuracy'][-1] * 100

print("\nResumen:")
print(f"Exactitud entrenamiento: {final_train_acc:.2f}%")
print(f"Exactitud prueba:        {final_val_acc:.2f}%")

if final_val_acc >= 80:
    print("✅ Rendimiento aceptable (≥ 80%)")
else:
    print("⚠️  Rendimiento insuficiente (< 80%)")

predicciones = red.predict(x_val, verbose=0)
pred_clases = np.argmax(predicciones, axis=1)
verdaderas = np.argmax(y_val, axis=1)

matriz = confusion_matrix(verdaderas, pred_clases)
mostrar = ConfusionMatrixDisplay(confusion_matrix=matriz)
mostrar.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Matriz de Confusión")
plt.savefig("confusion_irreconocible.png", dpi=300)
plt.show()

# ----------------------------
# FRONTERA DE DECISIÓN
# ----------------------------


visualizar_region(x_ent, y_ent, red, "Frontera de decisión")

