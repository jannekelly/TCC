
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Carregar modelo treinado
model = keras.models.load_model("modelo_frutas.h5")

# Defina manualmente os nomes das classes (ajuste conforme o seu modelo)
class_names = ['apple', 'banana', 'orange']

# Função para processar a imagem e fazer a previsão
def carregar_e_prever(path_imagem):
    img = keras.preprocessing.image.load_img(path_imagem, target_size=(150, 150))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Cria batch de 1 imagem

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_index = np.argmax(score)
    confidence = 100 * np.max(score)

    return class_names[class_index], confidence, img

# Função chamada ao clicar no botão
def escolher_imagem():
    file_path = filedialog.askopenfilename()
    if file_path:
        classe, confianca, imagem_pil = carregar_e_prever(file_path)

        # Exibir imagem na interface
        imagem_pil = imagem_pil.resize((150, 150))
        imagem_tk = ImageTk.PhotoImage(imagem_pil)
        label_imagem.configure(image=imagem_tk)
        label_imagem.image = imagem_tk

        # Exibir resultado
        resultado.set(f"Fruta: {classe} | Confiança: {confianca:.2f}%")

# Interface principal
janela = tk.Tk()
janela.title("Identificador de Frutas com IA")
janela.geometry("400x400")

btn = tk.Button(janela, text="Escolher Imagem", command=escolher_imagem, font=("Arial", 12))
btn.pack(pady=10)

label_imagem = tk.Label(janela)
label_imagem.pack()

resultado = tk.StringVar()
resultado.set("Resultado aparecerá aqui.")
label_resultado = tk.Label(janela, textvariable=resultado, font=("Arial", 12))
label_resultado.pack(pady=10)

janela.mainloop()
