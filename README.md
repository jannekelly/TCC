# 🍎 Classificação Inteligente de Alimentos com Deep Learning

Este repositório contém o código-fonte e os experimentos do meu Trabalho de Conclusão de Curso (TCC) em Ciência da Computação pela Universidade Federal do Ceará (UFC). O projeto consiste no desenvolvimento de um sistema inteligente capaz de classificar de forma automatizada diferentes tipos de frutas e legumes a partir de imagens digitais.

## 🧠 Sobre o Projeto

O objetivo principal deste trabalho foi aplicar técnicas de **Inteligência Artificial**, especificamente **Visão Computacional** e **Deep Learning**, para analisar o padrão visual de alimentos. O modelo foi treinado para extrair características morfológicas e cromáticas das imagens, gerando uma predição precisa que pode ser futuramente integrada com tabelas nutricionais para automação em setores de saúde e logística.

## 🛠️ Tecnologias e Frameworks Utilizados

- **Linguagem Principal:** Python 3.x
- **Ambiente de Experimentos:** Jupyter Notebook (`cnnTCC.ipynb`)
- **Frameworks de Deep Learning:** TensorFlow e Keras
- **Interface Gráfica:** Python (`interface_indetificador_frutas.py`)
- **Algoritmo Base:** Redes Neurais Convolucionais (CNN - Convolutional Neural Networks)

## 📂 Organização dos Arquivos

- `cnnTCC.ipynb`: Notebook contendo a pipeline de dados (carregamento do dataset, pré-processamento de imagens, aumento de dados/data augmentation), arquitetura de camadas da CNN, treinamento do modelo e gráficos de avaliação (acurácia e perda).
- `interface_indetificador_frutas.py`: Script em Python desenvolvido para criar uma interface amigável que permite carregar uma imagem local e visualizar a predição realizada pelo modelo treinado.

## 🔧 Como Executar os Experimentos

### Pré-requisitos
Certifique-se de ter o Python e o gerenciador de pacotes pip instalados. É altamente recomendável a instalação das bibliotecas principais:
```bash
pip install tensorflow keras jupyter notebook
```

### Executando o Treinamento
1. Abra o ambiente do Jupyter:
   ```bash
   jupyter notebook
   ```
2. Execute as células do arquivo `cnnTCC.ipynb` para visualizar a arquitetura da rede e o processo de treinamento.

### Testando a Interface
Para rodar a ferramenta interativa de identificação de frutas, execute o comando:
```bash
python interface_indetificador_frutas.py
```

---
🧬 **Trabalho de Conclusão de Curso** — Bacharelado em Ciência da Computação (UFC)  
Desenvolvido por **Janne Kelly Oliveira Pereira**
