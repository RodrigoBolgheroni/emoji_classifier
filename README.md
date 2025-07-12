# Classificador de Emoções com Emojis

Projeto para classificar frases em emoções (alegria, tristeza, raiva, medo e amor) usando rede neural com TensorFlow/Keras e retornar um emoji correspondente.

---

## Funcionalidades

- Treinamento do modelo LSTM bidirecional para classificação de emoções em texto  
- Pré-processamento de texto com tokenização personalizada  
- Salva o melhor modelo durante o treinamento  
- Predição interativa via terminal com emoji para cada emoção  
- Armazena labels e tokenizer para reutilização  

---

## Estrutura do projeto

.
├── data/
│ └── dataset.csv # Dataset com frases e emoções
├── models/
│ ├── emocoes.keras # Modelo treinado completo
│ ├── emocoes_best.keras # Melhor modelo salvo durante treino
│ ├── tokenizer.pickle # Tokenizer salvo
│ └── emotion_labels.json # Labels das emoções
├── preprocess.py # Script/função para limpeza e tokenização
├── utils.py # Funções utilitárias (ex: get_emoji)
├── train.py # Script para treinar o modelo
├── predict.py # Script para predição interativa
├── requirements.txt # Dependências do projeto
└── README.md # Este arquivo
---

## Como usar

### 1. Instalação

Recomenda-se criar e ativar um ambiente virtual:

```bash
# Cria o ambiente virtual
python -m venv venv

# Ativa o ambiente (Linux/Mac)
source venv/bin/activate

# Ativa o ambiente (Windows)
# venv\Scripts\activate
```

Instale as dependências do projeto:
```bash
pip install -r requirements.txt
```

### 2. Treinar o modelo
Para treinar o modelo com o dataset fornecido, execute:
```bash
python train.py
```

### 3. Fazer predições
Após o treinamento, você pode fazer predições interativas:
```bash
python predict.py
```

#### Exemplo de uso

O script solicitará que você digite uma frase. Para encerrar, digite `sair`.
```console
Digite uma frase (ou 'sair' para terminar): Hoje eu me sinto muito feliz
Emoji: 😄

Digite uma frase (ou 'sair' para terminar): Estou com medo do futuro
Emoji: 😨

Digite uma frase (ou 'sair' para terminar): sair
```

Personalização
Para adicionar ou modificar emojis, altere o dicionário emoji_map em utils.py.

O arquivo dataset.csv pode ser atualizado com novas frases para melhorar o modelo.

Requisitos
Python 3.8+

TensorFlow 2.x

Pandas

NumPy

scikit-learn

Autor
Rodrigo Bolgheroni

Licença
MIT License