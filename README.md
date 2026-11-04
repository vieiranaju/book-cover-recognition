# Book Cover Recognition

Sistema de visão computacional em Python e OpenCV para identificar capas de livros em fotos usando ORB, mesmo com variações de rotação e distância.

## Como executar

**1. Clone o repositório:**

```bash
git clone https://github.com/vieiranaju/book-cover-recognition.git
cd book-cover-recognition
```

**2. Instale as dependências:**

```bash
python -m pip install opencv-python==4.10.0.84 numpy
```

**3. Adicione as imagens das capas:**

Coloque as imagens na pasta `capas`, utilizando os formatos `.jpg`, `.jpeg` ou `.png`.

**4. Execute o programa:**

```bash
python main.py
```

A webcam será aberta para identificar as capas cadastradas. Pressione `q` para encerrar.

## Visualização dos pontos ORB

Para gerar as figuras dos pontos de interesse, altere o caminho da imagem no arquivo `figura_fundamentacao.py` e execute:

```bash
python figura_fundamentacao.py
```

As imagens geradas serão salvas na pasta `resultados`.