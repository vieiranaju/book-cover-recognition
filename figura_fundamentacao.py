import cv2
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent
PASTA_CAPAS = BASE / "capas"
PASTA_RESULTADOS = BASE / "resultados"
PASTA_RESULTADOS.mkdir(exist_ok=True)

# escolher uma imagem na pasta capas
CAMINHO = PASTA_CAPAS / "A Empregada.jpg"

imagem = cv2.imread(str(CAMINHO))

if imagem is None:
    raise FileNotFoundError(
        f"Não foi possível abrir {CAMINHO}"
    )

# converter a imagem para tons de cinza
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# detectar pontos de interesse
orb = cv2.ORB_create(nfeatures=1500)
pontos, descritores = orb.detectAndCompute(cinza, None)

# desenhar os pontos encontrados
com_pontos = cv2.drawKeypoints(
    imagem,
    pontos,
    None,
    color=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# criar uma versão rotacionada
altura, largura = imagem.shape[:2]
centro = (largura / 2, altura / 2)

matriz = cv2.getRotationMatrix2D(centro, 25, 1)
rotacionada = cv2.warpAffine(
    imagem,
    matriz,
    (largura, altura),
    borderValue=(255, 255, 255)
)

cinza_rot = cv2.cvtColor(
    rotacionada, cv2.COLOR_BGR2GRAY
)
pontos_rot, _ = orb.detectAndCompute(cinza_rot, None)

com_pontos_rot = cv2.drawKeypoints(
    rotacionada,
    pontos_rot,
    None,
    color=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# salvar as imagens
cv2.imwrite(
    str(PASTA_RESULTADOS / "01_original.png"), imagem
)
cv2.imwrite(
    str(PASTA_RESULTADOS / "02_pontos_orb.png"),
    com_pontos
)
cv2.imwrite(
    str(PASTA_RESULTADOS / "03_rotacionada_orb.png"),
    com_pontos_rot
)

print("Pontos na imagem original:", len(pontos))
print("Pontos na imagem rotacionada:", len(pontos_rot))
print("Figuras salvas na pasta resultados!")
