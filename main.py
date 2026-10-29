from pathlib import Path

import cv2
import numpy as np


def carregar_capas(orb):
    capas = []
    pasta = Path(__file__).parent / "capas"
    for caminho in sorted(pasta.glob("*")):
        if caminho.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        imagem = cv2.imdecode(np.fromfile(caminho, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if imagem is None:
            continue
        _, descritores = orb.detectAndCompute(imagem, None)
        if descritores is not None:
            capas.append((caminho.stem, descritores))
    if not capas:
        raise ValueError("Coloque fotos das capas na pasta capas.")
    return capas


def identificar_capa(imagem, capas, orb, comparador):
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    _, descritores = orb.detectAndCompute(cinza, None)
    if descritores is None or len(descritores) < 2:
        return "Capa nao identificada"

    melhor_nome = "Capa nao identificada"
    maior_quantidade = 0
    for nome, referencia in capas:
        pares = comparador.knnMatch(referencia, descritores, k=2)
        quantidade = 0
        for par in pares:
            # A primeira correspondência precisa ser melhor que a segunda.
            if len(par) == 2 and par[0].distance < 0.75 * par[1].distance:
                quantidade += 1
        if quantidade > maior_quantidade:
            maior_quantidade = quantidade
            melhor_nome = nome

    # Ajustar este mínimo depois dos testes com os livros.
    if maior_quantidade < 15:
        return "Capa nao identificada"
    return f"Capa: {melhor_nome}"


def main():
    orb = cv2.ORB_create(nfeatures=1500)
    comparador = cv2.BFMatcher(cv2.NORM_HAMMING)
    capas = carregar_capas(orb)
    camera = cv2.VideoCapture(0)
    try:
        if not camera.isOpened():
            raise ValueError("Não foi possível abrir a câmera.")
        print("Mostre a capa para a câmera. Pressione Q para sair.")
        while True:
            sucesso, imagem = camera.read()
            if not sucesso:
                raise ValueError("Não foi possível receber a imagem da câmera.")
            resposta = identificar_capa(imagem, capas, orb, comparador)
            cv2.putText(imagem, resposta, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
            cv2.imshow("Identificacao de capas - Q para sair", imagem)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, cv2.error) as erro:
        print(f"Erro: {erro}")
