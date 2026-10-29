
import cv2
import numpy as np

# =========================================================
# CONTADOR DE PARAFUSOS - PYTHON 3.12 + OPENCV
# Fundo preto texturizado e parafusos dourados
# =========================================================

CAMERA = 0

# Redimensionamento para facilitar o processamento.
ESCALA = 0.8

# Limites iniciais para filtrar áreas muito pequenas.
AREA_MINIMA = 250
AREA_MAXIMA = 50000

# Faixa inicial da cor dourada no espaço HSV.
# O OpenCV usa H entre 0 e 179.
H_MIN = 5
H_MAX = 40
S_MIN = 45
V_MIN = 35

# Separação de objetos pelo Watershed.
# Valores menores podem separar mais; valores maiores,
# menos. O resultado depende da imagem e da iluminação.
FATOR_DISTANCIA = 0.35


def identificar_parafusos(frame):
    """
    Recebe uma imagem e devolve:
    - imagem anotada;
    - máscara dos pixels dourados;
    - quantidade estimada de peças.
    """

    # Redimensiona a imagem.
    frame = cv2.resize(
        frame,
        None,
        fx=ESCALA,
        fy=ESCALA
    )

    original = frame.copy()

    # -----------------------------------------------------
    # 1. SUAVIZA A IMAGEM
    # -----------------------------------------------------

    suavizada = cv2.GaussianBlur(
        frame,
        (5, 5),
        0
    )

    # -----------------------------------------------------
    # 2. CONVERTE PARA HSV
    # -----------------------------------------------------

    hsv = cv2.cvtColor(
        suavizada,
        cv2.COLOR_BGR2HSV
    )

    # -----------------------------------------------------
    # 3. SELECIONA A COR DOURADA
    # -----------------------------------------------------

    limite_inferior = np.array(
        [H_MIN, S_MIN, V_MIN],
        dtype=np.uint8
    )

    limite_superior = np.array(
        [H_MAX, 255, 255],
        dtype=np.uint8
    )

    mascara = cv2.inRange(
        hsv,
        limite_inferior,
        limite_superior
    )

    # -----------------------------------------------------
    # 4. REDUZ RUÍDOS E UNE PARTES DA MESMA PEÇA
    # -----------------------------------------------------

    kernel_abertura = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    kernel_fechamento = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mascara = cv2.morphologyEx(
        mascara,
        cv2.MORPH_OPEN,
        kernel_abertura
    )

    mascara = cv2.morphologyEx(
        mascara,
        cv2.MORPH_CLOSE,
        kernel_fechamento,
        iterations=2
    )

    # -----------------------------------------------------
    # 5. PREPARA A SEPARAÇÃO PELO WATERSHED
    # -----------------------------------------------------

    kernel = np.ones((3, 3), np.uint8)

    fundo_seguro = cv2.dilate(
        mascara,
        kernel,
        iterations=3
    )

    distancia = cv2.distanceTransform(
        mascara,
        cv2.DIST_L2,
        5
    )

    maximo = distancia.max()

    # Evita divisão ou limiar inválido quando não há objetos.
    if maximo <= 0:
        return original, mascara, 0

    _, frente_segura = cv2.threshold(
        distancia,
        FATOR_DISTANCIA * maximo,
        255,
        cv2.THRESH_BINARY
    )

    frente_segura = frente_segura.astype(np.uint8)

    desconhecido = cv2.subtract(
        fundo_seguro,
        frente_segura
    )

    # Cada região inicial recebe um identificador.
    quantidade, marcadores = cv2.connectedComponents(
        frente_segura
    )

    marcadores = marcadores + 1

    marcadores[desconhecido == 255] = 0

    # O Watershed usa os pixels da imagem para delimitar
    # as regiões entre objetos.
    cv2.watershed(
        original,
        marcadores
    )

    # -----------------------------------------------------
    # 6. MEDE CADA REGIÃO ENCONTRADA
    # -----------------------------------------------------

    anotada = original.copy()
    contador = 0

    for identificador in np.unique(marcadores):

        # 1 representa o fundo conhecido.
        # -1 representa as linhas divisórias.
        if identificador <= 1:
            continue

        regiao = np.zeros(
            mascara.shape,
            dtype=np.uint8
        )

        regiao[marcadores == identificador] = 255

        # Limita a região aos pixels dourados detectados.
        regiao = cv2.bitwise_and(
            regiao,
            mascara
        )

        area = cv2.countNonZero(regiao)

        if area < AREA_MINIMA or area > AREA_MAXIMA:
            continue

        contornos, _ = cv2.findContours(
            regiao,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contornos:
            continue

        contorno = max(
            contornos,
            key=cv2.contourArea
        )

        x, y, largura, altura = cv2.boundingRect(
            contorno
        )

        contador += 1

        cv2.rectangle(
            anotada,
            (x, y),
            (x + largura, y + altura),
            (0, 255, 0),
            2
        )

        cv2.putText(
            anotada,
            str(contador),
            (x, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # -----------------------------------------------------
    # 7. MOSTRA A CONTAGEM
    # -----------------------------------------------------

    cv2.rectangle(
        anotada,
        (10, 10),
        (330, 65),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        anotada,
        f"PARAFUSOS: {contador}",
        (20, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 255),
        2
    )

    return anotada, mascara, contador


# =========================================================
# INICIA A CAMERA
# =========================================================

camera = cv2.VideoCapture(CAMERA)

if not camera.isOpened():
    print("Erro: nao foi possivel abrir a camera.")
    raise SystemExit(1)

# Solicita uma resolucao inicial.
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Contador de parafusos iniciado.")
print("Q = sair")
print("Janela Mascara = pixels reconhecidos como dourados")

try:
    while True:

        sucesso, frame = camera.read()

        if not sucesso:
            print("Erro ao capturar imagem.")
            break

        anotada, mascara, contador = identificar_parafusos(
            frame
        )

        cv2.imshow(
            "Contador de Parafusos",
            anotada
        )

        cv2.imshow(
            "Mascara - Dourado Detectado",
            mascara
        )

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()

print("Programa finalizado.")
