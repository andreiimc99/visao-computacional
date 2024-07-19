import cv2
import numpy as np

# Caminhos dos arquivos
ARQUIVO_CFG = "deteccao-objetos/yolov3{}.cfg".format("-tiny" if TINY else "")
ARQUIVO_PESOS = "deteccao-objetos/yolov3{}.weights".format("-tiny" if TINY else "")
ARQUIVO_CLASSES = "deteccao-objetos/coco{}.names".format("-tiny" if TINY else "")

def carregar_modelo(ARQUIVO_MODELO, ARQUIVO_CFG):
    '''
    Carrega o modelo de deep learning do YOLO para detecção de objetos.
    '''
    try:
        modelo = cv2.dnn.readNetFromDarknet(ARQUIVO_CFG, ARQUIVO_MODELO)
    except cv2.error as erro:
        print(f"Erro ao carregar o modelo: {erro}")
        exit()
    return modelo

def obter_classes(ARQUIVO_CLASSES):
    '''
    Carrega os nomes das classes do COCO dataset.
    '''
    with open(ARQUIVO_CLASSES, 'rt') as f:
        return f.read().strip().split('\n')

def main():
    '''
    Função principal que executa o sistema de detecção de objetos.
    '''
    captura = cv2.VideoCapture(ARQUIVO_VIDEO)
    modelo = carregar_modelo(ARQUIVO_MODELO, ARQUIVO_CFG)
    classes = obter_classes(ARQUIVO_CLASSES)

    while True:
        ret, frame = captura.read()
        if not ret:
            break

        # Criação do blob a partir do frame e realização da detecção
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        modelo.setInput(blob)
        saidas = modelo.forward(modelo.getUnconnectedOutLayersNames())

        caixas = []
        confiancas = []
        class_ids = []

        # Extração das caixas delimitadoras e confianças das detecções
        for saida in saidas:
            for deteccao in saida:
                scores = deteccao[5:]
                class_id = np.argmax(scores)
                confianca = scores[class_id]
                if confianca > 0.5:
                    caixa = deteccao[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (centroX, centroY, largura, altura) = caixa.astype("int")
                    inicioX = int(centroX - (largura / 2))
                    inicioY = int(centroY - (altura / 2))
                    caixas.append([inicioX, inicioY, int(largura), int(altura)])
                    confiancas.append(float(confianca))
                    class_ids.append(class_id)

        # Aplicação da supressão não máxima para finalizar as caixas delimitadoras
        indices = cv2.dnn.NMSBoxes(caixas, confiancas, 0.5, 0.4)

        if len(indices) > 0:
            for i in indices.flatten():
                (inicioX, inicioY) = (caixas[i][0], caixas[i][1])
                (largura, altura) = (caixas[i][2], caixas[i][3])
                cor = [int(c) for c in np.random.randint(0, 255, size=(3,))]
                cv2.rectangle(frame, (inicioX, inicioY), (inicioX + largura, inicioY + altura), cor, 2)
                texto = f"{classes[class_ids[i]]}: {confiancas[i]:.2f}"
                cv2.putText(frame, texto, (inicioX, inicioY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

        # Exibição do frame com as detecções
        cv2.imshow('Detecção de Objetos', frame)

        # Verificação de tecla para saída
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    captura.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
