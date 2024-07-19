# Detecção de Objetos com YOLOv3 e OpenCV em Python

Nome: Andrei Moraes Cardoso

Este repositório contém um exemplo de uso do modelo YOLOv3 para detecção de objetos em tempo real utilizando OpenCV e Python. O código inclui o carregamento do modelo pré-treinado YOLOv3, a configuração do OpenCV DNN, e a realização da detecção de objetos em tempo real a partir de uma captura de vídeo (por exemplo, webcam).

![Exemplo de Detecção de Objetos](exemplo_detecao_objetos.png)

## Links para Download de Modelos YOLO

### YOLOv3:

- Arquivo de configuração: [yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
- Arquivo de pesos: [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)
- Arquivo de nomes das classes: [coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

### YOLOv3-tiny:

- Arquivo de configuração: [yolov3-tiny.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg)
- Arquivo de pesos: [yolov3-tiny.weights](https://pjreddie.com/media/files/yolov3-tiny.weights)
- Arquivo de nomes das classes: [coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

## Como Executar

### Instalação das Dependências

Para executar o projeto, é necessário instalar as bibliotecas OpenCV e NumPy. Use o seguinte comando:

```bash
pip install opencv-python numpy