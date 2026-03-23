import argparse

import ultralytics
ultralytics.checks()
from ultralytics import YOLO

from ultralytics import settings
settings.update({"mlflow": True})

parser = argparse.ArgumentParser(description="Realiza a predição de um conjunto de dados ou imagem separada")
parser.add_argument("-d", "--isdataset", action="store_true", help="Ative se for um conjunto de dados")
parser.add_argument("-m", "--path-model", type=str, help="Path do modelo")
parser.add_argument("-y", "--path-yaml", type=str, help="Path do arquivo yaml")
parser.add_argument("-i", "--path-img", type=str, help="Path da imagem")
args = parser.parse_args()


def predict_many(path_model, path_yaml):
    model = YOLO(path_model)
    metrics = model.val(data=path_yaml, split='test')

    return metrics

def predict(path_model, path_img, confidence=0.5):
    model = YOLO(path_model)
    results = model.predict(source=path_img, save=True, conf=confidence)

    for result in results:
        boxes = result.boxes
        print('classes', boxes.cls)
        print('confidences', boxes.conf)

def main():
    path_model = args.path_model if args.path_model else "notebooks/runs/detect/yolo26_projeto_pothole/experimento_15/weights/best.pt"
    path_yaml = args.path_yaml if args.path_yaml else "data/Pothole.v1-raw.yolo26/data.yaml"

    if args.isdataset:
        results = predict_many(path_model, path_yaml)
        print(results)
    else:
        if args.path_img:
            path_img = args.path_img
            predict(path_model, path_img)
        else:
            raise Exception("É necessário passar uma imagem. Use -i para passar o caminho.")

if __name__ == "__main__":
    main()
    
