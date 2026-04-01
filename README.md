# Pothole Detection
Este é um projeto para detectar buracos em estradas.

## Características e Decisões Técnicas

- **Arquitetura**: YOLO26n (Arquitetura leve e com uma boa precisão, ideal para dispositivos edge e detecção em tempo real)

- **Dataset**: [pothole-roboflow](https://public.roboflow.com/object-detection/pothole/1)

## Treinamento e Primeiros Resultados

* Em termos de **precisão** o modelo está com **0.758**, isso significa que a cada 100 buracos que o modelo
previu ser buraco, 75 eram realmente. Ele evita falsos positivos bem.

* Para detecção de buracos em estradas, uma das métricas mais importantes é o recall.
O nosso modelo está com o **recall** de **0.628**, o que significa que o modelo ignora 1 a cada 3 buracos, e isso
para segurança viária pode ser perigoso.

* Um **mAP50** de **0.71** indica que ele consegue localizar e classificar bem um buraco quando a precisão da caixa IoU
é 50%.

* Um **mAP50-95** de **0.447** é muito bom para detecção de buracos em estradas e asfaltos, isso significa que ele ajusta bem a caixa
delimitadora com muita precisão geométrica.

#### Primeiras impressões
* Dá pra usar em um mapeamento geral. Porém para um carro autônomo ainda é muito perigoso.

## Predição
Para realizar a predição em um conjunto de dados, certifique-se do conjunto de dados estar no padrão **YOLO 26**.

Para executar a predição em um conjunto de dados, execute:
```python prediction.py isdataset True --path-yaml caminho/arquivo.yaml```

Para executar a predição em apenas uma imagem, execute:
```python prediction.py --path-img caminho/imagem.jpg```

## Estrutura de Arquivos

```
pothole-detection/
├── README.md                                    (este arquivo)
├── notebooks/                                   (pasta de notebooks para experimentos/testes)
├   ├── viz-augmentation.ipynb                   (Responsável por mostrar como estão ficando as imagens aumentadas)
├   ├── data-visualization.ipynb                 (Visualiza e reduz a dimensão do dataset no FiftyOne)
├   ├── train-yolo26n.ipynb                      (Realiza experimentos de fine-tuning)
├   ├── pred-yolo26n.ipynb                       (Realiza experimentos e testes de predição do modelo ajustado)
├── utils/                                       (pasta de utilitários)
├   ├── albumentations_helper.py                 (funções para auxiliar na visualização de imagens aumentadas)
├── requirements.txt                             (dependências Python)
├── .gitignore                                   (responsável para gerenciar arquivos que não podem ser monitorados)
├── prediction.py                                (script para facilitar a predição automática de um dataset ou imagem individual)
```

## Próximos passos

* Ainda há espaço para melhoria do modelo!
* Aumentar o dataset com outros tipos de composição de augmentation.
* Aumentar o dataset com mais imagens reais.
* Aumentar o dataset com mais imagens criadas artificialmente, com modelos generativos.