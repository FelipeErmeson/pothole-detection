import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A


def plot_augmentation_grid(
    image: np.ndarray,
    transform: A.Compose = None,
    n_samples: int = 8,
    cols: int = 4,
    figsize: tuple = (16, 8),
    title: str = "Augmentations",
) -> None:
    """
    Plota uma grade com a imagem original e versões augmentadas.

    Args:
        image: Imagem lida com OpenCV (BGR ou RGB, uint8).
        transform: Pipeline do Albumentations. Se None, usa flip + rotate + brilho.
        n_samples: Número de imagens augmentadas a gerar.
        cols: Número de colunas na grade.
        figsize: Tamanho da figura matplotlib.
        title: Título geral da figura.
    """
    if transform is None:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=30, p=0.8),
            A.RandomBrightnessContrast(p=0.8),
        ])

    # Converte BGR -> RGB para exibição correta no matplotlib
    if image.ndim == 3 and image.shape[2] == 3:
        display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        display_image = image

    augmented = [transform(image=display_image)["image"] for _ in range(n_samples)]

    all_images = [display_image] + augmented
    all_titles = ["Original"] + [f"Aug {i + 1}" for i in range(n_samples)]

    rows = (len(all_images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).flatten()

    for ax, img, label in zip(axes, all_images, all_titles):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    for ax in axes[len(all_images):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
