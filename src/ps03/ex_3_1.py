'''
Exercise 3.1. Consider the Yale face dataset.
Apply a dimension reduction with PCA. Ver-
ify that it is possible to recognise individuals
on the principal plane.
'''

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.decomposition import PCA
import os
os.makedirs("results", exist_ok=True)
filename = os.path.splitext(os.path.basename(__file__))[0]

img_path = "src/ps03/yalefaces/subject01.glasses"
pixels = plt.imread(img_path)


fig, ax = plt.subplots(1,2)

ax[0].imshow(pixels)
ax[0].set_title("Original")
ax[0].axis('off')

pca = PCA(n_components=1)
pca.fit(pixels)
pixel_reconstructed = pca.inverse_transform(pca.transform(pixels))
im = ax[1].imshow(pixel_reconstructed)
ax[1].set_title("PCA (1 component)")
ax[1].axis('off')

plt.savefig(f"results/{filename}_pca.pdf")

ax_slider = fig.add_axes([0.2, 0.15, 0.7, 0.03])
component_slider = Slider(
    ax=ax_slider,
    label='Components',
    valmin=1,
    valmax=int(min(pixels.shape[:2])/5),
    valinit=1,
    valstep=1
)

def update(val):
    pca = PCA(n_components=val)
    pca.fit(pixels)
    pixel_reconstructed = pca.inverse_transform(pca.transform(pixels))
    im.set_data(pixel_reconstructed)
    ax[1].set_title(f"PCA ({int(val)} components)")
    fig.canvas.draw_idle()

component_slider.on_changed(update)

plt.show()