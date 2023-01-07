from pathlib import Path
import cv2
import numpy as np


def main():
    input_img_path = Path("image/input/lena.jpg")
    output_img_path = Path("image/output/lena_output.jpg")
    
    input_img = cv2.imread(str(input_img_path))

    ### ここにいろいろやる
    output_img = process_dithering(input_img)

    # cv2.imwrite(str(output_img_path), output_img)


def process_dithering(img):
    # 4x4配列ディザリングで参照するmatrix
    dither_matrix = np.array([
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5]
    ]) * (2**4)
    
    # 画像サイズの取得, 処理後の画像を宣言
    hight, width, ch = img.shape

    # 画像サイズのdither_matrixを取得(切り上げ演算)
    tiled_dither_matrix_grayscale = np.tile(dither_matrix, (-(-hight // 4), -(-width // 4)))[:hight, :width]

    tiled_dither_matrix = np.stack(
        [tiled_dither_matrix_grayscale, tiled_dither_matrix_grayscale, tiled_dither_matrix_grayscale],
        axis=2
    )

    return np.where(img < tiled_dither_matrix, 0, 255)


if __name__ == "__main__":
    main()
