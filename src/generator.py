from pathlib import Path
import cv2
import numpy as np

# parameta
img_name = "009"
resize_ratio = 1
saturation_ratio = 1.5


def main():
    input_img_path = Path(f"image/input/{img_name}.jpg")
    output_img_path = Path(f"image/output/{img_name}.png")
    
    # 画像読み込み
    input_img = cv2.imread(str(input_img_path))

    # もろもろの処理
    resized_img = resize_img(input_img, resize_ratio)
    corrected_img = saturation_up(gamma_curve(resized_img), saturation_ratio)
    dithered_img = process_dithering(corrected_img)
    binarized_img = binarize_img(dithered_img)

    # 画像書き込み
    cv2.imwrite(str(output_img_path), binarized_img)


def process_dithering_8color(img):
    """
    ディザリングで R:G:B=1bit:1bit:1bit の8色へ減色
    """
    # 4x4配列ディザリングで参照するmatrix
    dither_matrix = np.array([
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5]
    ]) * (2**4)
    
    # 画像サイズの取得
    hight, width, _ = img.shape

    # 画像サイズのdither_matrixを取得(切り上げ演算)
    tiled_dither_matrix_grayscale = np.tile(
        dither_matrix, 
        (-(-hight // 4), -(-width // 4))
    )[:hight, :width]

    tiled_dither_matrix = np.stack(
        [tiled_dither_matrix_grayscale, tiled_dither_matrix_grayscale, tiled_dither_matrix_grayscale],
        axis=2
    )

    return np.where((img < tiled_dither_matrix), 0, 255).astype(np.uint8)



def process_dithering_64color(img):
    """
    ディザリングで R:G:B=2bit:2bit:2bit の64色へ減色
    """
    # 4x4配列ディザリングで参照するmatrix
    dither_matrix = np.array([
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5]
    ]) * (2**4)
    
    # 画像サイズの取得
    hight, width, _ = img.shape

    # 画像サイズのdither_matrixを取得(切り上げ演算)
    tiled_dither_matrix_grayscale = np.tile(
        dither_matrix, 
        (-(-hight // 4), -(-width // 4))
    )[:hight, :width]

    tiled_dither_matrix = np.stack(
        [tiled_dither_matrix_grayscale, tiled_dither_matrix_grayscale, tiled_dither_matrix_grayscale],
        axis=2
    )

    # 入力画像とMatrixの差を取り、それを0-255の値域におさめる
    diff_img_matrix = (img - tiled_dither_matrix + 255) / 2

    # うまいこと4値化する
    processed_img = np.where((0 <= diff_img_matrix) & (diff_img_matrix < 255/4), 0, diff_img_matrix)
    processed_img = np.where((255/4 <= diff_img_matrix) & (diff_img_matrix < 255/2), 255//3, processed_img)
    processed_img = np.where((255/2 <= diff_img_matrix) & (diff_img_matrix < 255/4*3), 255//3*2, processed_img)
    processed_img = np.where((255/4*3 <= diff_img_matrix), 255, processed_img)
    
    return processed_img.astype(np.uint8)


def process_dithering_256color(img):
    """
    ディザリングで R:G:B=3bit:3bit:2bit の256色へ減色
    """
    # TODO: 実装
    return 


def process_dithering(img, level=8):
    """
    ディザリングで R:G:B=level:level:level[bit] の level^3[色] へ減色
    """
    # 4x4配列ディザリングで参照するmatrix
    dither_matrix = np.array([
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5]
    ]) * (1/4**2) * 32 + ((256-32)/2) 
    
    # 画像サイズの取得
    hight, width, _ = img.shape

    # 画像サイズのdither_matrixを取得(切り上げ演算)
    tiled_dither_matrix_grayscale = np.tile(
        dither_matrix, 
        (-(-hight // 4), -(-width // 4))
    )[:hight, :width]

    tiled_dither_matrix = np.stack(
        [tiled_dither_matrix_grayscale, tiled_dither_matrix_grayscale, tiled_dither_matrix_grayscale],
        axis=2
    )

    # 入力画像とMatrixの差を取り、それを0-255の値域におさめる
    diff_img_matrix = (img - tiled_dither_matrix + 255) / 2

    # うまいこと<level>値化する
    for l in range(level):
        processed_img = np.where(
            (255*l/level <= diff_img_matrix) & (diff_img_matrix < 255*(l+1)/level), 
            255*l // (level-1), 
            diff_img_matrix
        )

    return processed_img.astype(np.uint8)


def binarize_img(img):
    """
    変わるかわからんが大津の手法を利用して2値化
    NOTE: OpenCVの大津の手法はグレースケールしか対応してない
    """
    # _, processed_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    processed_img = np.zeros(img.shape)
    b, g, r = cv2.split(img)
    
    _, processed_img[:, :, 0] = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, processed_img[:, :, 1] = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, processed_img[:, :, 2] = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return processed_img


def resize_img(img, ratio):
    """
    リサイズ
    """
    processed_img = cv2.resize(img, dsize=None, fx=ratio, fy=ratio)
    return processed_img


def gamma_curve(img, gamma=2.2):
    """
    暗いところを持ち上げるように補正
    """
    nomalized_img = img.astype(np.float32) / 255
    gammaed_img = nomalized_img ** (1/gamma)
    return (gammaed_img * 255).astype(np.uint8)


def tone_curve(img):
    """
    HSV色空間に変換して、明度(v)に対してヒストグラム平坦化
    NOTE: 本当は逆S字トーンカーブをやりたかったけど、ちょうどいい関数を簡単に実装できなかった
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
    return cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)


def saturation_up(img, ratio=1.5):
    """
    HSV色空間に変換して、彩度(s)をあげる
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 1] = img_hsv[:, :, 1] * ratio
    return cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)


if __name__ == "__main__":
    main()
