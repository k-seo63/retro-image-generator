from pathlib import Path
import cv2
import numpy as np

# parameta
img_name = "lena"
resize_ratio = 1
saturation_ratio = 1.5


def main():
    input_img_path = Path(f"image/input/{img_name}.jpg")
    output_img_path = Path(f"image/output/{img_name}.png")
    
    input_img = cv2.imread(str(input_img_path))
    resized_img = resize_img(input_img, resize_ratio)

    corrected_img = saturation_up(gamma_correction(tonecurve(resized_img)), saturation_ratio)
    dithered_64color_img = process_dithering_64color(corrected_img)

    # cv2.imwrite(str(output_img_path.with_name(f"{img_name}_dithered_64.png").as_posix()), dithered_64color_img)
    cv2.imwrite(str(output_img_path), binarize_img(dithered_64color_img))


def process_dithering_8color(img):
    """ディザリングで R:G:B=1bit:1bit:1bit の8色へ減色
    """
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
    # 3ch化
    tiled_dither_matrix = np.stack(
        [tiled_dither_matrix_grayscale, tiled_dither_matrix_grayscale, tiled_dither_matrix_grayscale],
        axis=2
    )

    return np.where((img < tiled_dither_matrix), 0, 255)



def process_dithering_64color(img):
    """ディザリングで R:G:B=2bit:2bit:2bit の64色へ減色
    """
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
    # 3ch化
    tiled_dither_matrix = np.stack(
        [tiled_dither_matrix_grayscale, tiled_dither_matrix_grayscale, tiled_dither_matrix_grayscale],
        axis=2
    )

    # 入力画像とMatrixの差を取り、それを0-255の値域におさめる
    diff_img_matrix = (img - tiled_dither_matrix + 255) / 2

    # うまいこと4値化する
    processed_img = np.where((0 <= diff_img_matrix) & (diff_img_matrix < 255/4), 0, diff_img_matrix)
    processed_img = np.where((255/4 <= diff_img_matrix) & (diff_img_matrix < 255/2), 255/3, processed_img)
    processed_img = np.where((255/2 <= diff_img_matrix) & (diff_img_matrix < 255/4*3), 255/3*2, processed_img)
    processed_img = np.where((255/4*3 <= diff_img_matrix), 255, processed_img)
    
    return processed_img


# TODO: 実装
def process_dithering_256color(img):
    """ディザリングで R:G:B=3bit:3bit:2bit の64色へ減色
    """
    return 0


def binarize_img(img):
    """2値化
    """
    _, processed_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    return processed_img


def resize_img(img, ratio):
    """リサイズ
    """
    processed_img = cv2.resize(img, None, fx=ratio, fy=ratio)
    return processed_img


def gamma_correction(img, gamma=2.2):
    """暗いところを持ち上げるように補正
    """
    nomalized_img = img.astype(np.float32) / 255
    gammaed_img = nomalized_img ** (1/gamma)
    return (gammaed_img * 255).astype(np.uint8)


def tonecurve(img):
    """HSV色空間に変換して、明度(v)に対してヒストグラム平坦化
        NOTE: 本当は逆S字トーンカーブをかけたかったけど、ちょうどいい関数がなかった
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
    return cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)


def saturation_up(img, ratio=1.5):
    """HSV色空間に変換して、彩度(s)をあげる
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 1] = img_hsv[:, :, 1] * ratio
    return cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)


if __name__ == "__main__":
    main()
