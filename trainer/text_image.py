import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class TextImage():
    """テキスト画像生成クラス
    NOTE:縦書き(vert=True)の場合は、斜体と下付き線は正常に動作しない
    NOTE:apt install libraqm-dev Pillow>=8.0が必要
    """

    def __init__(self, text, font_path, text_color=(0,0,0),
                 background_color=(255,255,255), font_size=20, margin_wh=(10,10),
                 pad_xy=(0,0), underlined=False, bold=False, italic=False, blur=False, noise=False, vert=False):
        self.text = text
        self.text_color = text_color
        self.background_color = background_color
        self.font_path = font_path
        self.font_size = font_size
        self.margin_wh = margin_wh
        self.pad_xy = pad_xy
        self.underlined = underlined
        self.bold = bold
        self.italic = italic
        self.blur = blur
        self.noise = noise
        self.vert = vert

    @staticmethod
    def pil2cv(img_pil):
        img_cv_bgr = np.array(img_pil, dtype = np.uint8)[:, :, ::-1]
        return img_cv_bgr

    @staticmethod
    def cv2pil(img_cv_bgr):
        img_cv_rgb = img_cv_bgr[:, :, ::-1]
        img_pil = Image.fromarray(img_cv_rgb)
        return img_pil

    @staticmethod
    def text_size(text, font_path, font_size, vert=False):

        direction = 'ttb' if vert else None

        # 文字サイズを取得するためだけに画像生成
        img_pil = Image.new('RGB', (2000, 2000), (255, 255, 255))
        draw = ImageDraw.Draw(img_pil)
        font_pil = ImageFont.truetype(font=font_path, size=font_size)
        _,_, w, h = draw.textbbox((0,0), text, font=font_pil, direction=direction)
        return w, h

    def trans_italic(self, img_cv, deg=-20):
        a = np.deg2rad(deg)
        mat = np.float32(
            [[1, np.tan(a), abs(int(self.text_h*np.tan(a)))],
            [0, 1, 0]])
        italic_img_cv = cv2.warpAffine(img_cv, mat, (self.text_w+abs(int(self.text_h*np.tan(a))), self.text_h), borderValue=self.background_color)

        return italic_img_cv

    def to_cv(self):

        text_w, text_h = self.text_size(self.text, self.font_path, self.font_size, self.vert)

        self.text_w = text_w
        self.text_h = text_h
        direction = 'ttb' if self.vert else None

        img_cv = np.full((self.text_h, self.text_w, 3), self.background_color, dtype=np.uint8)

        x, y = self.pad_xy
        img_pil = self.cv2pil(img_cv)
        draw = ImageDraw.Draw(img_pil)
        font_pil = ImageFont.truetype(font=self.font_path, size=self.font_size)
        draw.text(xy=(x, y), text=self.text, fill=self.text_color, font=font_pil, direction=direction)

        if self.underlined:
            draw.line(((x, self.text_h+1), (self.text_w, self.text_h+1)), fill=self.text_color, width=5)

        if self.bold:
            draw.text(xy = (x+1, y), text=self.text, fill=self.text_color, font=font_pil, direction=direction)
            draw.text(xy = (x+2, y), text=self.text, fill=self.text_color, font=font_pil, direction=direction)
        img_cv = self.pil2cv(img_pil)

        if self.italic:
            img_cv = self.trans_italic(img_cv)

        if self.blur:
            img_cv = cv2.GaussianBlur(img_cv, (5,5), 0)

        if self.margin_wh[0] == 0 and self.margin_wh[1] == 0:
            pass
        else:

            # 背景追加
            fg_h, fg_w, _ = img_cv.shape
            img_margin_cv = np.full((fg_h+2*self.margin_wh[1], fg_w+2*self.margin_wh[0], 3), self.background_color, dtype=np.uint8)
            img_margin_cv[self.margin_wh[1]:self.margin_wh[1]+fg_h, self.margin_wh[0]:self.margin_wh[0]+fg_w,:] = img_cv
            img_cv = img_margin_cv

        if self.noise:
            noise_level = 50
            bg_h, bg_w, _ = img_margin_cv.shape
            noise = np.random.randint(0, noise_level, (bg_h, bg_w))
            noise_3ch = np.zeros_like(img_margin_cv).astype(np.uint32)
            noise_3ch[:,:,0] = noise
            noise_3ch[:,:,1] = noise
            noise_3ch[:,:,2] = noise
            img_cv = img_cv + noise_3ch
            img_cv = np.clip(img_cv, 0, 255).astype(np.uint8)
            img_cv = cv2.GaussianBlur(img_cv, (3,3), 0)

        return img_cv