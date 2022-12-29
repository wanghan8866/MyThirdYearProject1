import tkinter as tk
import customtkinter as ctk
from PIL import Image
import os


class ImageLoader:
    All_Images = {}
    Names = set()
    DARK = 0
    LIGHT = 1

    @staticmethod
    def getTkImage(name: str):
        return ImageLoader.All_Images[name]

    @staticmethod
    def loadAll():
        image_path = "resources/images/"
        addresses = ["Flappy_Bird_icon.png"
                     ]
        ImageLoader.Names = {
            "bird"
        }
        for address, name in zip(addresses, ImageLoader.Names):
            my_image = ctk.CTkImage(dark_image=Image.open(f"{image_path}{address}"), size=(50, 50))
            ImageLoader.All_Images[name] = my_image

        logo_image = ctk.CTkImage(Image.open(os.path.join(image_path, "CustomTkinter_logo_single.png")), size=(26, 26))
        large_test_image = ctk.CTkImage(Image.open(os.path.join(image_path, "large_test_image.png")), size=(500, 150))
        image_icon_image = ctk.CTkImage(Image.open(os.path.join(image_path, "image_icon_light.png")), size=(20, 20))
        home_image = ctk.CTkImage(light_image=Image.open(os.path.join(image_path, "home_dark.png")),
                                  dark_image=Image.open(os.path.join(image_path, "home_light.png")), size=(20, 20))
        chat_image = ctk.CTkImage(light_image=Image.open(os.path.join(image_path, "chat_dark.png")),
                                  dark_image=Image.open(os.path.join(image_path, "chat_light.png")), size=(20, 20))
        add_user_image = ctk.CTkImage(light_image=Image.open(os.path.join(image_path, "add_user_dark.png")),
                                      dark_image=Image.open(os.path.join(image_path, "add_user_light.png")),
                                      size=(20, 20))
        ImageLoader.All_Images["logo"] = logo_image
        ImageLoader.All_Images["test"] = large_test_image
        ImageLoader.All_Images["icon"] = image_icon_image
        ImageLoader.All_Images["home"] = home_image
        ImageLoader.All_Images["chat"] = chat_image
        ImageLoader.All_Images["user"] = add_user_image


if __name__ == '__main__':
    ImageLoader.loadAll()
    print(ImageLoader.All_Images["bird"])
