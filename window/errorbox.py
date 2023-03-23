from tkinter import messagebox


class ErrorBox:
    def __init__(self, title: str, error_msg: str):
        messagebox.showerror(title, error_msg)
