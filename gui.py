import tkinter as tk
import tkinter.font as font
from tkinter import ttk
import time

from piece_classifier import Classifier
import cv2
import numpy as np
from logger import Logger
from PIL import Image, ImageTk

IMAGE_SIZE = (720, 480)
OPTION_SIZE = (240, 240)


class Application(tk.Frame):
    def __init__(self, master=None, src=0) -> None:
        super().__init__(master)
        self.logReport = Logger("GUI")
        self.logReport.logger.info("initializing GUI")
        self.src = src
        self.master = master
        self.width = 1080
        self.height = 720
        self.zetas = 0
        self.tensores = 0
        self.anillos = 0
        self.arandelas = 0
        self.rotas = 0
        self.master.geometry("%dx%d" % (self.width, self.height))

        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(expand=True, fill="both")

        self.tab_configuracion = tk.Frame(
            self.notebook, width=self.width, height=self.height
        )
        self.tab_configuracion.pack(fill="both", expand=True)

        self.tab_funcionamiento = tk.Frame(
            self.notebook, width=self.width, height=self.height
        )
        self.tab_funcionamiento.pack(fill="both", expand=True)

        self.tab_informes = tk.Frame(
            self.notebook, width=self.width, height=self.height
        )
        self.tab_informes.pack(fill="both", expand=True)

        self.notebook.add(self.tab_configuracion, text="Configuración")
        self.notebook.add(self.tab_funcionamiento, text="Funcionamiento")
        self.notebook.add(self.tab_informes, text="Informes")

        self.createWidgets()
        self.createFeedFrame()
        self.createPieceFrame()
        self.createStateFrame()
        self.createTableFrame()
        self.createTimeFrame()

        names, images = self.getOptions()
        self.createOptionFrames(images, names)
        self.master.mainloop()

    def createWidgets(self):
        self.fontLabelText = font.Font(family="Helvetica", size=8)
        # Parent changed from self.master to self.tab_main
        self.btInitCamera = tk.Button(
            self.tab_funcionamiento,
            text="Iniciar Camara",
            bg="#007a39",
            fg="#ffffff",
            width=12,
            command=self.initCamera,
        )
        self.btInitCamera.place(x=10, y=90 + IMAGE_SIZE[1] * 2)

        # Parent changed from self.master to self.tab_main
        self.btStopCamera = tk.Button(
            self.tab_funcionamiento,
            text="Parar camara",
            bg="#7a0039",
            fg="#ffffff",
            width=12,
            command=self.stopCamera,
        )
        self.btStopCamera.place(x=10 + IMAGE_SIZE[0], y=90 + IMAGE_SIZE[1] * 2)

    def getOptions(self):
        images = ["zeta.jpeg", "tensor.jpeg", "anillo.jpeg", "arandela.jpeg"]
        names = ["zetas", "tensors", "anillos", "arandelas"]
        return names, images

    def createOptionFrames(self, images, names):
        if not hasattr(self, "optionLabel_by_name"):
            self.optionLabel = {}
        if not hasattr(self, "optionCheck_by_name"):
            self.optionCheck = {}
        if not hasattr(self, "optionVar_by_name"):
            self.optionVar = {}

        for i, name in enumerate(names):
            imageTk = self.readImage(images[i])
            self.optionLabel[name] = tk.Label(
                self.tab_configuracion, borderwidth=2, relief="solid"
            )
            self.optionLabel[name].place(x=(10 + OPTION_SIZE[0]) * i, y=30)
            self.optionLabel[name].configure(image=imageTk)
            self.optionLabel[name].image = imageTk

            self.optionVar[name] = tk.BooleanVar(value=False)
            self.optionCheck[name] = (
                tk.Checkbutton(  # Actually, it should be a simple check
                    self.tab_configuracion,
                    text=name,
                    variable=self.optionVar[name],
                    onvalue=True,
                    offvalue=False,
                    anchor="w",
                    relief="groove",
                    padx=10,
                    pady=5,
                    font=font.Font(family="Helvetica", size=32),
                )
            )
            self.optionCheck[name].place(
                x=(10 + OPTION_SIZE[0]) * i, y=60 + OPTION_SIZE[1]
            )
            self.optionCheck[name].configure(width=10, height=5)

    def readImage(self, file):
        img = Image.open(file)
        resized_img = img.resize(OPTION_SIZE, Image.LANCZOS)

        imageTk = ImageTk.PhotoImage(resized_img)
        return imageTk

    def createFeedFrame(self):
        # Parent changed from self.master to self.tab_main
        self.labelVideo_1 = tk.Label(
            self.tab_funcionamiento, borderwidth=2, relief="solid"
        )
        self.labelVideo_1.place(x=10, y=30)
        imageTk = self.createImageZeros()
        self.labelVideo_1.configure(image=imageTk)
        self.labelVideo_1.image = imageTk

    def createPieceFrame(self):
        # Parent changed from self.master to self.tab_main
        self.labelVideo_2 = tk.Label(
            self.tab_funcionamiento, borderwidth=2, relief="solid"
        )
        self.labelVideo_2.place(x=IMAGE_SIZE[0] + 20, y=30)
        imageTk = self.createImageZeros()
        self.labelVideo_2.configure(image=imageTk)
        self.labelVideo_2.image = imageTk

    def createStateFrame(self):
        # Parent changed from self.master to self.tab_main
        self.frameState = tk.Frame(
            self.tab_funcionamiento,
            width=IMAGE_SIZE[0],
            height=IMAGE_SIZE[1],
            background="light yellow",
        )
        self.frameState.place(x=IMAGE_SIZE[0] + 20, y=IMAGE_SIZE[1] + 60)

        # Parent remains self.frameState (correct)
        self.labelStateText = tk.Label(
            self.frameState,
            text="Esperando Pieza",
            background="light yellow",
            foreground="orange",
            font=(
                "Helvetica",
                32,
                "bold",
            ),
            padx=10,
            pady=10,
        )
        self.labelStateText.place(relx=0.5, rely=0.5, anchor="center")

    def createTimeFrame(self):
        # Parent changed from self.master to self.tab_main
        self.frameTime = tk.Frame(
            self.tab_funcionamiento,
            width=IMAGE_SIZE[0],
            height=IMAGE_SIZE[1],
        )
        self.frameTime.place(x=10 + 20, y=IMAGE_SIZE[1] + 60)

        # Parent remains self.frameState (correct)
        self.labelTimeText = tk.Label(
            self.frameTime,
            text="Esperando Pieza",
            font=(
                "Helvetica",
                32,
                "bold",
            ),
            padx=10,
            pady=10,
        )

        self.labelTimeText.place(relx=0.5, rely=0.5, anchor="center")

    def createTableFrame(self):
        # Parent changed from self.master to self.tab_main
        self.frameState = tk.Frame(
            self.tab_informes,
            width=IMAGE_SIZE[0],
            height=IMAGE_SIZE[1],
            background="white",
        )
        self.frameState.place(x=10, y=30)
        style = ttk.Style(self.master)

        style.configure(
            "mystyle.Treeview", font=("Helvetica", 20, "bold"), rowheight=40
        )
        style.configure("mystyle.Treeview.Heading", font=("Helvetica", 24, "bold"))
        style.configure(
            "mystyle.Treeview", font=("Helvetica", 20, "bold")
        )  # Data rows font
        style.configure("mystyle.Treeview.Heading", font=("Helvetica", 24, "bold"))

        self.frameState.grid_propagate(False)

        columns = ("#1", "#2")
        self.tree = ttk.Treeview(
            self.frameState, columns=columns, show="headings", style="mystyle.Treeview"
        )

        self.tree.heading("#1", text="Piece type")
        self.tree.heading("#2", text="Count")

        frame_width = IMAGE_SIZE[0]
        self.tree.column("#1", width=frame_width // 2, anchor="w")
        self.tree.column("#2", width=frame_width // 2, anchor="center")

        self.tree.grid(row=0, column=0, sticky="nsew")

        self.frameState.grid_columnconfigure(0, weight=1)
        self.frameState.grid_rowconfigure(0, weight=1)

    def createImageZeros(self):
        frame = np.zeros([IMAGE_SIZE[1], IMAGE_SIZE[0], 3], dtype=np.uint8)
        imagetk = self.convertToFrameTk(frame)
        return imagetk

    def initCamera(self):
        self.camera_1 = Classifier(self.src)
        self.camera_1.start()
        self.showvideo()
        print("Iniciando cámara...")

    def stopCamera(self):
        print("Parando cámara...")
        self.camera_1.stop()

    def showvideo(self):
        try:
            ret, frame = self.camera_1.read()
            if frame is not None:
                imageTk = self.convertToFrameTk(frame=cv2.resize(frame, IMAGE_SIZE))
                self.labelVideo_1.configure(image=imageTk)
                self.labelVideo_1.image = imageTk
                if self.camera_1.separate_frame():
                    self.eval_frame()

            self.labelVideo_1.after(1, self.showvideo)
        except Exception as e:
            print(str(e))

    def eval_frame(self):
        start_time = time.time()
        piece = self.camera_1.read_piece()
        self.logReport.debug("Pieza en frame")
        imageTkPiece = self.convertToFrameTk(
            frame=pad_frame(piece, IMAGE_SIZE[0], IMAGE_SIZE[1])
        )
        self.labelVideo_2.configure(image=imageTkPiece)
        self.labelVideo_2.image = imageTkPiece

        piece_type, condition, features_dict = self.camera_1.classify_piece()
        if condition == "Defectuosa" or piece_type == "Unknown":
            self.updateStateLabel("Pieza rota", "dark red", "red")
            self.rotas += 1
        else:
            self.updateStateLabel(f"{piece_type}", "green", "light green")
            if piece_type == "Zeta":
                self.zetas += 1
            if piece_type == "Tensor":
                self.tensores += 1
            if piece_type == "Anillo":
                self.anillos += 1
            if piece_type == "Arandela":
                self.arandelas += 1
        new_data = [
            ("Zetas", self.zetas),
            ("Tensores", self.tensores),
            ("Anillos", self.anillos),
            ("Arandelas", self.arandelas),
            ("Piezas rotas", self.rotas),
        ]
        processing_time = time.time() - start_time
        self.updateTimeLabel(processing_time)
        self.updateTable(new_data)

    def updateTable(self, new_data):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for property_name, value in new_data:
            self.tree.insert("", tk.END, values=(property_name, value))

    def convertToFrameTk(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgArray = Image.fromarray(frame)
        imageTk = ImageTk.PhotoImage(image=imgArray)
        return imageTk

    def updateStateLabel(self, new_text, text_color, background_color="light yellow"):
        try:
            self.labelStateText.config(
                text=new_text, fg=text_color, bg=background_color
            )
            self.frameState.config(background=background_color)
        except AttributeError:
            print("Error: labelStateText is not yet defined.")

    def updateTimeLabel(self, processing_time):
        try:
            self.labelTimeText.config(
                text=f"Tiempo de procesamiento = \n{processing_time:.3f} Segundos"
            )
        except AttributeError:
            print("Error: labelStateText is not yet defined.")


def pad_frame(frame, target_width, target_height):
    H_in, W_in = frame.shape

    if W_in > target_width or H_in > target_height:
        print(
            "Warning: Input frame is larger than target. Cropping might be necessary instead of padding."
        )
        return frame

    pad_w_total = target_width - W_in
    pad_h_total = target_height - H_in

    pad_left = pad_w_total // 2
    pad_top = pad_h_total // 2

    pad_right = pad_w_total - pad_left
    pad_bottom = pad_h_total - pad_top

    padded_frame = cv2.copyMakeBorder(
        frame,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    return padded_frame


def main():
    root = tk.Tk()
    root.title("GUI CAMERA")
    appRunCamera = Application(
        master=root, src="./Vid_Piezas/Anillos/Anillos_Malos.mp4"
    )


if __name__ == "__main__":
    main()
