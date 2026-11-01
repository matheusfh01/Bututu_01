"""Sprite flutuante (sempre no topo), balão de fala e chat. Compatível com a pasta de sprites do Bututu v2:
idle_0.png, thinking_0.png, speaking_0.png, happy_0.png, angry_0.png, sad_0.png, scared_0.png (vários quadros = animação).
Sem sprites, desenha um corpo provisório para você já testar."""
import math
import platform
import queue
import threading
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk

SYS = platform.system()
BG, PANEL, CYAN, MAGENTA, YELLOW, KEY = "#0b0b14", "#12121f", "#00f0ff", "#ff2a6d", "#fcee0a", "#010203"
STATE_COLOR = {"idle": CYAN, "thinking": YELLOW, "speaking": MAGENTA}
EMO_COLOR = {"happy": YELLOW, "angry": "#ff3030", "sad": "#5a7bff", "scared": "#d8d8ff"}
SIZE = 180
FONT = ("Consolas", 10)


class Overlay:
    def __init__(self, cfg, sprite_dir, handlers):
        self.h = handlers  # dict: submit, look, watch, voice, quit, status
        self.sprite_dir = sprite_dir
        self.state, self.emotion, self._tick = "idle", None, 0
        self.q = queue.Queue()
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        canvas_bg = BG
        if SYS == "Windows":
            self.root.attributes("-transparentcolor", KEY)
            canvas_bg = KEY
        elif SYS == "Darwin":
            self.root.attributes("-transparent", True)
            self.root.config(bg="systemTransparent")
            canvas_bg = "systemTransparent"
        self.root.geometry(f"{SIZE}x{SIZE}+80+80")
        self.canvas = tk.Canvas(self.root, width=SIZE, height=SIZE, bg=canvas_bg, highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<Button-3>", self._menu)
        self.canvas.bind("<Button-2>", self._menu)  # macOS
        self.frames = {}
        self.reload_sprites()
        self.bubble = self.chat = None
        self._bubble_job = None
        self.root.after(100, self._poll)
        self.root.after(120, self._animate)

    # ----- thread-safety: qualquer thread chama post(); só a thread do Tk executa
    def post(self, fn):
        self.q.put(fn)

    def _poll(self):
        try:
            while True:
                self.q.get_nowait()()
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def ask(self, text):
        ev, res = threading.Event(), []
        def show():
            res.append(messagebox.askyesno("Bututu", text, parent=self.root))
            ev.set()
        self.post(show)
        ev.wait(60)
        return bool(res and res[0])

    # ----- sprites
    def reload_sprites(self):
        self.frames = {}
        for key in ("idle", "thinking", "speaking", "happy", "angry", "sad", "scared"):
            files = sorted(self.sprite_dir.glob(f"{key}_*.png")) if self.sprite_dir.exists() else []
            imgs = []
            for f in files[:60]:
                im = Image.open(f).convert("RGBA")
                im.thumbnail((SIZE - 20, SIZE - 20))
                imgs.append(ImageTk.PhotoImage(im))
            if imgs:
                self.frames[key] = imgs

    def set_state(self, state):
        self.state = state

    def set_emotion(self, emo, hold_ms=6000):
        self.emotion = emo
        self.root.after(hold_ms, lambda: setattr(self, "emotion", None) if self.emotion == emo else None)

    def _animate(self):
        self._tick += 1
        key = self.emotion if self.emotion in self.frames else self.state
        frames = self.frames.get(key) or self.frames.get("idle")
        self.canvas.delete("all")
        bob = math.sin(self._tick / 4) * (6 if self.state == "speaking" else 3)
        if frames:
            self.canvas.create_image(SIZE // 2, SIZE // 2 + bob, image=frames[(self._tick // 2) % len(frames)])
        else:  # corpo provisório
            color = EMO_COLOR.get(self.emotion) or STATE_COLOR[self.state]
            y = SIZE // 2 + bob
            self.canvas.create_oval(35, y - 50, SIZE - 35, y + 50, fill=PANEL, outline=color, width=3)
            for dx in (-18, 18):
                self.canvas.create_oval(SIZE // 2 + dx - 7, y - 15, SIZE // 2 + dx + 7, y - 1, fill=color, outline="")
            mouth = 8 if self.state == "speaking" and self._tick % 4 < 2 else 2
            self.canvas.create_rectangle(SIZE // 2 - 12, y + 15, SIZE // 2 + 12, y + 15 + mouth, fill=color, outline="")
        if self.h["status"]().get("watching"):
            self.canvas.create_oval(SIZE - 22, 8, SIZE - 10, 20, fill=MAGENTA, outline="")  # ponto: observação ligada
        self.root.after(120, self._animate)

    # ----- mouse
    def _press(self, e):
        self._dx, self._dy, self._moved = e.x_root - self.root.winfo_x(), e.y_root - self.root.winfo_y(), False

    def _drag(self, e):
        self._moved = True
        self.root.geometry(f"+{e.x_root - self._dx}+{e.y_root - self._dy}")

    def _release(self, _e):
        if not self._moved:
            self.toggle_chat()

    def _menu(self, e):
        st = self.h["status"]()
        m = tk.Menu(self.root, tearoff=0, bg=PANEL, fg=CYAN, font=FONT)
        m.add_command(label="👁 Olhar a tela agora", command=self.h["look"])
        m.add_command(label=("✔ " if st.get("watching") else "") + "Observação automática", command=self.h["watch"])
        m.add_command(label=("✔ " if st.get("voice") else "") + "Voz (ler respostas)", command=self.h["voice"])
        m.add_separator()
        m.add_command(label="Sair", command=self.h["quit"])
        m.tk_popup(e.x_root, e.y_root)

    # ----- balão e chat
    def say(self, text, who="bot", append=False):
        """Mostra no balão e no histórico do chat (se estiver aberto). append=True continua a mesma resposta."""
        self._log(text, who)
        if who != "bot":
            return
        self._btext = (getattr(self, "_btext", "") + " " + text).strip() if append else text
        text = self._btext
        if self.bubble is None:
            self.bubble = tk.Toplevel(self.root)
            self.bubble.overrideredirect(True)
            self.bubble.attributes("-topmost", True)
            self._blabel = tk.Label(self.bubble, bg=PANEL, fg="white", font=FONT, wraplength=280, justify="left",
                                    padx=10, pady=6, highlightbackground=CYAN, highlightthickness=1)
            self._blabel.pack()
        self._blabel.config(text=text)
        self.bubble.deiconify()
        self.bubble.update_idletasks()
        x = self.root.winfo_x() + SIZE // 2 - self.bubble.winfo_width() // 2
        y = self.root.winfo_y() - self.bubble.winfo_height() - 6
        self.bubble.geometry(f"+{max(0, x)}+{max(0, y)}")
        if self._bubble_job:
            self.root.after_cancel(self._bubble_job)
        self._bubble_job = self.root.after(min(20000, 3500 + 60 * len(text)), self.bubble.withdraw)

    def toggle_chat(self):
        if self.chat is not None and self.chat.winfo_exists():
            self.chat.withdraw() if self.chat.state() == "normal" else self.chat.deiconify()
            return
        self.chat = tk.Toplevel(self.root)
        self.chat.title("Bututu")
        self.chat.configure(bg=BG)
        self.chat.attributes("-topmost", True)
        self.log = tk.Text(self.chat, width=48, height=16, bg=PANEL, fg="white", font=FONT, wrap="word",
                           state="disabled", relief="flat")
        self.log.pack(padx=8, pady=(8, 4), fill="both", expand=True)
        self.entry = tk.Entry(self.chat, bg=PANEL, fg=CYAN, insertbackground=CYAN, font=FONT, relief="flat")
        self.entry.pack(padx=8, pady=(0, 8), fill="x")
        self.entry.bind("<Return>", self._send)
        self.entry.focus_set()

    def _send(self, _e):
        text = self.entry.get().strip()
        self.entry.delete(0, "end")
        if text:
            self._log(text, "user")
            self.h["submit"](text)

    def _log(self, text, who):
        if self.chat is None or not self.chat.winfo_exists():
            return
        self.log.configure(state="normal")
        self.log.insert("end", ("você: " if who == "user" else "") + text + "\n")
        self.log.configure(state="disabled")
        self.log.see("end")

    def run(self):
        self.root.mainloop()
