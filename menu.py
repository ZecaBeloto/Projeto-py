import tkinter as tk
from tkinter import ttk, messagebox

import eficiencia
import analise_risco_mult
import backtestmark  # novo script importado


# =========================
# FUNÇÕES DE ABERTURA
# =========================
def abrir_analise_risco():
    try:
        analise_risco_mult.main()
    except Exception as e:
        messagebox.showerror(
            "Erro",
            f"Erro ao abrir Análise de Risco\n\n{e}"
        )


def abrir_eficiencia():
    try:
        eficiencia.main()
    except Exception as e:
        messagebox.showerror(
            "Erro",
            f"Erro ao abrir Eficiência de Carteira\n\n{e}"
        )

def abrir_backtest():
    try:
        backtestmark.main()  # abre o script backtestmark.py
    except Exception as e:
        messagebox.showerror(
            "Erro",
            f"Erro ao abrir Backtest Mark\n\n{e}"
        )


# =========================
# INTERFACE MENU
# =========================
root = tk.Tk()
root.title("Menu de Análises – Zeca(AI)")
root.geometry("420x300")
root.resizable(False, False)

frame = ttk.Frame(root, padding=20)
frame.pack(expand=True, fill="both")

titulo = ttk.Label(
    frame,
    text="Selecione o tipo de análise",
    font=("Segoe UI", 14, "bold")
)
titulo.pack(pady=10)

btn1 = ttk.Button(
    frame,
    text="📊 Análise de Risco Multi-carteira",
    command=abrir_analise_risco,
    width=40
)
btn1.pack(pady=8)

btn2 = ttk.Button(
    frame,
    text="📈 Eficiência de Carteira (VaR)",
    command=abrir_eficiencia,
    width=40
)
btn2.pack(pady=8)

btn3 = ttk.Button(
    frame,
    text="💹 Alocação de Carteira Markowitz",
    command=abrir_backtest,
    width=40
)
btn3.pack(pady=8)

ttk.Separator(frame).pack(fill="x", pady=15)

rodape = ttk.Label(
    frame,
    text="Zeca(AI) – Ferramentas de Análise Financeira",
    font=("Segoe UI", 9),
    foreground="gray"
)
rodape.pack()

root.mainloop()
