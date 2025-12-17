import threading
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams.update({'figure.max_open_warning': 0})

RODAPE = (
    "Produzido por Helio Beloto Junior, Assessor de Investimento atuante pelo BTG. "
    "Cel:(14)99717-5510"
)

# =========================
# CONSTANTES DE IMPRESSÃO
# =========================
A4_LANDSCAPE = (11.69, 8.27)  # polegadas

# =====================================================
# DADOS
# =====================================================
def baixar_dados(ticker, data_ref, n):
    start = data_ref - timedelta(days=n * 4)
    end = data_ref + timedelta(days=1)

    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return None

    df = df[['Close']].dropna()
    df = df[df.index <= pd.to_datetime(data_ref)]
    return df.tail(n)

# =====================================================
# ANÁLISE
# =====================================================
def analisar(ticker_raw, data_str, n):
    ticker = ticker_raw.upper().strip()
    if not ticker.endswith(".SA"):
        ticker += ".SA"

    data_ref = datetime.strptime(data_str, "%d/%m/%Y")

    acao = baixar_dados(ticker, data_ref, n)
    ibov = baixar_dados("^BVSP", data_ref, n)

    if acao is None or acao.empty:
        raise ValueError("Sem dados suficientes da ação.")

    df = acao.rename(columns={'Close': 'acao'})
    if ibov is not None and not ibov.empty:
        df = df.join(ibov.rename(columns={'Close': 'ibov'}), how='inner')

    df['ret_acao'] = df['acao'].pct_change() * 100
    df['ret_ibov'] = df['ibov'].pct_change() * 100 if 'ibov' in df else np.nan
    df.dropna(inplace=True)

    media = df['ret_acao'].mean()
    vol = df['ret_acao'].std()
    sharpe = media / vol if vol != 0 else np.nan

    var_param = media - 1.65 * vol
    var_hist = np.percentile(df['ret_acao'], 5)

    correlacao = (
        df[['ret_acao', 'ret_ibov']].corr().iloc[0, 1]
        if 'ret_ibov' in df else np.nan
    )

    df['var_acao'] = (df['acao'] / df['acao'].iloc[0] - 1) * 100
    if 'ibov' in df:
        df['var_ibov'] = (df['ibov'] / df['ibov'].iloc[0] - 1) * 100

    return df, {
        "ticker": ticker,
        "sharpe": sharpe,
        "var_param": var_param,
        "var_hist": var_hist,
        "correlacao": correlacao
    }

# =====================================================
# INTERFACE
# =====================================================
class App:
    def __init__(self, root):
        self.root = root
        root.title("Análise de Risco – Relatório A4")

        self.df = None
        self.info = None
        self.fig1 = None
        self.fig2 = None

        self._build()

    def _build(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Ticker").grid(row=0, column=0)
        self.ticker = ttk.Entry(top)
        self.ticker.insert(0, "PETR4")
        self.ticker.grid(row=0, column=1)

        ttk.Label(top, text="Data").grid(row=1, column=0)
        self.data = ttk.Entry(top)
        self.data.insert(0, datetime.today().strftime("%d/%m/%Y"))
        self.data.grid(row=1, column=1)

        ttk.Label(top, text="Pregões").grid(row=2, column=0)
        self.n = ttk.Entry(top)
        self.n.insert(0, "252")
        self.n.grid(row=2, column=1)

        ttk.Button(top, text="Gerar Análise", command=self.run).grid(row=3, column=0, pady=6)
        ttk.Button(top, text="Exportar PDF A4", command=self.exportar_pdf).grid(row=3, column=1)

        self.preview = ttk.Frame(self.root)
        self.preview.pack(fill="both", expand=True)

    def run(self):
        threading.Thread(target=self._thread, daemon=True).start()

    def _thread(self):
        try:
            self.df, self.info = analisar(
                self.ticker.get(),
                self.data.get(),
                int(self.n.get())
            )
            self.root.after(0, self.render)
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def render(self):
        for w in self.preview.winfo_children():
            w.destroy()

        # ========= FIGURA 1 – VARIAÇÃO ACUMULADA =========
        self.fig1, ax1 = plt.subplots(figsize=A4_LANDSCAPE)
        ax1.plot(self.df['var_acao'], label=self.info['ticker'])

        if 'var_ibov' in self.df:
            ax1.plot(self.df['var_ibov'], '--', label='Ibovespa')

        ax1.legend()
        ax1.grid(alpha=0.3)

        texto = (
            f"Correlação: {self.info['correlacao']:.2f}\n"
            f"Sharpe: {self.info['sharpe']:.2f}\n"
            f"VaR Param: {self.info['var_param']:.2f}%\n"
            f"VaR Hist: {self.info['var_hist']:.2f}%"
        )

        ax1.text(0.02, 0.95, texto, transform=ax1.transAxes,
                 fontsize=12, va="top", color="darkred")

        ax1.text(0.5, 0.03, RODAPE, transform=ax1.transAxes,
                 fontsize=9, color="gray", ha="center")

        # ========= FIGURA 2 – RETORNOS DIÁRIOS =========
        self.fig2, ax2 = plt.subplots(figsize=A4_LANDSCAPE)
        cores = ['green' if x >= 0 else 'red' for x in self.df['ret_acao']]
        ax2.bar(self.df.index, self.df['ret_acao'], color=cores)

        ax2.axhline(self.info['var_param'], linestyle='--', label='VaR Param')
        ax2.axhline(self.info['var_hist'], linestyle=':', label='VaR Hist')

        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        ax2.text(0.5, 0.03, RODAPE, transform=ax2.transAxes,
                 fontsize=9, color="gray", ha="center")

        FigureCanvasTkAgg(self.fig1, self.preview).get_tk_widget().pack(
            side="top", fill="both", expand=True)

    def exportar_pdf(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")]
        )
        if not path:
            return

        with PdfPages(path) as pdf:
            pdf.savefig(self.fig1)
            pdf.savefig(self.fig2)

        messagebox.showinfo("Sucesso", "Relatório A4 exportado com sucesso.")

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
