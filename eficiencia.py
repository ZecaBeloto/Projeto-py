import threading
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import itertools
import math
import os

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams.update({'figure.max_open_warning': 0})

# =========================
# CONSTANTES
# =========================
RODAPE = (
    "Produzido por Helio Beloto Junior, Assessor de Investimento atuante pelo BTG. "
    "Cel:(14)99717-5510"
)

A4_LANDSCAPE = (11.69, 8.27)
Z_SCORE = 1.65  # 95%
PESO_MIN = 0.05  # 5%

COR_CABECALHO = "#1f4e79"   # azul escuro
COR_LINHA = "#ddebf7"      # azul claro
COR_TEXTO = "#000000"

# =========================
# DOWNLOAD DE DADOS
# =========================
def baixar_dados(ticker, data_ref, n):
    inicio = data_ref - timedelta(days=n * 4)
    fim = data_ref + timedelta(days=1)

    df = yf.download(ticker, start=inicio, end=fim, progress=False)
    if df.empty:
        return None

    df = df[['Close']].dropna()
    df = df[df.index <= pd.to_datetime(data_ref)]
    return df.tail(n)

# =========================
# ANÁLISE INDIVIDUAL
# =========================
def analisar(ticker_raw, data_str, n):
    ticker = ticker_raw.upper().strip()
    if not ticker.endswith(".SA"):
        ticker += ".SA"

    data_ref = datetime.strptime(data_str, "%d/%m/%Y")

    acao = baixar_dados(ticker, data_ref, n)
    ibov = baixar_dados("^BVSP", data_ref, n)

    if acao is None or acao.empty:
        raise ValueError(f"Sem dados suficientes para {ticker}")

    df = acao.rename(columns={'Close': 'acao'})
    df = df.join(ibov.rename(columns={'Close': 'ibov'}), how='inner')

    df['ret_acao'] = df['acao'].pct_change()
    df['ret_ibov'] = df['ibov'].pct_change()
    df.dropna(inplace=True)

    return df, ticker.replace(".SA", "")

# =========================
# GERADOR DE PESOS
# =========================
def gerar_pesos(n_ativos, passo):
    total = int(1 / passo)
    minimo = int(PESO_MIN / passo)
    restante = total - minimo * n_ativos

    if restante < 0:
        raise ValueError("Incremento incompatível com o número de ativos.")

    combinacoes = set()

    for divisao in itertools.combinations_with_replacement(
        range(restante + 1), n_ativos
    ):
        if sum(divisao) == restante:
            base = [minimo + d for d in divisao]
            for perm in set(itertools.permutations(base)):
                combinacoes.add(tuple(p / total for p in perm))

    return np.array(sorted(combinacoes))

# =========================
# VAR DA CARTEIRA
# =========================
def montar_df_retorno(resultados):
    retornos = []
    nomes = []

    for df, ticker in resultados:
        retornos.append(df['ret_acao'])
        nomes.append(ticker)

    df_ret = pd.concat(retornos, axis=1)
    df_ret.columns = nomes
    return df_ret.dropna()

def calcular_var_carteira(retornos, pesos, aporte_total):
    cov = retornos.cov().values
    media = retornos.mean().values

    vol = math.sqrt(pesos @ cov @ pesos)
    var_pct = media @ pesos - Z_SCORE * vol
    var_rs = abs(var_pct) * aporte_total

    return var_pct * 100, var_rs

def tabela_var_combinacoes(resultados, passo, aporte_total):
    retornos = montar_df_retorno(resultados)
    n = retornos.shape[1]

    pesos_lista = gerar_pesos(n, passo)
    linhas = []

    for pesos in pesos_lista:
        var_pct, var_rs = calcular_var_carteira(
            retornos, pesos, aporte_total
        )

        linha = {
            f"Peso {retornos.columns[i]} (%)": pesos[i] * 100
            for i in range(n)
        }
        linha["VaR %"] = var_pct
        linha["VaR R$"] = var_rs

        linhas.append(linha)

    return pd.DataFrame(linhas).sort_values("VaR R$").reset_index(drop=True)

# =========================
# INTERFACE
# =========================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Eficiência de Carteira – Zeca(AI)")
        self.inputs = []
        self.resultados = []
        self.caminho_pdf = None
        self._build()

    def _build(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill="x")

        ttk.Label(frame, text="Data (dd/mm/aaaa)").grid(row=0, column=0)
        self.data = ttk.Entry(frame, width=15)
        self.data.insert(0, datetime.today().strftime("%d/%m/%Y"))
        self.data.grid(row=0, column=1)

        ttk.Label(frame, text="Pregões").grid(row=1, column=0)
        self.n = ttk.Entry(frame, width=15)
        self.n.insert(0, "252")
        self.n.grid(row=1, column=1)

        ttk.Label(frame, text="Aporte Total da Carteira (R$)").grid(row=2, column=0)
        self.aporte_total = ttk.Entry(frame, width=15)
        self.aporte_total.insert(0, "100000")
        self.aporte_total.grid(row=2, column=1)

        ttk.Label(frame, text="Incremento dos pesos (%)").grid(row=3, column=0)
        self.incremento = ttk.Entry(frame, width=15)
        self.incremento.insert(0, "5")
        self.incremento.grid(row=3, column=1)

        self.frame_tickers = ttk.LabelFrame(frame, text="Ações")
        self.frame_tickers.grid(row=4, column=0, columnspan=3, pady=10)

        self.adicionar_ticker()

        ttk.Button(frame, text="+ Adicionar Ação",
                   command=self.adicionar_ticker).grid(row=5, column=0)

        ttk.Button(frame, text="Exportar PDF",
                   command=self.executar).grid(row=5, column=1)

        ttk.Button(frame, text="Visualizar PDF",
                   command=self.visualizar_pdf).grid(row=5, column=2)

    def adicionar_ticker(self):
        linha = ttk.Frame(self.frame_tickers)
        linha.pack(fill="x")

        ticker = ttk.Entry(linha, width=10)
        ticker.insert(0, "PETR4")
        ticker.pack(side="left", padx=5)

        self.inputs.append(ticker)

    def executar(self):
        threading.Thread(target=self._thread, daemon=True).start()

    def _thread(self):
        try:
            self.resultados.clear()

            passo = float(self.incremento.get()) / 100
            aporte_total = float(self.aporte_total.get())

            for t in self.inputs:
                df, ticker = analisar(
                    t.get(),
                    self.data.get(),
                    int(self.n.get())
                )
                self.resultados.append((df, ticker))

            tabela = tabela_var_combinacoes(
                self.resultados, passo, aporte_total
            )

            self.exportar_pdf(tabela)

            messagebox.showinfo(
                "Sucesso",
                f"Relatório gerado com {len(tabela)} combinações!"
            )

        except Exception as e:
            messagebox.showerror("Erro", str(e))

    # =========================
    # PDF
    # =========================
    def exportar_pdf(self, tabela):
        self.caminho_pdf = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")]
        )
        if not self.caminho_pdf:
            return

        linhas_por_pagina = 20

        with PdfPages(self.caminho_pdf) as pdf:
            for i in range(0, len(tabela), linhas_por_pagina):
                fatia = tabela.iloc[i:i + linhas_por_pagina]

                fig, ax = plt.subplots(figsize=A4_LANDSCAPE)
                ax.axis("off")

                ax.text(0.5, 0.94,
                        "Carteiras Ordenadas por VaR (Crescente)",
                        fontsize=16, weight="bold",
                        ha="center", color=COR_CABECALHO)

                table = ax.table(
                    cellText=np.round(fatia.values, 2),
                    colLabels=fatia.columns,
                    loc="center",
                    cellLoc="center"
                )

                table.scale(1, 1.5)

                for (row, col), cell in table.get_celld().items():
                    if row == 0:
                        cell.set_facecolor(COR_CABECALHO)
                        cell.set_text_props(color="white", weight="bold")
                    else:
                        cell.set_facecolor(COR_LINHA)
                        cell.set_text_props(color=COR_TEXTO)

                ax.text(0.5, 0.03, RODAPE, fontsize=8,
                        color="gray", ha="center",
                        transform=ax.transAxes)

                pdf.savefig(fig)
                plt.close(fig)

    def visualizar_pdf(self):
        if self.caminho_pdf and os.path.exists(self.caminho_pdf):
            os.startfile(self.caminho_pdf)
        else:
            messagebox.showwarning(
                "Aviso", "Nenhum PDF gerado ainda."
            )

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
