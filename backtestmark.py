import threading
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
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
A4_PORTRAIT = (8.27, 11.69)

Z_SCORE = 1.65
PESO_MIN = 0.05

COR_CABECALHO = "#1f4e79"
COR_LINHA = "#ddebf7"
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

    for divisao in itertools.combinations_with_replacement(range(restante + 1), n_ativos):
        if sum(divisao) == restante:
            base = [minimo + d for d in divisao]
            for perm in set(itertools.permutations(base)):
                combinacoes.add(tuple(p / total for p in perm))

    return np.array(sorted(combinacoes))

# =========================
# VAR E RETORNOS
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

# =========================
# SIMULAÇÃO DE APORTES
# =========================
def simular_montante(df, aporte_mensal, meses, capital_inicial):
    capital = capital_inicial
    retorno_medio = df['ret_acao'].mean()

    for _ in range(meses):
        capital = (capital + aporte_mensal) * (1 + retorno_medio)

    return capital

# =========================
# TABELA FINAL
# =========================
def tabela_var_combinacoes(resultados, passo, aporte_total, aporte_mensal):
    retornos = montar_df_retorno(resultados)
    n = retornos.shape[1]
    meses = int(len(retornos) / 21)

    pesos_lista = gerar_pesos(n, passo)
    linhas = []

    for pesos in pesos_lista:
        var_pct, var_rs = calcular_var_carteira(retornos, pesos, aporte_total)

        montante_final = 0
        for (df, _), peso in zip(resultados, pesos):
            capital_inicial = aporte_total * peso
            montante_final += simular_montante(
                df,
                aporte_mensal * peso,
                meses,
                capital_inicial
            )

        linha = {f"Peso {retornos.columns[i]} (%)": pesos[i] * 100 for i in range(n)}
        linha["VaR %"] = var_pct
        linha["VaR R$"] = var_rs
        linha["Montante Final (R$)"] = montante_final

        linhas.append(linha)

    return pd.DataFrame(linhas).sort_values("VaR %").reset_index(drop=True)

# =========================
# GRÁFICO INTERATIVO
# =========================
def mostrar_grafico_interativo(tabela):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        tabela["VaR %"],
        tabela["Montante Final (R$)"],
        s=40,
        picker=True
    )

    ax.invert_xaxis()
    ax.set_xlabel("VaR % (Risco)")
    ax.set_ylabel("Montante Final (R$)")
    ax.set_title("Risco x Retorno – Carteiras Simuladas")
    ax.grid(True, linestyle="--", alpha=0.4)

    anotacao = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->")
    )
    anotacao.set_visible(False)

    def on_pick(event):
        ind = event.ind[0]
        linha = tabela.iloc[ind]

        x = linha["VaR %"]
        y = linha["Montante Final (R$)"]

        pesos_txt = ""
        for col in tabela.columns:
            if col.startswith("Peso"):
                ativo = col.replace("Peso ", "").replace(" (%)", "")
                pesos_txt += f"{ativo}: {linha[col]:.1f}%\n"

        anotacao.xy = (x, y)
        anotacao.set_text(
            f"VaR %: {x:.2f}\n"
            f"Montante: R$ {y:,.2f}\n\n"
            f"Composição:\n{pesos_txt}"
        )
        anotacao.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)

    # carteira atual (já existente)
    pesos_cols = [c for c in tabela.columns if c.startswith("Peso")]
    pesos_usuario = []

    for col in pesos_cols:
        val = simpledialog.askfloat(
            "Carteira Atual",
            f"Informe o {col}:",
            minvalue=0,
            maxvalue=100
        )
        if val is None:
            break
        pesos_usuario.append(val)

    if len(pesos_usuario) == len(pesos_cols):
        pesos_usuario = np.array(pesos_usuario)
        base = tabela[pesos_cols].values
        dist = np.linalg.norm(base - pesos_usuario, axis=1)
        idx = np.argmin(dist)

        x_u = tabela.iloc[idx]["VaR %"]
        y_u = tabela.iloc[idx]["Montante Final (R$)"]

        ax.scatter(x_u, y_u, s=120, marker="*", zorder=5)
        ax.annotate(
            "Carteira Atual",
            (x_u, y_u),
            textcoords="offset points",
            xytext=(10, -15),
            fontsize=10,
            weight="bold"
        )

    plt.show()

# =========================
# INTERFACE
# =========================
class App:

    def exportar_pdf(self, tabela):
        self.caminho_pdf = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")]
        )
        if not self.caminho_pdf:
            return

        linhas_por_pagina = 20

        with PdfPages(self.caminho_pdf) as pdf:

            fig, ax = plt.subplots(figsize=A4_PORTRAIT)
            ax.axis("off")

            ax.text(0.5, 0.60,
                    "Relatório de alocação eficiente de carteira",
                    fontsize=23, color="lightblue",
                    ha="center", va="center", weight="bold")

            ax.text(0.5, 0.52,
                    f"Data final da análise: {self.data.get()}\n"
                    f"Janela considerada: {self.n.get()} pregões",
                    fontsize=14, ha="center", va="center", color="gray")

            ax.text(0.5, 0.06, RODAPE,
                    fontsize=9, color="gray", ha="center")

            pdf.savefig(fig)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=A4_LANDSCAPE)
            ax.scatter(
                tabela["VaR %"],
                tabela["Montante Final (R$)"],
                s=60,
                color="darkblue"
            )
            ax.invert_xaxis()
            ax.set_xlabel("VaR % (Risco)")
            ax.set_ylabel("Montante Final (R$)")
            ax.set_title("Fronteira Eficiente – Risco x Retorno")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.text(0.5, 0.03, RODAPE,
                    fontsize=8, color="gray",
                    ha="center", transform=ax.transAxes)
            pdf.savefig(fig)
            plt.close(fig)

            for i in range(0, len(tabela), linhas_por_pagina):
                fatia = tabela.iloc[i:i + linhas_por_pagina]

                fig, ax = plt.subplots(figsize=A4_LANDSCAPE)
                ax.axis("off")

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

                ax.text(0.5, 0.03, RODAPE,
                        fontsize=8, color="gray",
                        ha="center", transform=ax.transAxes)

                pdf.savefig(fig)
                plt.close(fig)

    def __init__(self, root):
        self.root = root
        self.root.title("Alocação de Carteira – Zeca(AI)")
        self.inputs = []
        self.resultados = []
        self.caminho_pdf = None

        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        self._build()

    def _build(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill="x")

        ttk.Label(frame, text="Data").grid(row=0, column=0)
        self.data = ttk.Entry(frame, width=15)
        self.data.insert(0, datetime.today().strftime("%d/%m/%Y"))
        self.data.grid(row=0, column=1)

        ttk.Label(frame, text="Pregões").grid(row=1, column=0)
        self.n = ttk.Entry(frame, width=15)
        self.n.insert(0, "252")
        self.n.grid(row=1, column=1)

        ttk.Label(frame, text="Aporte Total").grid(row=2, column=0)
        self.aporte_total = ttk.Entry(frame, width=15)
        self.aporte_total.insert(0, "100000")
        self.aporte_total.grid(row=2, column=1)

        ttk.Label(frame, text="Aporte Mensal").grid(row=3, column=0)
        self.aporte_mensal = ttk.Entry(frame, width=15)
        self.aporte_mensal.insert(0, "2000")
        self.aporte_mensal.grid(row=3, column=1)

        ttk.Label(frame, text="Incremento (%)").grid(row=4, column=0)
        self.incremento = ttk.Entry(frame, width=15)
        self.incremento.insert(0, "5")
        self.incremento.grid(row=4, column=1)

        self.frame_tickers = ttk.LabelFrame(frame, text="Ações")
        self.frame_tickers.grid(row=5, column=0, columnspan=3, pady=10)

        self.adicionar_ticker()

        ttk.Button(frame, text="+ Adicionar Ação",
                   command=self.adicionar_ticker).grid(row=6, column=0)

        ttk.Button(frame, text="Exportar PDF",
                   command=self.executar).grid(row=6, column=1)

        ttk.Button(frame, text="Visualizar PDF",
                   command=self.visualizar_pdf).grid(row=6, column=2)

        ttk.Button(frame, text="Gráfico Risco x Retorno",
                   command=self.abrir_grafico).grid(row=7, column=1, pady=5)

    def adicionar_ticker(self):
        linha = ttk.Frame(self.frame_tickers)
        linha.pack(fill="x")

        ticker = ttk.Entry(linha, width=10)
        ticker.insert(0, "PETR4")
        ticker.pack(side="left", padx=5)

        btn = ttk.Button(
            linha,
            text="❌",
            command=lambda: self.remover_ticker(linha, ticker)
        )
        btn.pack(side="left")

        self.inputs.append(ticker)

    def remover_ticker(self, frame, ticker):
        if len(self.inputs) == 1:
            messagebox.showwarning("Aviso", "É necessário manter ao menos um ticker.")
            return
        frame.destroy()
        self.inputs.remove(ticker)

    def executar(self):
        threading.Thread(target=self._thread, daemon=True).start()

    def _thread(self):
        try:
            self.resultados.clear()

            passo = float(self.incremento.get()) / 100
            aporte_total = float(self.aporte_total.get())
            aporte_mensal = float(self.aporte_mensal.get())

            for t in self.inputs:
                df, ticker = analisar(t.get(), self.data.get(), int(self.n.get()))
                self.resultados.append((df, ticker))

            self.tabela = tabela_var_combinacoes(
                self.resultados, passo, aporte_total, aporte_mensal
            )

            self.exportar_pdf(self.tabela)

            messagebox.showinfo(
                "Sucesso",
                f"Relatório gerado com {len(self.tabela)} combinações!"
            )

        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def abrir_grafico(self):
        if hasattr(self, "tabela"):
            mostrar_grafico_interativo(self.tabela)
        else:
            messagebox.showwarning("Aviso", "Execute a simulação antes.")

    def visualizar_pdf(self):
        if self.caminho_pdf and os.path.exists(self.caminho_pdf):
            os.startfile(self.caminho_pdf)
        else:
            messagebox.showwarning("Aviso", "Nenhum PDF gerado ainda.")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()

def main():
    root = tk.Tk()
    App(root)
    root.mainloop()