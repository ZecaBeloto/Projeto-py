import threading
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
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
# ANÁLISE FINANCEIRA
# =========================
def analisar(ticker_raw, data_str, n, aporte):
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

    media = df['ret_acao'].mean()
    vol = df['ret_acao'].std()
    sharpe = media / vol if vol != 0 else np.nan

    cov = np.cov(df['ret_acao'], df['ret_ibov'])[0][1]
    var_ibov = np.var(df['ret_ibov'])
    beta = cov / var_ibov if var_ibov != 0 else np.nan

    correlacao = df[['ret_acao', 'ret_ibov']].corr().iloc[0, 1]

    df['ret_acao_pct'] = df['ret_acao'] * 100
    var_param = df['ret_acao_pct'].mean() - 1.65 * df['ret_acao_pct'].std()
    var_reais = aporte * abs(var_param) / 100

    df['var_acao'] = (df['acao'] / df['acao'].iloc[0] - 1) * 100
    df['var_ibov'] = (df['ibov'] / df['ibov'].iloc[0] - 1) * 100

    info = {
        "ticker": ticker,
        "aporte": aporte,
        "sharpe": sharpe,
        "beta": beta,
        "correlacao": correlacao,
        "var_param": var_param,
        "var_reais": var_reais
    }

    return df, info

# =========================
# INTERFACE
# =========================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Análise de Risco de Carteira – Zeca(AI)")

        self.inputs = []
        self.resultados = []
        self.ultimo_pdf = None

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

        self.frame_tickers = ttk.LabelFrame(frame, text="Ações e Aportes")
        self.frame_tickers.grid(row=2, column=0, columnspan=3, pady=10, sticky="ew")

        self.adicionar_ticker()

        ttk.Button(frame, text="+ Adicionar Ação", command=self.adicionar_ticker)\
            .grid(row=3, column=0, pady=6)

        ttk.Button(frame, text="Exportar PDF A4", command=self.executar)\
            .grid(row=3, column=1, pady=6)

        self.btn_abrir = ttk.Button(
            frame, text="Abrir relatório PDF",
            command=self.abrir_pdf, state="disabled"
        )
        self.btn_abrir.grid(row=3, column=2, pady=6)

    def adicionar_ticker(self):
        linha = ttk.Frame(self.frame_tickers)
        linha.pack(fill="x", pady=2)

        ticker = ttk.Entry(linha, width=10)
        ticker.insert(0, "PETR4")
        ticker.pack(side="left", padx=5)

        ttk.Label(linha, text="Aporte em R$").pack(side="left")
        aporte = ttk.Entry(linha, width=15)
        aporte.insert(0, "10000")
        aporte.pack(side="left", padx=5)

        self.inputs.append((ticker, aporte))

    def executar(self):
        threading.Thread(target=self._thread, daemon=True).start()

    def _thread(self):
        try:
            self.resultados.clear()
            for t, a in self.inputs:
                df, info = analisar(
                    t.get(),
                    self.data.get(),
                    int(self.n.get()),
                    float(a.get())
                )
                self.resultados.append((df, info))

            self.exportar_pdf()

            self.btn_abrir.config(state="normal")
            messagebox.showinfo("Sucesso", "Relatório gerado com sucesso!")

        except Exception as e:
            messagebox.showerror("Erro", str(e))

    # =========================
    # PDF
    # =========================
    def exportar_pdf(self):
        caminho = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")]
        )
        if not caminho:
            return

        self.ultimo_pdf = caminho
        var_total = 0

        with PdfPages(caminho) as pdf:

            for df, info in self.resultados:
                # ===== Gráfico 1 =====
                fig, ax = plt.subplots(figsize=A4_LANDSCAPE)
                ax.plot(df['var_acao'], label=info['ticker'])
                ax.plot(df['var_ibov'], '--', label='Ibovespa')
                ax.legend()
                ax.grid(alpha=0.3)

                texto = (
                    f"Correlação: {info['correlacao']:.2f}\n"
                    f"Beta: {info['beta']:.2f}\n"
                    f"Sharpe: {info['sharpe']:.2f}"
                )

                ax.text(0.02, 0.95, texto, transform=ax.transAxes,
                        fontsize=12, va="top",
                        bbox=dict(boxstyle="round", fc="white", ec="gray"))

                ax.text(0.5, 0.02, RODAPE, fontsize=8,
                        color="gray", ha="center", transform=ax.transAxes)

                pdf.savefig(fig)
                plt.close(fig)

                # ===== Gráfico 2 =====
                fig, ax = plt.subplots(figsize=A4_LANDSCAPE)
                valores = df['ret_acao_pct']
                cores = ['green' if x >= 0 else 'red' for x in valores]
                ax.bar(df.index, valores, color=cores)

                ax.axhline(info['var_param'], linestyle='--',
                           color='darkred', label='VaR Paramétrico (%)')
                ax.legend()

                ax.text(
                    0.02, 0.95,
                    f"VaR (%): {info['var_param']:.2f}%\n"
                    f"VaR (R$): R$ {info['var_reais']:,.2f}",
                    transform=ax.transAxes,
                    fontsize=12, va="top", color="darkred"
                )

                ymin = min(valores.min(), info['var_param'])
                ymax = max(valores.max(), 0)
                margem = (ymax - ymin) * 0.25
                ax.set_ylim(ymin - margem, ymax + margem)

                offset = (ymax - ymin) * 0.08

                for idx, v in valores.items():
                    if v < info['var_param']:
                        ax.text(
                            idx, v - offset,
                            idx.strftime("%d/%m/%Y"),
                            rotation=90,
                            fontsize=8,
                            ha='center',
                            va='top'
                        )

                ax.grid(axis='y', alpha=0.3)
                fig.subplots_adjust(bottom=0.20)

                ax.text(0.5, 0.02, RODAPE, fontsize=8,
                        color="gray", ha="center", transform=ax.transAxes)

                pdf.savefig(fig)
                plt.close(fig)

                var_total += info['var_reais']

            # ===== Página resumo =====
            fig, ax = plt.subplots(figsize=A4_LANDSCAPE)
            ax.axis("off")

            y = 0.85
            ax.text(0.05, y, "Resumo de Risco da Carteira",
                    fontsize=16, weight="bold")
            y -= 0.08

            for _, info in self.resultados:
                ax.text(
                    0.05, y,
                    f"{info['ticker']} | "
                    f"Aporte: R$ {info['aporte']:,.2f} | "
                    f"VaR (%): {info['var_param']:.2f}% | "
                    f"VaR (R$): R$ {info['var_reais']:,.2f}",
                    fontsize=12
                )
                y -= 0.06

            ax.text(
                0.05, y - 0.04,
                f"VaR Total da Carteira (soma simples): "
                f"R$ {var_total:,.2f}",
                fontsize=14, weight="bold", color="darkred"
            )

            ax.text(0.5, 0.02, RODAPE, fontsize=8,
                    color="gray", ha="center", transform=ax.transAxes)

            pdf.savefig(fig)
            plt.close(fig)

            # ===== Página explicativa =====
            texto = (
                "** Entendendo os Indicadores de Risco da Carteira**\n\n"
                "-> O que é Correlação?\n"
                "A correlação indica o quanto o preço de uma ação tende a se mover junto com o Ibovespa.\n\n"
                "• Correlação próxima de +1: a ação costuma subir e cair junto com o índice.\n"
                "• Correlação próxima de 0: a ação se move de forma mais independente.\n"
                "• Correlação negativa: a ação tende a se mover na direção oposta ao índice.\n\n"
                "• Por que isso importa?\n"
                "Ativos com correlação menor ajudam a diversificar a carteira, reduzindo o risco total.\n\n"
                "-> O que é Beta?\n"
                "O beta mede a sensibilidade da ação às oscilações do mercado (Ibovespa).\n\n"
                "• Beta = 1 → a ação oscila, em média, como o mercado.\n"
                "• Beta > 1 → a ação tende a oscilar mais que o mercado (maior risco).\n"
                "• Beta < 1 → a ação tende a oscilar menos que o mercado (menor risco).\n\n"
                "👉 Exemplo prático:\n"
                "Se uma ação tem beta 1,2, uma variação de 1% do Ibovespa tende a gerar cerca de 1,2% nessa ação.\n\n"
                "-> O que é o Índice de Sharpe?\n"
                "O Índice de Sharpe mede quanto retorno um investimento entrega para cada unidade de risco assumida.\n\n"
                "• Sharpe maior que 1: boa relação risco-retorno.\n"
                "• Sharpe próximo de 0: retorno baixo para o risco assumido.\n"
                "• Sharpe negativo: retorno inferior ao ativo livre de risco.\n\n"
                "-> O que é VaR (Value at Risk)?\n"
                "O VaR estima quanto um investimento pode perder em um único dia, em condições normais de mercado, "
                "com um determinado nível de confiança.\n\n"
                "• VaR em %: mostra a perda percentual estimada.\n"
                "• VaR em R$: mostra o impacto financeiro considerando o valor investido.\n\n"
                "Exemplo prático:\n"
                "Se o VaR for -2,0% e o investimento for de R$ 10.000, espera-se que a perda não ultrapasse "
                "R$ 200 em 95% dos dias.\n\n"
                "⚠️ Importante:\n"
                "O VaR não elimina o risco e não prevê eventos extremos, mas é uma ferramenta fundamental "
                "para mensurar e controlar o risco da carteira."
            )

            fig, ax = plt.subplots(figsize=A4_PORTRAIT)
            ax.axis("off")

            # espaço inferior reservado para o rodapé
            fig.subplots_adjust(bottom=0.00010)

            ax.text(0.05, 0.92, texto, fontsize=12, va="top", wrap=True)

            ax.text(
                0.5, 0.04,
                RODAPE,
                fontsize=8,
                color="gray",
                ha="center",
                transform=ax.transAxes
            )

            pdf.savefig(fig)
            plt.close(fig)

    def abrir_pdf(self):
        if self.ultimo_pdf and os.path.exists(self.ultimo_pdf):
            os.startfile(self.ultimo_pdf)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
