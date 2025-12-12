

import threading
import io
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams.update({'figure.max_open_warning': 0})


# -------------------- Utilitários -------------------- #
def extrair_preco(valor):
    """Extrai um float de um valor possivelmente Series/int/float."""
    try:
        if isinstance(valor, pd.Series):
            return float(valor.iloc[0])
        elif isinstance(valor, (int, float, np.floating, np.integer)):
            return float(valor)
        else:
            return float(valor)
    except Exception:
        return None


def buscar_dados_ibovespa(data_obj):
    """Busca os últimos 252 pregões do Ibovespa até a data informada."""
    try:
        start_date = data_obj - timedelta(days=900)
        end_date = data_obj + timedelta(days=1)

        df = yf.download(
            '^BVSP',
            start=str(start_date),
            end=str(end_date),
            progress=False,
            auto_adjust=False,
            prepost=False
        )

        if df is None or df.empty:
            return None

        if 'Close' in df.columns:
            close_data = df['Close']
        elif 'Adj Close' in df.columns:
            close_data = df['Adj Close']
        else:
            return None

        close_data.index = pd.to_datetime(close_data.index)
        close_data = close_data.sort_index()
        data_limite = pd.to_datetime(data_obj)
        close_data = close_data[close_data.index <= data_limite]
        close_data = close_data.dropna()
        close_data = close_data[close_data > 0]

        if close_data.empty:
            return None

        return close_data.tail(252)

    except Exception as e:
        print(f"Erro ao buscar dados do Ibovespa: {e}")
        return None


# -------------------- Análise e gráficos -------------------- #
def analisar(ticker_raw, data_str):
    """Faz toda a lógica de baixar dados, calcular métricas e construir figuras matplotlib.
    Retorna (fig1, fig2, resumo_dict) ou (None, None, error_message)."""
    try:
        ticker = ticker_raw.strip().upper()
        if not ticker:
            return None, None, "Ticker vazio."

        if not ticker.endswith(".SA"):
            ticker = ticker + ".SA"

        # parse date em vários formatos
        formatos = [
            "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d",
            "%d/%m/%y", "%d-%m-%y", "%Y/%m/%d",
            "%d.%m.%Y", "%d.%m.%y"
        ]
        data_obj = None
        for fmt in formatos:
            try:
                data_obj = datetime.strptime(data_str.strip(), fmt).date()
                break
            except Exception:
                continue

        if data_obj is None:
            return None, None, f"Erro: não consegui entender a data '{data_str}'."

        # período para busca
        start_date = data_obj - timedelta(days=900)
        end_date = data_obj + timedelta(days=1)

        df = yf.download(
            ticker,
            start=str(start_date),
            end=str(end_date),
            progress=False,
            auto_adjust=False,
            prepost=False
        )

        if df is None or df.empty:
            return None, None, f"Nenhum dado encontrado para {ticker}."

        # extrai fechamento
        if 'Close' in df.columns:
            close_data = df['Close']
        elif 'Adj Close' in df.columns:
            close_data = df['Adj Close']
        else:
            return None, None, f"Coluna de fechamento não encontrada. Colunas: {list(df.columns)}"

        close_data.index = pd.to_datetime(close_data.index)
        close_data = close_data.sort_index()
        data_limite = pd.to_datetime(data_obj)
        close_data = close_data[close_data.index <= data_limite]
        close_data = close_data.dropna()
        close_data = close_data[close_data > 0]

        if close_data.empty:
            return None, None, f"Nenhum fechamento válido disponível até {data_obj.strftime('%d/%m/%Y')}."

        ultimos_252 = close_data.tail(252)
        n = len(ultimos_252)

        # arrays de preços e variações
        precos = [extrair_preco(v) for v in ultimos_252.values]
        # garantir numpy array para operações
        precos_arr = np.array([np.nan if v is None else v for v in precos], dtype=float)

        variacoes_float = [0.0] * len(precos_arr)
        for i in range(1, len(precos_arr)):
            if np.isnan(precos_arr[i]) or np.isnan(precos_arr[i-1]) or precos_arr[i-1] == 0:
                variacoes_float[i] = 0.0
            else:
                variacoes_float[i] = ((precos_arr[i] - precos_arr[i-1]) / precos_arr[i-1]) * 100.0

        # média variações ação
        media_acao = float(np.nanmean([v for v in variacoes_float if not np.isnan(v)])) if len(variacoes_float) > 0 else 0.0

        # buscar ibov
        ultimos_252_ibov = buscar_dados_ibovespa(data_obj)
        precos_ibov_lista = []
        variacoes_ibov = []
        media_ibov = 0.0
        if ultimos_252_ibov is not None and not ultimos_252_ibov.empty:
            precos_ibov_lista = [extrair_preco(v) for v in ultimos_252_ibov.values]
            # calc var pct ibov
            prec_ibov_arr = np.array([np.nan if v is None else v for v in precos_ibov_lista], dtype=float)
            variacoes_ibov = [0.0] * len(prec_ibov_arr)
            for i in range(1, len(prec_ibov_arr)):
                if np.isnan(prec_ibov_arr[i]) or np.isnan(prec_ibov_arr[i-1]) or prec_ibov_arr[i-1] == 0:
                    variacoes_ibov[i] = 0.0
                else:
                    variacoes_ibov[i] = ((prec_ibov_arr[i] - prec_ibov_arr[i-1]) / prec_ibov_arr[i-1]) * 100.0
            media_ibov = float(np.nanmean([v for v in variacoes_ibov if not np.isnan(v)])) if len(variacoes_ibov) > 0 else 0.0

        # var acumulada ação e ibov
        var_pct_acumulada_acao = []
        if len(precos_arr) > 0 and not np.isnan(precos_arr[0]) and precos_arr[0] != 0:
            base_acao = precos_arr[0]
            for i in range(len(precos_arr)):
                if np.isnan(precos_arr[i]):
                    var_pct_acumulada_acao.append(0.0)
                else:
                    var_pct_acumulada_acao.append(((precos_arr[i] - base_acao) / base_acao) * 100.0)
        else:
            var_pct_acumulada_acao = [0.0] * len(precos_arr)

        var_pct_acumulada_ibov = []
        if precos_ibov_lista:
            prec_ibov_arr = np.array([np.nan if v is None else v for v in precos_ibov_lista], dtype=float)
            if not np.isnan(prec_ibov_arr[0]) and prec_ibov_arr[0] != 0:
                base_ibov = prec_ibov_arr[0]
                for i in range(len(prec_ibov_arr)):
                    if np.isnan(prec_ibov_arr[i]):
                        var_pct_acumulada_ibov.append(0.0)
                    else:
                        var_pct_acumulada_ibov.append(((prec_ibov_arr[i] - base_ibov) / base_ibov) * 100.0)
            else:
                var_pct_acumulada_ibov = [0.0] * len(prec_ibov_arr)

        # datas para eixos
        datas = [pd.Timestamp(idx).strftime('%d/%m') for idx in ultimos_252.index]
        datas_ibov = []
        if ultimos_252_ibov is not None and not ultimos_252_ibov.empty:
            datas_ibov = [pd.Timestamp(idx).strftime('%d/%m') for idx in ultimos_252_ibov.index]

        # ----------------- Criar figuras matplotlib ----------------- #
        # Figura 1: Variação Acumulada - ação vs ibov (mesmo plano)
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        x_acao = np.arange(len(var_pct_acumulada_acao))
        ax1.plot(x_acao, var_pct_acumulada_acao, marker='o', linestyle='-', label=f'{ticker} (Acumulado)')
        if var_pct_acumulada_ibov:
            x_ibov = np.arange(len(var_pct_acumulada_ibov))
            # alinhar os eixos horizontais: se tamanhos diferentes, mostrar ambos (mesmo plano)
            ax1.plot(x_ibov, var_pct_acumulada_ibov, marker='s', linestyle='--', label='Ibovespa (Acumulado)')
        ax1.set_title(f'Variação Percentual Acumulada - {ticker} vs Ibovespa')
        ax1.set_xlabel('Dias (do mais antigo para o mais recente)')
        ax1.set_ylabel('Variação (%)')
        ax1.axhline(y=0, linestyle='--', linewidth=0.8, alpha=0.5)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Figura 2: Variação Percentual Diária - barras coloridas
        fig2, ax2 = plt.subplots(figsize=(10, 4.5))
        x = np.arange(len(variacoes_float))
        cores = ['green' if v >= 0 else 'red' for v in variacoes_float]
        ax2.bar(x, variacoes_float, color=cores, alpha=0.8, width=0.8)
        ax2.set_title(f'Variação Percentual Diária (%) - {ticker}')
        ax2.set_xlabel('Dias (do mais antigo para o mais recente)')
        ax2.set_ylabel('Variação (%)')
        ax2.axhline(y=0, linestyle='-', linewidth=0.8)
        ax2.axhline(y=media_acao, linestyle='--', linewidth=1.0, alpha=0.8, label=f'Média: {media_acao:.4f}%')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend(loc='best', fontsize=9)

        # Ajustes visuais
        fig1.tight_layout()
        fig2.tight_layout()

        resumo = {
            'ticker': ticker,
            'data_referencia': data_obj.strftime('%d/%m/%Y'),
            'n_pregoes_acao': n,
            'media_acao_pct': media_acao,
            'media_ibov_pct': media_ibov,
            'dif_acao_ibov_pct': media_acao - media_ibov
        }

        return fig1, fig2, resumo

    except Exception as e:
        return None, None, f"Erro inesperado durante a análise: {e}"


# -------------------- Funções de UI -------------------- #
class App:
    def __init__(self, root):
        self.root = root
        root.title("Análise: Ação vs Ibovespa - GUI")
        self.create_widgets()
        self.fig1 = None
        self.fig2 = None
        self.resumo = None
        self.canvas1 = None
        self.canvas2 = None

    def create_widgets(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.grid(column=0, row=0, sticky='nsew')

        ttk.Label(frm, text="Ticker (ex: PETR4 ou PETR4.SA):").grid(column=0, row=0, sticky='w')
        self.ticker_entry = ttk.Entry(frm, width=20)
        self.ticker_entry.grid(column=1, row=0, sticky='w')
        self.ticker_entry.insert(0, "PETR4")  # exemplo

        ttk.Label(frm, text="Data (ex: 20/12/2024):").grid(column=0, row=1, sticky='w')
        self.date_entry = ttk.Entry(frm, width=20)
        self.date_entry.grid(column=1, row=1, sticky='w')
        self.date_entry.insert(0, datetime.today().strftime('%d/%m/%Y'))

        self.status_label = ttk.Label(frm, text="", foreground="blue")
        self.status_label.grid(column=0, row=2, columnspan=2, sticky='w', pady=(6, 6))

        btn_frame = ttk.Frame(frm)
        btn_frame.grid(column=0, row=3, columnspan=2, sticky='w')

        self.run_btn = ttk.Button(btn_frame, text="Gerar Análise", command=self.on_run)
        self.run_btn.grid(column=0, row=0, padx=(0, 8))

        self.save_btn = ttk.Button(btn_frame, text="Salvar relatório em PDF", command=self.on_save_pdf, state='disabled')
        self.save_btn.grid(column=1, row=0)

        # Área de visualização dos gráficos
        self.preview_frame = ttk.Frame(self.root, padding=6)
        self.preview_frame.grid(column=0, row=1, sticky='nsew')

        # Painel de texto do resumo
        self.text_frame = ttk.Frame(self.root, padding=6)
        self.text_frame.grid(column=0, row=2, sticky='nsew')
        ttk.Label(self.text_frame, text="Resumo:").grid(column=0, row=0, sticky='w')
        self.text_summary = tk.Text(self.text_frame, height=6, wrap='word')
        self.text_summary.grid(column=0, row=1, sticky='nsew')

        # Expansão
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.preview_frame.columnconfigure(0, weight=1)
        self.preview_frame.columnconfigure(1, weight=1)

    def set_status(self, msg, color="blue"):
        self.status_label.config(text=msg, foreground=color)
        self.root.update_idletasks()

    def on_run(self):
        ticker = self.ticker_entry.get()
        data_str = self.date_entry.get()
        if not ticker or not data_str:
            messagebox.showwarning("Aviso", "Preencha ticker e data.")
            return
        # desabilitar botões enquanto processa
        self.run_btn.config(state='disabled')
        self.save_btn.config(state='disabled')
        self.text_summary.delete('1.0', tk.END)
        self.clear_previews()
        self.set_status("Buscando dados... (pode demorar alguns segundos)", "black")
        # rodar em thread para não travar UI
        thread = threading.Thread(target=self._run_thread, args=(ticker, data_str), daemon=True)
        thread.start()

    def _run_thread(self, ticker, data_str):
        fig1, fig2, resumo_or_msg = analisar(ticker, data_str)
        if fig1 is None and isinstance(resumo_or_msg, str):
            self.set_status(resumo_or_msg, "red")
            self.run_btn.config(state='normal')
            return
        # sucesso
        self.fig1, self.fig2, self.resumo = fig1, fig2, resumo_or_msg
        # mostrar figuras na UI (no thread principal)
        self.root.after(0, self.show_results)

    def clear_previews(self):
        # remover canvas anteriores
        if self.canvas1:
            self.canvas1.get_tk_widget().destroy()
            self.canvas1 = None
        if self.canvas2:
            self.canvas2.get_tk_widget().destroy()
            self.canvas2 = None

    def show_results(self):
        self.clear_previews()
        # embutir fig1 e fig2 em widgets
        if self.fig1:
            self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.preview_frame)
            widget1 = self.canvas1.get_tk_widget()
            widget1.grid(column=0, row=0, sticky='nsew', padx=4, pady=4)
            self.canvas1.draw()

        if self.fig2:
            self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.preview_frame)
            widget2 = self.canvas2.get_tk_widget()
            widget2.grid(column=1, row=0, sticky='nsew', padx=4, pady=4)
            self.canvas2.draw()

        # preencher resumo textual
        if isinstance(self.resumo, dict):
            txt = []
            txt.append(f"Análise para {self.resumo.get('ticker')} até {self.resumo.get('data_referencia')}")
            txt.append(f"Número de pregões (ação): {self.resumo.get('n_pregoes_acao')}")
            txt.append(f"Média de variação diária - ação: {self.resumo.get('media_acao_pct'):+.4f}%")
            txt.append(f"Média de variação diária - Ibovespa: {self.resumo.get('media_ibov_pct'):+.4f}%")
            txt.append(f"Diferença (Ação - Ibov): {self.resumo.get('dif_acao_ibov_pct'):+.4f}%")
            resumo_text = "\n".join(txt)
            self.text_summary.insert('1.0', resumo_text)

        self.set_status("Análise concluída.", "green")
        self.run_btn.config(state='normal')
        self.save_btn.config(state='normal')

    def on_save_pdf(self):
        if not (self.fig1 and self.fig2 and isinstance(self.resumo, dict)):
            messagebox.showwarning("Aviso", "Não há análise para salvar. Gere a análise primeiro.")
            return

        # pedir nome do arquivo
        default_name = f"relatorio_{self.resumo.get('ticker')}_{self.resumo.get('data_referencia').replace('/','-')}.pdf"
        path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")], initialfile=default_name)
        if not path:
            return

        try:
            # salvar figuras em PDF com uma primeira página de texto
            with PdfPages(path) as pdf:
                # página de resumo (texto)
                fig_text = plt.figure(figsize=(8.27, 11.69))  # A4 portrait tamanho em polegadas
                fig_text.clf()
                txt = [
                    f"Relatório: {self.resumo.get('ticker')}",
                    f"Data de referência: {self.resumo.get('data_referencia')}",
                    "",
                    f"Número de pregões (ação): {self.resumo.get('n_pregoes_acao')}",
                    f"Média de variação diária - ação: {self.resumo.get('media_acao_pct'):+.4f}%",
                    f"Média de variação diária - Ibovespa: {self.resumo.get('media_ibov_pct'):+.4f}%",
                    f"Diferença (Ação - Ibov): {self.resumo.get('dif_acao_ibov_pct'):+.4f}%",
                    "",
                    "Observações:",
                    "- Gráfico 1: Variação percentual acumulada (Ação vs Ibovespa).",
                    "- Gráfico 2: Variação percentual diária (barras verdes para ganhos, vermelhas para perdas).",
                ]
                fig_text.text(0.05, 0.95, "Relatório de Análise - Ação vs Ibovespa", fontsize=14, weight='bold')
                y = 0.88
                for line in txt:
                    fig_text.text(0.05, y, line, fontsize=10, va='top')
                    y -= 0.035
                fig_text.tight_layout()
                pdf.savefig(fig_text)
                plt.close(fig_text)

                # salvar figuras (copiando para PDF)
                pdf.savefig(self.fig1)
                pdf.savefig(self.fig2)

            messagebox.showinfo("Sucesso", f"Relatório salvo em:\n{path}")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao salvar PDF: {e}")


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
