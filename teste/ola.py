import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

def extrair_preco(valor):
    """Extrai o preço de um valor que pode ser float, Series ou outro tipo."""
    try:
        if isinstance(valor, pd.Series):
            return float(valor.iloc[0])
        elif isinstance(valor, (int, float)):
            return float(valor)
        else:
            return float(valor)
    except (ValueError, TypeError, KeyError) as e:
        print(f"Erro ao extrair preço de {valor}: {e}")
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

        # Extrai apenas a coluna Close (ou Adj Close)
        if isinstance(df, pd.Series):
            close_data = df
        elif 'Close' in df.columns:
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

def main():
    while True:
        try:
            ticker = input("Qual é o ticker da ação que você deseja analisar? (ex: PETR4): ").strip().upper()
            if not ticker:
                print("Ticker vazio.")
                continue

            # adiciona sufixo .SA caso não esteja presente
            if not ticker.endswith(".SA"):
                ticker = ticker + ".SA"

            data_str = input("Qual dia você deseja analisar? (ex: 20/12/2024 ou 2024-12-20): ").strip()
            formatos = [
                "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d",
                "%d/%m/%y", "%d-%m-%y", "%Y/%m/%d",
                "%d.%m.%Y", "%d.%m.%y"
            ]

            data_obj = None
            for fmt in formatos:
                try:
                    data_obj = datetime.strptime(data_str, fmt).date()
                    break
                except ValueError:
                    continue

            if data_obj is None:
                print(f"Erro: não consegui entender a data '{data_str}'.")
                continue

            data_str_formatada = data_obj.strftime('%d/%m/%Y')
            print(f"✓ Data reconhecida: {data_str_formatada}")
            print(f"Buscando dados para {ticker}...")

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
                print(f"Nenhum dado encontrado para {ticker}.")
                continue

            # extrai coluna de fechamento
            if isinstance(df, pd.Series):
                close_data = df
            elif 'Close' in df.columns:
                close_data = df['Close']
            elif 'Adj Close' in df.columns:
                close_data = df['Adj Close']
            else:
                print(f"Coluna de fechamento não encontrada. Colunas disponíveis: {list(df.columns)}")
                continue

            close_data.index = pd.to_datetime(close_data.index)
            close_data = close_data.sort_index()

            data_limite = pd.to_datetime(data_obj)
            close_data = close_data[close_data.index <= data_limite]

            close_data = close_data.dropna()
            close_data = close_data[close_data > 0]

            if close_data.empty:
                print(f"Nenhum fechamento válido disponível até {data_str_formatada}.")
                continue

            ultimos_252 = close_data.tail(252)
            n = len(ultimos_252)
            print(f"✓ Encontrados {n} pregões até {data_str_formatada} (máximo 252).")

            # ===== PRIMEIRA LISTA: Preços de fechamento =====
            print('\n' + '='*60)
            print('PREÇOS DE FECHAMENTO - Do mais recente para o mais antigo:')
            print('='*60)

            for i in range(len(ultimos_252) - 1, -1, -1):
                data_idx = ultimos_252.index[i]
                preco = extrair_preco(ultimos_252.iloc[i])
                data_formatada = pd.Timestamp(data_idx).strftime('%d/%m/%Y')
                if preco is not None:
                    print(f"{data_formatada}: R$ {preco:.2f}")
                else:
                    print(f"{data_formatada}: ERRO ao ler preço")

            # ===== SEGUNDA LISTA: Variação percentual diária =====
            print('\n' + '='*60)
            print('VARIAÇÃO PERCENTUAL - Do mais recente para o mais antigo:')
            print('='*60)
            print(f"{'Data':<12} {'Variação %':<15}")
            print('-' * 50)

            variacoes = []
            precos_lista = []

            for i in range(len(ultimos_252)):
                preco_atual = extrair_preco(ultimos_252.iloc[i])
                if preco_atual is None:
                    print(f"Erro: Não consegui extrair preço do índice {i}")
                    preco_atual = np.nan

                precos_lista.append(preco_atual)

                if i > 0 and not np.isnan(precos_lista[i-1]) and not np.isnan(preco_atual) and precos_lista[i-1] != 0:
                    variacao_pct = ((preco_atual - precos_lista[i - 1]) / precos_lista[i - 1]) * 100
                    variacao_str = f"{variacao_pct:+.2f}%"
                else:
                    variacao_str = "N/A (primeiro)"
                variacoes.append(variacao_str)

            for i in range(len(ultimos_252) - 1, -1, -1):
                data_idx = ultimos_252.index[i]
                data_formatada = pd.Timestamp(data_idx).strftime('%d/%m/%Y')
                print(f"{data_formatada}  {variacoes[i]:>14}")

            if n < 252:
                print(f"\nℹ Apenas {n} pregões encontrados (menos que 252).")

            # ===== TERCEIRA LISTA: Preços de fechamento do Ibovespa =====
            print('\n' + '='*60)
            print('IBOVESPA - Preços de fechamento (últimos 252 pregões):')
            print('='*60)

            ultimos_252_ibov = buscar_dados_ibovespa(data_obj)

            if ultimos_252_ibov is not None and not ultimos_252_ibov.empty:
                n_ibov = len(ultimos_252_ibov)
                print(f"✓ Encontrados {n_ibov} pregões do Ibovespa até {data_str_formatada}")
                print('-' * 50)

                for i in range(len(ultimos_252_ibov) - 1, -1, -1):
                    data_idx = ultimos_252_ibov.index[i]
                    preco = extrair_preco(ultimos_252_ibov.iloc[i])
                    data_formatada = pd.Timestamp(data_idx).strftime('%d/%m/%Y')
                    if preco is not None:
                        print(f"{data_formatada}: {preco:,.2f}")
                    else:
                        print(f"{data_formatada}: ERRO ao ler preço")
            else:
                print("✗ Não consegui buscar dados do Ibovespa para este período.")

            # ===== QUARTA LISTA: Variação percentual diária do Ibovespa =====
            print('\n' + '='*60)
            print('VARIAÇÃO PERCENTUAL DO IBOVESPA - Do mais recente para o mais antigo:')
            print('='*60)
            print(f"{'Data':<12} {'Variação %':<15}")
            print('-' * 50)

            variacoes_ibov = []
            precos_ibov_lista = []

            if ultimos_252_ibov is not None and not ultimos_252_ibov.empty:
                for i in range(len(ultimos_252_ibov)):
                    preco_atual = extrair_preco(ultimos_252_ibov.iloc[i])
                    if preco_atual is None:
                        continue

                    precos_ibov_lista.append(preco_atual)

                    if i > 0 and precos_ibov_lista[i-1] != 0:
                        variacao_pct = ((preco_atual - precos_ibov_lista[i - 1]) / precos_ibov_lista[i - 1]) * 100
                        variacao_str = f"{variacao_pct:+.2f}%"
                    else:
                        variacao_str = "N/A (primeiro)"

                    variacoes_ibov.append(variacao_str)

                for i in range(len(ultimos_252_ibov) - 1, -1, -1):
                    data_idx = ultimos_252_ibov.index[i]
                    data_formatada = pd.Timestamp(data_idx).strftime('%d/%m/%Y')
                    if i < len(variacoes_ibov):
                        print(f"{data_formatada}  {variacoes_ibov[i]:>14}")
            else:
                print("✗ Não consegui buscar dados de variação do Ibovespa para este período.")

            # ===== MÉDIAS DAS VARIAÇÕES PERCENTUAIS =====
            print('\n' + '='*60)
            print('RESUMO - MÉDIAS DAS VARIAÇÕES PERCENTUAIS:')
            print('='*60)

            def media_variacoes(lista_variacoes):
                valores = []
                for s in lista_variacoes:
                    if isinstance(s, str) and 'N/A' in s:
                        continue
                    try:
                        valores.append(float(s.rstrip('%')))
                    except Exception:
                        continue
                return sum(valores) / len(valores) if valores else 0.0

            media_acao = media_variacoes(variacoes)
            media_ibov = media_variacoes(variacoes_ibov)

            print(f"\nMédia de variação diária - {ticker}: {media_acao:+.4f}%")
            print(f"Média de variação diária - Ibovespa: {media_ibov:+.4f}%")
            print(f"Diferença (Ação - Ibovespa): {(media_acao - media_ibov):+.4f}%")
            print('='*60)

            # ===== GRÁFICOS =====
            datas = [pd.Timestamp(idx).strftime('%d/%m') for idx in ultimos_252.index]
            precos = [p for p in precos_lista]

            # variações diárias (float) — mantém 0 para o primeiro
            variacoes_float = []
            for i in range(len(precos)):
                if i == 0 or precos[i-1] == 0 or np.isnan(precos[i]) or np.isnan(precos[i-1]):
                    variacoes_float.append(0)
                else:
                    variacoes_float.append(((precos[i] - precos[i-1]) / precos[i-1]) * 100)

            var_pct_acumulada_acao = []
            for i in range(len(precos)):
                if i == 0 or np.isnan(precos[0]) or precos[0] == 0:
                    var_pct_acumulada_acao.append(0)
                else:
                    var_pct_acumulada_acao.append(((precos[i] - precos[0]) / precos[0]) * 100)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            fig.suptitle(f'Análise de {ticker} vs Ibovespa - Últimos {n} Pregões', fontsize=16, fontweight='bold')

            ax1.plot(range(len(var_pct_acumulada_acao)), var_pct_acumulada_acao, marker='o', linestyle='-')
            ax1.set_title('Variação Percentual Acumulada - Ação')
            ax1.set_xlabel('Dias (do mais antigo para o mais recente)')
            ax1.set_ylabel('Variação (%)')
            ax1.axhline(y=0, linestyle='--', linewidth=0.8, alpha=0.5)
            ax1.grid(True, alpha=0.3)

            ax2.bar(range(len(variacoes_float)), variacoes_float, alpha=0.7, width=0.8)
            ax2.set_title(f'Variação Percentual Diária (%) - {ticker}')
            ax2.set_xlabel('Dias (do mais antigo para o mais recente)')
            ax2.set_ylabel('Variação (%)')
            ax2.axhline(y=0, linestyle='-', linewidth=0.8)
            ax2.axhline(y=media_acao, linestyle='--', linewidth=1.2, alpha=0.7, label=f'Média: {media_acao:.4f}%')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.legend(loc='best', fontsize=10)

            plt.tight_layout()
            plt.show()

        except KeyboardInterrupt:
            print("\nOperação cancelada pelo usuário.")
            break
        except Exception as e:
            print(f"Erro inesperado: {e}")
            # não quebra automaticamente; permite nova tentativa
        # pergunta se deseja nova consulta
        try:
            nova_consulta = input("\nDeseja fazer uma nova consulta? (s/n): ").strip().lower()
            if nova_consulta != 's':
                print("Encerrando programa...")
                break
        except KeyboardInterrupt:
            print("\nOperação cancelada pelo usuário.")
            break

if __name__ == "__main__":
    main()
