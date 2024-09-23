import cupy as cp
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from io import StringIO

# Definindo cores em tons mais escuros
colors = ['#5F4B32', '#4A4A4A', '#385E38', '#3D444B', '#5C5448', '#967B4F',
          '#A08665', '#7D6648', '#664C4C', '#935A3A', '#B8977E', '#88705A']

# Dados Sintéticos Realistas (Binance) - Adaptado para leitura de CSV
binance_csv = """Altcoin,Preço (USD),Volume (24h),Market Cap (USD),Variação (24h)
Binance Coin (BNB),215,400000000,34000000000,1.2
Cardano (ADA),0.27,180000000,8500000000,-1.5
Solana (SOL),20.5,350000000,7000000000,2.1
Dogecoin (DOGE),0.062,200000000,8500000000,0.5
Polkadot (DOT),4.25,150000000,7800000000,-0.2
Shiba Inu (SHIB),0.000008,100000000,4500000000,3.0
Polygon (MATIC),0.65,80000000,5500000000,-0.7
Avalanche (AVAX),11.2,120000000,4000000000,1.8"""

df_altcoins_binance = pd.read_csv(StringIO(binance_csv))

# Dados (GitHub)
github_csv = """Altcoin,Número de Commits (último mês),Número de Issues Abertas,Número de Pull Requests (último mês),Atividade da Comunidade (escala 1-10)
Cosmos (ATOM),250,35,80,9
Polkadot (DOT),200,40,70,8
Filecoin (FIL),120,20,40,7
Avalanche (AVAX),180,30,60,8.5
Chainlink (LINK),150,25,50,7.5
Solana (SOL),300,50,100,9.5
Cardano (ADA),220,45,85,8.5
Polygon (MATIC),280,55,90,9
Algorand (ALGO),170,32,65,7.8"""

df_altcoins_github = pd.read_csv(StringIO(github_csv))

# Função para plotar gráfico de barras
def plot_bar_chart(df, x_col, y_col, title, color=colors[0]):
    fig = px.bar(df, x=x_col, y=y_col, title=title,
                 color_discrete_sequence=[color])
    fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white')
    fig.show()

# Função para plotar gráfico de pizza
def plot_pie_chart(df, values, names, title):
    fig = px.pie(df, values=values, names=names, title=title,
                 color_discrete_sequence=colors)
    fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white')
    fig.show()

# Função para plotar boxplot
def plot_boxplot(df, x_col, y_col, title, color=colors[0]):
    fig = px.box(df, x=x_col, y=y_col, title=title,
                 color_discrete_sequence=[color])
    fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white')
    fig.show()

# Função para plotar gráfico de densidade
def plot_density_chart(df, x_col, title, color=colors[0]):
    fig = px.histogram(df, x=x_col, title=title, marginal="rug",
                     color_discrete_sequence=[color], nbins=30)
    fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white',
                      xaxis_title=x_col, yaxis_title="Densidade")
    fig.show()

# Função para plotar matriz de correlação
def plot_correlation_matrix(df, title):
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    fig = go.Figure(data=go.Heatmap(z=corr_matrix.values,
                                   x=corr_matrix.index,
                                   y=corr_matrix.columns,
                                   colorscale='Viridis'))
    fig.update_layout(title=title,
                      xaxis_nticks=36,
                      plot_bgcolor='black', paper_bgcolor='black', font_color='white')
    fig.show()

# --- Visualizações dos Dados ---

# --- Binance ---

# 1. Gráfico de Barras: Preço das Altcoins
plot_bar_chart(df_altcoins_binance, 'Altcoin', 'Preço (USD)', 'Preço das Altcoins (USD)', color=colors[1])

# 2. Gráfico de Pizza: Market Cap das Altcoins
plot_pie_chart(df_altcoins_binance, 'Market Cap (USD)', 'Altcoin', 'Market Cap das Altcoins')

# 3. Boxplot: Distribuição do Volume (24h)
plot_boxplot(df_altcoins_binance, 'Altcoin', 'Volume (24h)', 'Distribuição do Volume (24h)', color=colors[2])

# 4. Gráfico de Densidade: Distribuição do Preço
plot_density_chart(df_altcoins_binance, 'Preço (USD)', 'Distribuição do Preço das Altcoins', color=colors[3])

# 5. Gráfico de Densidade: Distribuição do Volume (24h)
plot_density_chart(df_altcoins_binance, 'Volume (24h)', 'Distribuição do Volume (24h) das Altcoins', color=colors[4])

# 6. Boxplot: Variação (24h) por Altcoin
plot_boxplot(df_altcoins_binance, 'Altcoin', 'Variação (24h)', 'Variação (24h) por Altcoin', color=colors[5])


# --- GitHub ---

# 7. Gráfico de Barras: Número de Commits por Altcoin
plot_bar_chart(df_altcoins_github, 'Altcoin', 'Número de Commits (último mês)', 'Número de Commits por Altcoin', color=colors[6])

# 8. Boxplot: Número de Issues Abertas por Altcoin
plot_boxplot(df_altcoins_github, 'Altcoin', 'Número de Issues Abertas', 'Número de Issues Abertas por Altcoin', color=colors[7])

# 9. Gráfico de Densidade: Número de Commits
plot_density_chart(df_altcoins_github, 'Número de Commits (último mês)', 'Distribuição do Número de Commits', color=colors[8])

# 10. Gráfico de Densidade: Atividade da Comunidade
plot_density_chart(df_altcoins_github, 'Atividade da Comunidade (escala 1-10)', 'Distribuição da Atividade da Comunidade', color=colors[9])

# 11. Boxplot: Número de Pull Requests por Altcoin
plot_boxplot(df_altcoins_github, 'Altcoin', 'Número de Pull Requests (último mês)', 'Número de Pull Requests por Altcoin', color=colors[10])

# 12. Matriz de Correlação: Dados do GitHub
plot_correlation_matrix(df_altcoins_github, 'Matriz de Correlação - Dados GitHub')