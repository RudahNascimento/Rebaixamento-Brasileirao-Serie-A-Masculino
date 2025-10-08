import pandas as pd
from src import data_processing
from src import eda
from src import modelos

def main():

    # 1) Gera dataset final até a rodada 19
    rodada=19
    dados_gerais = data_processing.criar_tabela_historica(rodada=rodada)

    # 2) Identificação dos Outliers
    eda.identifica_outliers(dados_gerais)

    # 3) Limpeza de outliers específicos
    dados_gerais.drop([18, 45, 46, 60, 118, 170, 206], inplace=True)

    dados_gerais.reset_index(drop=True, inplace=True)

    # 4) Formatação dataseat para modelos preditivos
    dados_modelo = data_processing.tabela_modelo_format(dados_gerais)

    # 5) Separa o X e y para os treinos e testes
    X = dados_modelo.drop(columns='Situação')
    y = dados_modelo['Situação']

    # 6) Roda o modelo de Random Forest
    metricas_rf, parametros_rf, modelo_rf = modelos.random_forest(X,y)

    # 7) Roda o modelo de Gradient Boost
    metricas_gb, parametros_gb, modelo_gb = modelos.gradient_boosting(X,y)

    # 8) Roda o modelo de Regressão Logística
    metricas_rl, parametros_rl, modelo_rl = modelos.logistic_regression(X,y)

    # 9) Faz a previsão de rebaixamento ou não do time escolhido

    novo_time = pd.DataFrame({
        'V': [7],
        'D': [9],
        'E': [3],
        'GF': [22],
        'GT': [24],
        'Permanecimento': [0]
    })

    modelos.previsao(modelo_rf, novo_time)

if __name__ == "__main__":
    main()