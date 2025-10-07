import pandas as pd

def identifica_outliers(dados_gerais: pd.DataFrame):
    """

    Identifica os outliers do dataset
    """

    variaveis = ['V', 'D', 'E', 'GF', 'GT']
    outliers = {}

    for var in variaveis:
        outliers[var] = []

        for situacao in dados_gerais['Situação'].unique():
            grupo = dados_gerais[dados_gerais['Situação'] == situacao]

            q1 = grupo[var].quantile(0.25)
            q3 = grupo[var].quantile(0.75)
            iqr = q3 - q1

            limite_inferior = q1 - 1.5 * iqr
            limite_superior = q3 + 1.5 * iqr

            out = grupo[(grupo[var] < limite_inferior) | (grupo[var] > limite_superior)]

            if not out.empty:
                outliers[var].append(out[['Time', 'Situação', 'Temporada', var]])

    for var, lista in outliers.items():
        if lista:
            print(f"\n🔎 Outliers em {var}:")
            print(pd.concat(lista))

