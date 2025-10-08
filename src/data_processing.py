import pandas as pd
import numpy as np

def carregar_dados(jogos_csv:str)->pd.DataFrame:
    """

    Lê o arquivo CSV do brasileirão original e faz os tratamentos necessários.
    """

    df = pd.read_csv(jogos_csv)


    # Transformação da coluna datas para datetime
    df['data'] = pd.to_datetime(df['data'], format="%d/%m/%Y")

    # Corrigir jogos de 2020 que aconteceram em 2021 (pandemia)
    df.loc[(df['data'] > '2020-12-31') & (df['data'] < '2021-02-26'), 'data'] = '2020-06-01'

    # Retirando colunas que não serão utilizadas
    df['data'] = df['data'].dt.year

    df = df.drop(
        columns=['formacao_mandante', 'hora', 'formacao_visitante', 'tecnico_mandante', 'tecnico_visitante', 'arena',
                 'mandante_Estado', 'visitante_Estado'])

    df.rename(columns={'rodata': 'rodada'}, inplace=True)

    return df

def criar_tabela_temporada(df: pd.DataFrame, temporada: int, rodada: int, rebaixados:list, tabela_ano_anterior: pd.DataFrame = None) -> pd.DataFrame:
    """

    :param df:
        Dataframe com os jogos completos do Brasileirão de pontos corridos
    :param temporada:
        Ano da temporada
    :param rodada:
        Rodada que quer ser realizada a predição
    :param rebaixados:
        Times rebaixados no final da temporada
    :param tabela_ano_anterior:
        Tabela do temporada do ano anterior
    :return:
        Dataframe com a tabela da temporada até a rodada escolhida, com os dados de
        Vitórias(V), Derrotas(D), Empates(E), Gols Feitos(GF), Gols Tomados(GT), se
        o time veio da serie B ou não(Subiu, com 1 caso tenha subido e 0 caso não)
        e se foi rebaixado ou não ao final da temporada(Situação, com F se ficou
        ou R se foi rebaixado)
    """

    # Cria a Tabela
    dados_ano = df[df['data'] == temporada]
    times = dados_ano['mandante'].unique()

    tabela = pd.DataFrame({
        'Time': times,
        'Temporada': temporada,
        'V': 0,
        'D': 0,
        'E': 0,
        'GF': 0,
        'GT': 0,
        'Subiu': 0,
        'Situação': 'F'
    })

    # Rotula os times rebaixados na temporada
    tabela.loc[tabela['Time'].isin(rebaixados), 'Situação'] = 'R'

    # Marcar os promovidos da segunda divisão do ano anterior
    if tabela_ano_anterior is not None:
        times_anteriores = tabela_ano_anterior['Time'].values
        times_novos = np.setdiff1d(times, times_anteriores)
        for time in times_novos:
            tabela.loc[tabela['Time'] == time, 'Subiu'] += 1

    # Preenche a tabela com as estatísticas até a rodada escolhida
    for _, jogo in dados_ano[
        dados_ano['rodada'] <= rodada].iterrows():

        tabela.loc[tabela['Time'] == jogo['mandante'], 'GF'] += jogo['mandante_Placar']
        tabela.loc[tabela['Time'] == jogo['visitante'], 'GF'] += jogo['visitante_Placar']

        tabela.loc[tabela['Time'] == jogo['visitante'], 'GT'] += jogo['mandante_Placar']
        tabela.loc[tabela['Time'] == jogo['mandante'], 'GT'] += jogo['visitante_Placar']

        if jogo['mandante_Placar'] > jogo['visitante_Placar']:

            tabela.loc[tabela['Time'] == jogo['mandante'], 'V'] += 1
            tabela.loc[tabela['Time'] == jogo['visitante'], 'D'] += 1

        elif jogo['mandante_Placar'] < jogo['visitante_Placar']:

            tabela.loc[tabela['Time'] == jogo['mandante'], 'D'] += 1
            tabela.loc[tabela['Time'] == jogo['visitante'], 'V'] += 1

        else:

            tabela.loc[tabela['Time'] == jogo['mandante'], 'E'] += 1
            tabela.loc[tabela['Time'] == jogo['visitante'], 'E'] += 1

    return tabela

def criar_tabela_historica(rodada: int) -> pd.DataFrame:
    """

    Calcula o tempo de permanência do time na Série A, até no máximo 5 anos, e retorna a
    tabela geral com todos os times de 2003 até 2024 até a rodada selecionada.
    """

    # Faz o dataframe de partidas geral
    caminho = "data/campeonato-brasileiro-full.csv"
    df = carregar_dados(caminho)

    # Lista de times rebaixados por temporada
    rebaixados_por_ano = {
        2003: ['Paysandu', 'Fortaleza', 'Bahia', 'Ponte Preta'],
        2004: ['Criciuma', 'Guarani', 'Vitoria', 'Gremio'],
        2005: ['Brasiliense', 'Paysandu', 'Coritiba', 'Atletico-MG'],
        2006: ['Santa Cruz', 'Sao Caetano', 'Fortaleza', 'Ponte Preta'],
        2007: ['Parana', 'Corinthians', 'Juventude', 'America-RN'],
        2008: ['Ipatinga', 'Portuguesa', 'Vasco', 'Figueirense'],
        2009: ['Coritiba', 'Santo Andre', 'Nautico', 'Sport'],
        2010: ['Gremio Prudente', 'Goias', 'Guarani', 'Vitoria'],
        2011: ['Avai', 'America-MG', 'Ceara', 'Athletico-PR'],
        2012: ['Sport', 'Palmeiras', 'Atletico-GO', 'Figueirense'],
        2013: ['Portuguesa', 'Vasco', 'Ponte Preta', 'Nautico'],
        2014: ['Vitoria', 'Botafogo-RJ', 'Criciuma', 'Bahia'],
        2015: ['Avai', 'Vasco', 'Goias', 'Joinville'],
        2016: ['Internacional', 'Figueirense', 'Santa Cruz', 'America-MG'],
        2017: ['Coritiba', 'Avai', 'Ponte Preta', 'Atletico-GO'],
        2018: ['Parana', 'Vitoria', 'America-MG', 'Sport'],
        2019: ['Cruzeiro', 'CSA', 'Chapecoense', 'Avai'],
        2020: ['Vasco', 'Botafogo-RJ', 'Coritiba', 'Goias'],
        2021: ['Gremio', 'Bahia', 'Sport', 'Chapecoense'],
        2022: ['Ceara', 'Atletico-GO', 'Avai', 'Juventude'],
        2023: ['Santos', 'Goias', 'Coritiba', 'America-MG'],
        2024: ['Cuiaba', 'Atletico-GO', 'Criciuma', 'Athletico-PR']

        }

    # Promovidos apenas da primeira temporada (2003)
    promovidos_2003 = ['Criciuma', 'Fortaleza']


    tabela_ano_anterior = None
    tabelas = {}

    for ano in range(2003, 2025):  # de 2003 até 2024
        rebaixados = rebaixados_por_ano.get(ano, [])

        if ano == 2003:
            # Primeira temporada: precisa marcar manualmente quem subiu
            tabela_atual = criar_tabela_temporada(df, temporada=ano, rodada=rodada,
                                                  rebaixados=rebaixados)
            for time in promovidos_2003:
                tabela_atual.loc[tabela_atual['Time'] == time, 'Subiu'] += 1
        else:
            # A partir de 2004: promovidos calculados automaticamente
            tabela_atual = criar_tabela_temporada(df, temporada=ano, rodada=rodada,
                                                  rebaixados=rebaixados,
                                                  tabela_ano_anterior=tabela_ano_anterior)

        # Guarda tabela atual para utilizar na próxima temporada
        tabela_ano_anterior = tabela_atual
        tabelas[ano] = tabela_atual

        # Mostra um preview das tabelas
        print(f"Tabela da temporada {ano}")
        print(tabela_atual.head(20), "\n")

        # Salva os arquivos das tabelas até a rodada escolhida em csv
        tabela_atual.to_csv(f"data/tabela_{ano}_rodada{rodada}.csv", index=False)

    # Concatena todas as temporadas
    dados_gerais = pd.concat(list(tabelas.values()), ignore_index=True)

    # Colocando informação sobre tempo de estadia na Série A
    dados_gerais['Permanecimento'] = 0

    dados_gerais.loc[(dados_gerais['Temporada'] == 2003) & (dados_gerais['Time'] == 'Paysandu'), 'Permanecimento'] += 1
    dados_gerais.loc[
        (dados_gerais['Temporada'] == 2003) & (dados_gerais['Time'] == 'Figueirense'), 'Permanecimento'] += 1
    dados_gerais.loc[(dados_gerais['Temporada'] == 2003) & (dados_gerais['Time'] == 'Parana'), 'Permanecimento'] += 2
    dados_gerais.loc[
        (dados_gerais['Temporada'] == 2003) & (dados_gerais['Time'] == 'Sao Caetano'), 'Permanecimento'] += 2
    dados_gerais.loc[(dados_gerais['Temporada'] == 2003) & (dados_gerais['Time'] == 'Goias'), 'Permanecimento'] += 3

    dados_gerais.loc[
        (dados_gerais['Temporada'] == 2003) &
        (dados_gerais['Subiu'] == 0) &
        (dados_gerais['Permanecimento'] == 0),
        'Permanecimento'
    ] += 5

    for idx, time in dados_gerais.iterrows():
        if (time['Situação'] == 'F') & (dados_gerais.loc[idx, 'Permanecimento'] == 0):
            x = time['Temporada'] + 1
            index = dados_gerais[(dados_gerais['Temporada'] == x) & (dados_gerais['Time'] == time['Time'])].index
            dados_gerais.loc[index, 'Permanecimento'] = 1

        elif (time['Situação'] == 'F') & (dados_gerais.loc[idx, 'Permanecimento'] == 1):
            x = time['Temporada'] + 1
            index = dados_gerais[(dados_gerais['Temporada'] == x) & (dados_gerais['Time'] == time['Time'])].index
            dados_gerais.loc[index, 'Permanecimento'] = 2

        elif (time['Situação'] == 'F') & (dados_gerais.loc[idx, 'Permanecimento'] == 2):
            x = time['Temporada'] + 1
            index = dados_gerais[(dados_gerais['Temporada'] == x) & (dados_gerais['Time'] == time['Time'])].index
            dados_gerais.loc[index, 'Permanecimento'] = 3

        elif (time['Situação'] == 'F') & (dados_gerais.loc[idx, 'Permanecimento'] == 3):
            x = time['Temporada'] + 1
            index = dados_gerais[(dados_gerais['Temporada'] == x) & (dados_gerais['Time'] == time['Time'])].index
            dados_gerais.loc[index, 'Permanecimento'] = 4

        elif (time['Situação'] == 'F') & (dados_gerais.loc[idx, 'Permanecimento'] == 4):
            x = time['Temporada'] + 1
            index = dados_gerais[(dados_gerais['Temporada'] == x) & (dados_gerais['Time'] == time['Time'])].index
            dados_gerais.loc[index, 'Permanecimento'] = 5

        elif (time['Situação'] == 'F') & (dados_gerais.loc[idx, 'Permanecimento'] == 5):
            x = time['Temporada'] + 1
            index = dados_gerais[(dados_gerais['Temporada'] == x) & (dados_gerais['Time'] == time['Time'])].index
            dados_gerais.loc[index, 'Permanecimento'] = 5

    # Salva o arquivo final
    dados_gerais.to_csv(f"data/dados_gerais_temporada{rodada}.csv", index=False)
    print(f"✅ Dados finais salvos em data/dados_gerais_temporada{rodada}.csv")

    return dados_gerais

def tabela_modelo_format(dados_gerais: pd.DataFrame) -> pd.DataFrame:
    """

    Formata o dadaset para ser utilizado nos modelos preditivos.
    """

    # Transformação de F e R para 1 e 0
    dados_gerais['Situação'] = dados_gerais['Situação'].map({'F': 1, 'R': 0})

    # Remove colunas não utilizadas e posiciona 'Situação' por último
    dados_gerais.drop(columns=['Subiu', 'Temporada', 'Time'], inplace=True)
    coluna = dados_gerais.pop('Situação')
    dados_gerais['Situação'] = coluna

    return dados_gerais


