# Análise de Dados da Série A do Campeonato Brasileiro de Futebol Masculino (2003 - 2024)
Este projeto realiza uma análise histórica do Campeonato Brasileiro de Futebol Masculino, identificando padrões de rebaixamento e implementando modelos preditivos para prever se um time será rebaixado no final da temporada.

---

# Fontes
- A parte de modelagem foi baseada no paper "Predicting The Dutch Football Competition Using Public Data: A Machine Learning Approach", de Niek Tax e Yme Joustra.
- Foi utilizado o arquivo "campeonato-brasileiro-full.csv", presente no repositório https://github.com/raulrosapacheco/BrasileiraoAnalise.git, que concatena todos os jogos que ocorreram no campeonato brasileiro de futebol masculino nas temporadas de 2003 até 2024.

---
# Tecnologias Utilizadas
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Machine Learning
- Hyperparameter Tuning
- SMOTE

---
# Modelos Utilizados
- Random Forest
- Gradient Boosting
- Regressão Logística

---

# Fluxo do Projeto

0 - Todas as bibliotecas utilizadas no projeto estão listadas no arquivo **requirements.txt**.

As primeiras funções utilizadas estão concentradas no arquivo **data_processing.py**, explicitando suas funcionalidades de forma sequencial:  

**1 -** Os dados com todas as partidas concatenadas são lidos e ajustados pela função **carregar_dados**.  

**2 -** A função **criar_tabela_temporada** recebe os dados ajustados e cria a tabela específica da temporada até a rodada que quer ser feita a previsão.  

**3 -** A função **criar_tabela_historica** concatena as tabelas geradas pela função anterior e cria e preenche a coluna "Permanecimento".   referente à quantidade de anos de estadia do time na Série A. Ela então salva os dados de temporadas específicas como "tabela_TEMPORADA_rodadaRODADA.csv" e os dados concatenados como "data/dados_gerais_temporadaRODADA.csv", todos na pasta data.  

Após isso, no arquivo **eda.py**, temos: 

**4 -** A função **identifica_outliers** analisa os dados concatenados e devolve os outliers, tomando como base o padrão de Tukey.  

Voltando para o arquivo **data_processing.py**, temos:  

**5 -** A função **tabela_modelo_format** formata os dados para serem utilizados nos modelos preditivos.  

Finalmente, no arquivo **modelos.py**:  

**6 -** A função **random_forest** faz o hyperparameter tuning, equilibra a distribuição da classe minoritária "rebaixados" utilizando o SMOTE, tudo dentro de uma pipeline para evitar data leak, e constrói o modelo de Random Forest, retornando os parâmetros do hyperparameter tuning e as métricas de precisão, recall e F1, referentes à classe "rebaixados", e acurácia geral do modelo. 

**7 -** A função **gradient_boosting** faz o mesmo para o modelo de Gradient Boosting.  

**8 -** A função **logistic_regression** faz o mesmo para o modelo de Regressão Logística.  

Obs: No passo 3, é citado que a função **criar_tabela_historica** cria e preenche uma coluna "Permanecimento" referente à quantidade de anos de estadia do time na Série A. Isso foi um feature adicionado no modelo a partir da análise presente no arquivo **analise_exploratória_rebaixamento.ipynb**, da pasta notebook, onde foi identificado que 75.31% dos times que sobem para a Série A, caem em até 5 temporadas. O time ao ser promovido possui o valor de 0 na feature na sua primeira temporada na primeira divisão, sendo incrementado o valor em 1 à cada temporada que o time não cai, chegando ao valor máximo de 5.

---

# Exemplo Aplicado
Dentro do arquivo **main.py** , foi estruturado um exemplo de utilização do projeto:

Vamos supor que o usuário seja torcedor de um time recém promovido da Série B do campeonato brasileiro de futebol masculino chamado Santos, e que ele supostamente possua um temor intenso de que seu time caia novamente. Estando o campeonato na **19º rodada**, a tabela do seu suposto time é de **7 Vitórias**, **3 Empates**, **9 Derrotas**, **22 Gols Feitos** e **24 Gols Tomados**. 

Ele então roda a função **criar_tabela_historica**, determinando a rodada como 19. Após isso, ele roda a função **identifica_outliers**, e identifica que existem alguns outliers extremamente raros na sua sua análise, partindo então para uma análise exploratória dos dados dentro do Notebook do arquivo **analise_exploratória_rebaixamento.ipynb**. Nele, o suposto torcedor do Santos verifica que realmente 75.31% dos times que sobem da segunda divisão, são rebaixados em até 5 anos, justificando a feature de "Permanecimento". Após isso, produz os gráficos BoxPlot, identificando agora visualmente que alguns outliers dos dados distoam de forma extremamente drástica do dataset em geral, escolhendo então eliminá-los da análise. Finalmente, ele plota os gráficos de dispersão dos dados em relação às classes e verifica que o comportamento das Features de Vitórias, Derrotas, Gols Feitos e Gols Sofridos visualmente se encaixam no modelo de curva sinuosa, justificando a utilização por exemplo do modelo de regressão logística.

Ele então formata os dados para serem utilizados nos modelos, e roda as 3 funções preditivas, tendo como saída de cada uma o seguinte:

Métricas Médias (Random Forest):
precisao_rebaixado: 0.59
recal_rebaixado: 0.79
f1_rebaixado: 0.67
acuracia: 0.85

Métricas Médias (Gradient Boosting):
precisao_rebaixado: 0.59
recal_rebaixado: 0.69
f1_rebaixado: 0.63
acuracia: 0.85

Métricas Médias (Regressão Logística):
precisao_rebaixado: 0.50
recal_rebaixado: 0.79
f1_rebaixado: 0.61
acuracia: 0.81

O suposto torcedor sabe da natureza aleatória do campeonato brasileiro, onde times podem fazer campanhas de rebaixado mas que os colocariam na parte superior da tabela em outros anos. O modelo utilizado como base do projeto foi aplicado no Campeonato Holandês de Futebol Masculino, notoriamente menos aleatório que o Campeonato Brasileiro, e mesmo assim a maior acurácia atingida foi de 0.54 (importante citar que o paper procurava prever resultado de jogos, uma finalidade muito mais complexa de se atingir, mas a natureza aleatória compartilhada entre os sistemas analisados continua evidenciada). Mesmo a modelagem se utilizando de várias tecnologías e algoritmos que equilibram mais os dados, o máximo de acurácia que é conseguida é de 0.85, tendo as melhores métricas sobre a identificação da classe "Rebaixados" no modelo de Random Forest, ainda sendo bastante aquém de um modelo ótimo (com acurácia de 0.97 ou superior). 

Mas o suposto desesperado torcedor, se apegando no mínimo de certeza dentro do caos, utiliza então o modelo com as melhores métricas, o de Random Forest, e obtém o resultado de que seu suposto time Santos, pelo menos dessa vez, na 19º, no ano de 2025, na Série A do Campeonato Brasileiro de Futebol Masculino, com 59% de chance, não será rebaixado!

---

# Sugestão Para Projetos Futuros
O projeto possui os dados entre os anos de 2003, onde foi iniciado o modelo de pontos corridos do campeonato, até 2024, último campeonato finalizado até a data. Para ser realizado uma previsão em anos seguintes, o arquivo **campeonato-brasileiro-full**, dentro da pasta data, deve ser atualizado com os jogos dos subsequentes campeonatos, dentro da função **criar_tabela_historica**, devem ser adicionados manualmente na lista **rebaixados_por_ano** os seus times que foram rebaixados e no loop **"for ano in range(2003, 2025):"**, deve ser trocado o segundo termo para o ano adequado.

Existem diversos algorítmos diferentes de machine learning que possivelmente gerariam resultados diferentes que os obtidos nesse projeto, encorajo a serem testados.

Dentro da função **criar_tabela_temporada**, ao ser retirado o termo **"['rodada'] <= rodada]"** no loop **"for _, jogo in dados_ano[dados_ano['rodada'] <= rodada].iterrows():"**, são geradas as tabelas de todos os campeonatos finalizados, sendo possível então realizar uma análise exploratória da história da Série A do Campeonato Masculino de Futebol como um todo, encorajo a ser realizado.

---

# Contatos do Autor
Para possíveis dúvidas, críticas ou sugestões sobre o presente projeto.

E-mail: rud.ah765@gmail.com

Linkedin: www.linkedin.com/in/rudah-lages-05216920a
