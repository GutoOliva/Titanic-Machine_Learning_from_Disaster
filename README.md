# Titanic: Machine Learning from Disaster  
## Previsão de Sobrevivência no Titanic

---

### **Sobre o Projeto | About the Project**
O objetivo deste projeto é analisar e prever as chances de sobrevivência dos passageiros do Titanic com base em variáveis como gênero, idade e classe social.  
The goal of this project is to analyze and predict passengers’ survival chances on the Titanic based on variables such as gender, age, and class.

A tragédia do Titanic, ocorrida em 15 de abril de 1912, resultou na morte de 1.502 pessoas de um total de 2.224 passageiros. Com dados reais disponíveis, é possível realizar análises exploratórias, testar hipóteses e construir modelos preditivos para estimar as chances de sobrevivência.  
The Titanic disaster on April 15, 1912, killed 1,502 of 2,224 passengers. With real data available, it is possible to perform exploratory analyses, test hypotheses, and build predictive models to estimate survival chances.

---

### **Etapas do Projeto | Project Steps**
1. **Definição do Problema | Problem Definition**  
   Identificar o objetivo do modelo: prever se um passageiro sobreviveu ou não.  

2. **Coleta dos Dados | Data Collection**  
   Uso dos arquivos `train.csv` e `test.csv` disponibilizados pela competição do Kaggle.  

3. **Análise Exploratória | Data Exploration**  
   Verificação de tipos de variáveis, valores ausentes, outliers e padrões nos dados.  

4. **Preparação dos Dados | Data Preparation**  
   Tratamento de valores nulos, transformação de variáveis categóricas e criação de variáveis dummy.  

5. **Construção do Modelo | Model Building**  
   Modelos utilizados: Regressão Logística e Árvore de Decisão.  

6. **Avaliação | Evaluation**  
   Métrica de desempenho: **Acurácia** (accuracy).  

---

### **Descrição dos Dados | Data Description**
| Variável | Descrição |
|-----------|------------|
| **PassengerId** | Identificação do passageiro |
| **Survived** | 0 = Não sobreviveu, 1 = Sobreviveu |
| **Pclass** | Classe do ticket (1ª, 2ª, 3ª) |
| **Name** | Nome do passageiro |
| **Sex** | Sexo do passageiro |
| **Age** | Idade do passageiro |
| **SibSp** | Número de irmãos/cônjuges a bordo |
| **Parch** | Número de pais/filhos a bordo |
| **Ticket** | Número do ticket |
| **Fare** | Valor pago pela passagem |
| **Cabin** | Número da cabine |
| **Embarked** | Porto de embarque (C, Q, S) |

---

### **Principais Insights | Key Insights**
- **Mulheres tiveram 74% de chance média de sobrevivência**, enquanto homens apenas 19%.  
  Women had a 74% average survival rate, while men only 19%.  
- Passageiros da **1ª classe** sobreviveram mais do que os da 3ª.  
  1st-class passengers survived more often than 3rd-class ones.  
- As variáveis **idade, classe e sexo** tiveram maior influência no modelo.  
  Age, class, and gender were the most influential variables.

---

### **Tecnologias Utilizadas | Technologies**
Python, Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn.

---

### **Modelos e Desempenho | Models and Performance**
| Modelo | Acurácia | Descrição |
|---------|-----------|------------|
| **Regressão Logística** | 80.13% | Modelo simples e interpretável. |
| **Árvore de Decisão** | 82.72% | Melhor desempenho, maior explicabilidade. |

---

### **Resultados e Conclusões | Results and Conclusions**
- O modelo baseado em Árvore de Decisão apresentou o melhor desempenho.  
- Fatores como **classe social e gênero** tiveram grande peso na predição.  
- Segundo o modelo, **Julia sobreviveria, mas Augusto não**, replicando o enredo do filme Titanic.
