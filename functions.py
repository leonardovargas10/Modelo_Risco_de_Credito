## Bibliotecas De Manipulação de Dados e Visualização
import pandas as pd 
import builtins as builtins
import matplotlib.pyplot as plt
import seaborn as sns 
from IPython.display import display, Image
from tabulate import tabulate

## Bibliotecas de Modelagem Matemática e Estatística
import numpy as np
import scipy as sp 
import scipy.stats as stats
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import normaltest, ttest_ind, ttest_rel, mannwhitneyu, wilcoxon, kruskal, uniform, chi2_contingency
from statsmodels.stats.weightstats import ztest
from statsmodels.stats.diagnostic import lilliefors

# Bibliotecas de Seleção de Modelos
from skopt import BayesSearchCV
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate, cross_val_predict
from sklearn.feature_selection import VarianceThreshold, chi2, mutual_info_classif

# Bibliotecas de Pré-Processamento e Pipeline
from category_encoders import BinaryEncoder
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer

# Bibliotecas de Modelos de Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# Bibliotecas de Métricas de Machine Learning
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_score, recall_score, f1_score, confusion_matrix, silhouette_score

# Parâmetros de Otimização
import warnings
# %matplotlib inline
# sns.set(style="whitegrid", font_scale=1.2)
# plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['font.size'] = '14'
# plt.rcParams['figure.figsize'] = [10, 5]
# pd.set_option('display.max_rows', 100)
# pd.set_option('display.max_columns', 100)
# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.expand_frame_repr', False)
# pd.set_option('display.float_format', lambda x: '%.2f' % x) # Tira os números do formato de Notação Científica
# np.set_printoptions(suppress=True) # Tira os números do formato de Notação Científica em Numpy Arrays
# warnings.filterwarnings('ignore')
# warnings.simplefilter(action='ignore', category=FutureWarning) # Retira Future Warnings
def plota_barras_agrupadas(df, x, y, titulo):
    ax = sns.barplot(data = df, x = x, y = y)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                    textcoords='offset points')
    ax.set_title(f'{titulo}')
    ax.set_xlabel(f'{x}', fontsize = 14)
    ax.set_ylabel(f'Quantidade de Inadimplentes', fontsize = 14)
    ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks()], fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), ha='right', fontsize=10)
    plt.show()

def plota_barras(lista_variaveis, hue, df, linhas, colunas, titulo, rotation):
    if hue != False:
        if (linhas == 1) and (colunas == 1):
            k = 0
            ax = sns.countplot(x = lista_variaveis[k], data = df, orient = 'h', hue = hue)
            ax.set_title(f'{titulo}')
            ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
            ax.set_ylabel(f'Quantidade', fontsize = 14)
            total = []
            for bar in ax.patches:
                height = bar.get_height()
                total.append(height)
            total = builtins.sum(total)
            
            sizes = []
            for bar in ax.patches:
                height = bar.get_height()
                sizes.append(height)
                ax.text(bar.get_x() + bar.get_width()/1.6,
                        height,
                        f'{builtins.round((height/total)*100, 2)}%',
                        ha = 'center',
                        fontsize = 12
                )

            ax.set_ylim(0, builtins.max(sizes)*1.1)
            ax.set_xticklabels(df[lista_variaveis[k]].unique(), rotation = rotation, ha='right', fontsize=10)
            # Formatação manual dos rótulos do eixo y para remover a notação científica
            ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks()], fontsize=10)
            # Adicionamos os nomes das categorias no eixo x
            ax.set_xticklabels(ax.get_xticklabels(), ha='right', fontsize=10)
            plt.show()

        elif linhas == 1:
            fig, axis = plt.subplots(linhas, colunas, figsize=(14, 7), sharey=True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    ax = sns.countplot(x = lista_variaveis[k], data = df, orient = 'h', ax = axis[j], hue = hue)
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
                    ax.set_ylabel(f'Quantidade', fontsize = 14)
                    total = []
                    for bar in ax.patches:
                        height = bar.get_height()
                        total.append(height)
                    total = builtins.sum(total)
                    
                    sizes = []
                    for bar in ax.patches:
                        height = bar.get_height()
                        sizes.append(height)
                        ax.text(bar.get_x() + bar.get_width()/1.6,
                                height,
                                f'{builtins.round((height/total)*100, 2)}%',
                                ha = 'center',
                                fontsize = 12
                        )
                    ax.set_ylim(0, builtins.max(sizes)*1.1)
                    ax.set_xticklabels(df[lista_variaveis[k]].unique(), rotation = rotation, ha='right', fontsize=10)
                    # Formatação manual dos rótulos do eixo y para remover a notação científica
                    ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks()], fontsize=10)
                    # Adicionamos os nomes das categorias no eixo x
                    ax.set_xticklabels(ax.get_xticklabels(), ha='right', fontsize=10)
                    k = k + 1
                    
        elif colunas == 1:
            fig, axis = plt.subplots(linhas, colunas, figsize=(14, 7), sharey=True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    ax = sns.countplot(x = lista_variaveis[k], data = df, orient = 'h', ax = axis[i], hue = hue)
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
                    ax.set_ylabel(f'Quantidade', fontsize = 14)
                    total = []
                    for bar in ax.patches:
                        height = bar.get_height()
                        total.append(height)
                    total = builtins.sum(total)
                    
                    sizes = []
                    for bar in ax.patches:
                        height = bar.get_height()
                        sizes.append(height)
                        ax.text(bar.get_x() + bar.get_width()/1.6,
                                height,
                                f'{builtins.round((height/total)*100, 2)}%',
                                ha = 'center',
                                fontsize = 12
                        )
                    ax.set_ylim(0, builtins.max(sizes)*1.1)
                    ax.set_xticklabels(df[lista_variaveis[k]].unique(), rotation = rotation, ha='right', fontsize=10)
                    # Formatação manual dos rótulos do eixo y para remover a notação científica
                    ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks()], fontsize=10)
                    # Adicionamos os nomes das categorias no eixo x
                    ax.set_xticklabels(ax.get_xticklabels(), ha='right', fontsize=10)
                    k = k + 1
            
        else: 
            fig, axis = plt.subplots(linhas, colunas, figsize=(14, 7), sharey=True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    ax = sns.countplot(x = lista_variaveis[k], data = df, orient = 'h', ax = axis[i, j], hue = hue)
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
                    ax.set_ylabel(f'Quantidade', fontsize = 14)
                    total = []
                    for bar in ax.patches:
                        height = bar.get_height()
                        total.append(height)
                    total = builtins.sum(total)
                    
                    sizes = []
                    for bar in ax.patches:
                        height = bar.get_height()
                        sizes.append(height)
                        ax.text(bar.get_x() + bar.get_width()/1.6,
                                height,
                                f'{builtins.round((height/total)*100, 2)}%',
                                ha = 'center',
                                fontsize = 12
                        )
                    ax.set_ylim(0, builtins.max(sizes)*1.1)
                    ax.set_xticklabels(df[lista_variaveis[k]].unique(), rotation = rotation, ha='right', fontsize=10)
                    # Formatação manual dos rótulos do eixo y para remover a notação científica
                    ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks()], fontsize=10)
                    # Adicionamos os nomes das categorias no eixo x
                    ax.set_xticklabels(ax.get_xticklabels(), ha='right', fontsize=10)
                    k = k + 1
           
    else:
        if (linhas == 1) and (colunas == 1):
            k = 0
            ax = sns.countplot(x = lista_variaveis[k], data = df, orient = 'h', color='#1FB3E5')
            ax.set_title(f'{titulo}')
            ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
            ax.set_ylabel(f'Quantidade', fontsize = 14)
            total = []
            for bar in ax.patches:
                height = bar.get_height()
                total.append(height)
            total = builtins.sum(total)
            
            sizes = []
            for bar in ax.patches:
                height = bar.get_height()
                sizes.append(height)
                ax.text(bar.get_x() + bar.get_width()/1.6,
                        height,
                        f'{builtins.round((height/total)*100, 2)}%',
                        ha = 'center',
                        fontsize = 12
                )
            ax.set_ylim(0, builtins.max(sizes)*1.1)
            ax.set_xticklabels(df[lista_variaveis[k]].unique(), rotation = rotation, ha='right', fontsize=10)
            # Formatação manual dos rótulos do eixo y para remover a notação científica
            ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks()], fontsize=10)
            # Adicionamos os nomes das categorias no eixo x
            ax.set_xticklabels(ax.get_xticklabels(), ha='right', fontsize=10)
            plt.show()

        elif linhas == 1:
            fig, axis = plt.subplots(linhas, colunas, figsize=(14, 7), sharey=True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    ax = sns.countplot(x = lista_variaveis[k], data = df, orient = 'h', ax = axis[j], color='#1FB3E5')
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
                    ax.set_ylabel(f'Quantidade', fontsize = 14)
                    total = []
                    for bar in ax.patches:
                        height = bar.get_height()
                        total.append(height)
                    total = builtins.sum(total)
                    
                    sizes = []
                    for bar in ax.patches:
                        height = bar.get_height()
                        sizes.append(height)
                        ax.text(bar.get_x() + bar.get_width()/1.6,
                                height,
                                f'{builtins.round((height/total)*100, 2)}%',
                                ha = 'center',
                                fontsize = 12
                        )
                    ax.set_ylim(0, builtins.max(sizes)*1.1)
                    ax.set_xticklabels(df[lista_variaveis[k]].unique(), rotation = rotation, ha='right', fontsize=10)
                    # Formatação manual dos rótulos do eixo y para remover a notação científica
                    ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks()], fontsize=10)
                    # Adicionamos os nomes das categorias no eixo x
                    ax.set_xticklabels(ax.get_xticklabels(), ha='right', fontsize=10)
                    k = k + 1
            

        elif colunas == 1:
            fig, axis = plt.subplots(linhas, colunas, figsize=(14, 7), sharey=True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    ax = sns.countplot(x = lista_variaveis[k], data = df, orient = 'h', ax = axis[i], color='#1FB3E5')
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
                    ax.set_ylabel(f'Quantidade', fontsize = 14)
                    total = []
                    for bar in ax.patches:
                        height = bar.get_height()
                        total.append(height)
                    total = builtins.sum(total)
                    
                    sizes = []
                    for bar in ax.patches:
                        height = bar.get_height()
                        sizes.append(height)
                        ax.text(bar.get_x() + bar.get_width()/1.6,
                                height,
                                f'{builtins.round((height/total)*100, 2)}%',
                                ha = 'center',
                                fontsize = 12
                        )
                    ax.set_ylim(0, builtins.max(sizes)*1.1)
                    ax.set_xticklabels(df[lista_variaveis[k]].unique(), rotation = rotation, ha='right', fontsize=10)
                    # Formatação manual dos rótulos do eixo y para remover a notação científica
                    ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks()], fontsize=10)
                    # Adicionamos os nomes das categorias no eixo x
                    ax.set_xticklabels(ax.get_xticklabels(), ha='right', fontsize=10)
                    k = k + 1
            

        else:
            fig, axis = plt.subplots(linhas, colunas, figsize=(14, 7), sharey=True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    ax = sns.countplot(x = lista_variaveis[k], data = df, orient = 'h', ax = axis[i, j], color='#1FB3E5')
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
                    ax.set_ylabel(f'Quantidade', fontsize = 14)
                    total = []
                    for bar in ax.patches:
                        height = bar.get_height()
                        total.append(height)
                    total = builtins.sum(total)
                    
                    sizes = []
                    for bar in ax.patches:
                        height = bar.get_height()
                        sizes.append(height)
                        ax.text(bar.get_x() + bar.get_width()/1.6,
                                height,
                                f'{builtins.round((height/total)*100, 2)}%',
                                ha = 'center',
                                fontsize = 12
                        )
                    ax.set_ylim(0, builtins.max(sizes)*1.1)
                    ax.set_xticklabels(df[lista_variaveis[k]].unique(), rotation = rotation, ha='right', fontsize=10)
                    # Formatação manual dos rótulos do eixo y para remover a notação científica
                    ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks()], fontsize=10)
                    # Adicionamos os nomes das categorias no eixo x
                    ax.set_xticklabels(ax.get_xticklabels(), ha='right', fontsize=10)
                    k = k + 1

def plota_histograma(lista_variaveis, hue, df, linhas, colunas, titulo):
    if hue != False:

        if (linhas == 1) and (colunas == 1): 
            k = 0

            df_good = df.loc[df['hue'] == 'GOOD']

            mediana = df[lista_variaveis[k]].median()
            media = df[lista_variaveis[k]].mean()
            plt.figure(figsize = (14, 5))
            ax = sns.histplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', bins = 30, hue = hue)
            ax.set_title(f'{titulo}')
            ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
            ax.set_ylabel(f'Frequência', fontsize = 14)
            ax.axvline(x = mediana, ymax = 0.75 ,color = '#231F20', linestyle = '-', label = f'mediana = {mediana}')
            ax.axvline(x = media, ymax = 0.75,color = '#231F20', linestyle = '--', label = f'media = {media}')
            plt.ticklabel_format(style='plain')
            plt.legend(loc = 'best')
            plt.ticklabel_format(style='plain', axis='both')
            plt.show()
            
        elif linhas == 1:
            fig, axis = plt.subplots(linhas, colunas, figsize = (14, 5), sharey = True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    mediana = df[lista_variaveis[k]].median()
                    media = df[lista_variaveis[k]].mean().round()
                    ax = sns.histplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', ax = axis[j], bins = 30, hue = hue)
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
                    ax.set_ylabel(f'Frequência', fontsize = 14)
                    ax.axvline(x = mediana, ymax = 0.75 ,color = '#231F20', linestyle = '-', label = f'mediana = {mediana}')
                    ax.axvline(x = media, ymax = 0.75,color = '#231F20', linestyle = '--', label = f'media = {media}')
                    ax.ticklabel_format(style='plain')
                    ax.legend(loc = 'best')
                    ax.ticklabel_format(style='plain', axis='both')
                    k = k + 1
            

        elif colunas == 1:
            fig, axis = plt.subplots(linhas, colunas, figsize = (14, 5), sharey = True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    mediana = df[lista_variaveis[k]].median()
                    media = df[lista_variaveis[k]].mean().round()
                    ax = sns.histplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', ax = axis[i], bins = 30, hue = hue)
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
                    ax.set_ylabel(f'Frequência', fontsize = 14)
                    ax.axvline(x = mediana, ymax = 0.75 ,color = '#231F20', linestyle = '-', label = f'mediana = {mediana}')
                    ax.axvline(x = media, ymax = 0.75,color = '#231F20', linestyle = '--', label = f'media = {media}')
                    ax.ticklabel_format(style='plain')
                    ax.legend(loc = 'best')
                    ax.ticklabel_format(style='plain', axis='both')
                    k = k + 1
            
        else:
            fig, axis = plt.subplots(linhas, colunas, figsize = (14, 5), sharey = True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    mediana = df[lista_variaveis[k]].median()
                    media = df[lista_variaveis[k]].mean().round()
                    ax = sns.histplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', ax = axis[i, j], bins = 30, hue = hue)
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
                    ax.set_ylabel(f'Frequência', fontsize = 14)
                    ax.axvline(x = mediana, ymax = 0.75 ,color = '#231F20', linestyle = '-', label = f'mediana = {mediana}')
                    ax.axvline(x = media, ymax = 0.75,color = '#231F20', linestyle = '--', label = f'media = {media}')
                    ax.ticklabel_format(style='plain')
                    ax.legend(loc = 'best')
                    ax.ticklabel_format(style='plain', axis='both')
                    k = k + 1
            
    else:
    
        if (linhas == 1) and (colunas == 1): 
            k = 0
            mediana = df[lista_variaveis[k]].median()
            media = df[lista_variaveis[k]].mean()
            plt.figure(figsize = (14, 5))
            ax = sns.histplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', bins = 30)
            ax.set_title(f'{titulo}')
            ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
            ax.set_ylabel(f'Frequência', fontsize = 14)
            ax.axvline(x = mediana, ymax = 0.75 ,color = '#231F20', linestyle = '-', label = f'mediana = {mediana}')
            ax.axvline(x = media, ymax = 0.75,color = '#231F20', linestyle = '--', label = f'media = {media}')
            plt.ticklabel_format(style='plain')
            plt.legend(loc = 'best')
            plt.ticklabel_format(style='plain', axis='both')
            plt.show()

        elif linhas == 1:
            fig, axis = plt.subplots(linhas, colunas, figsize = (14, 5), sharey = True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    mediana = df[lista_variaveis[k]].median()
                    media = df[lista_variaveis[k]].mean().round()
                    ax = sns.histplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', ax = axis[j], bins = 30)
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
                    ax.set_ylabel(f'Frequência', fontsize = 14)
                    ax.axvline(x = mediana, ymax = 0.75 ,color = '#231F20', linestyle = '-', label = f'mediana = {mediana}')
                    ax.axvline(x = media, ymax = 0.75,color = '#231F20', linestyle = '--', label = f'media = {media}')
                    ax.ticklabel_format(style='plain')
                    ax.legend(loc = 'best')
                    ax.ticklabel_format(style='plain', axis='both')
                    k = k + 1
            
        elif colunas == 1:
            fig, axis = plt.subplots(linhas, colunas, figsize = (14, 5), sharey = True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    mediana = df[lista_variaveis[k]].median()
                    media = df[lista_variaveis[k]].mean().round()
                    ax = sns.histplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', ax = axis[i], bins = 30)
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
                    ax.set_ylabel(f'Frequência', fontsize = 14)
                    ax.axvline(x = mediana, ymax = 0.75 ,color = '#231F20', linestyle = '-', label = f'mediana = {mediana}')
                    ax.axvline(x = media, ymax = 0.75,color = '#231F20', linestyle = '--', label = f'media = {media}')
                    ax.ticklabel_format(style='plain')
                    ax.legend(loc = 'best')
                    ax.ticklabel_format(style='plain', axis='both')
                    k = k + 1
            

        else:
            fig, axis = plt.subplots(linhas, colunas, figsize = (14, 5), sharey = True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    mediana = df[lista_variaveis[k]].median()
                    media = df[lista_variaveis[k]].mean().round()
                    ax = sns.histplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', ax = axis[i, j], bins = 30)
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 14)
                    ax.set_ylabel(f'Frequência', fontsize = 14)
                    ax.axvline(x = mediana, ymax = 0.75 ,color = '#231F20', linestyle = '-', label = f'mediana = {mediana}')
                    ax.axvline(x = media, ymax = 0.75,color = '#231F20', linestyle = '--', label = f'media = {media}')
                    ax.ticklabel_format(style='plain')
                    ax.legend(loc = 'best')
                    ax.ticklabel_format(style='plain', axis='both')
                    k = k + 1

def plota_boxplot(lista_variaveis, hue, df, linhas, colunas, titulo):
    if hue != False:
        if (linhas == 1) and (colunas == 1): 
            k = 0
            plt.figure(figsize = (10, 7))
            ax = sns.boxplot(x = lista_variaveis[k], data = df, palette = ['#1FB3E5', '#64ED8F', '#B864ED'], orient = 'h', y = hue)
            ax.set_title(f'{titulo}')
            ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 10)
            ax.set_ylabel(f'Frequência', fontsize = 10)
            ax.xaxis.set_major_formatter('{:.0f}'.format)
            plt.show()

        elif linhas == 1:
            fig, axis = plt.subplots(linhas, colunas, figsize = (10, 7), sharey = True)
            fig.suptitle(f'{titulo}', fontsize = 10)
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    ax = sns.boxplot(x = lista_variaveis[k], data = df, palette = ['#1FB3E5', '#64ED8F', '#B864ED'], ax = axis[j], orient = 'h', y = hue)
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 10)
                    ax.set_ylabel(f'Frequência', fontsize = 10)
                    ax.set_xticklabels(ax.get_xticks(), fontsize=7) 
                    k = k + 1
            
        elif colunas == 1:
            fig, axis = plt.subplots(linhas, colunas, figsize = (10, 7), sharey = True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    ax = sns.boxplot(x = lista_variaveis[k], data = df, palette = ['#1FB3E5', '#64ED8F', '#B864ED'], ax = axis[i], orient = 'h', y = hue)
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 10)
                    ax.set_ylabel(f'Frequência', fontsize = 10)
                    ax.set_xticklabels(ax.get_xticks(), fontsize=7) 
                    k = k + 1
            
        else:
            fig, axis = plt.subplots(linhas, colunas, figsize = (10, 7), sharey = True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    ax = sns.boxplot(x = lista_variaveis[k], data = df, palette = ['#1FB3E5', '#64ED8F', '#B864ED'], ax = axis[i, j], orient = 'h', y = hue)
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 10)
                    ax.set_ylabel(f'Frequência', fontsize = 10)
                    ax.set_xticklabels(ax.get_xticks(), fontsize=7) 
                    k = k + 1

    else:
        if (linhas == 1) and (colunas == 1): 
            k = 0
            plt.figure(figsize = (10, 7))
            ax = sns.boxplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', orient = 'h')
            ax.set_title(f'{titulo}')
            ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 10)
            ax.set_ylabel(f'Frequência', fontsize = 10)
            ax.xaxis.set_major_formatter('{:.0f}'.format)
            plt.show()

        elif linhas == 1:
            fig, axis = plt.subplots(linhas, colunas, figsize = (10, 7), sharey = True)
            fig.suptitle(f'{titulo}', fontsize = 10)
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    ax = sns.boxplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', ax = axis[j], orient = 'h')
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 10)
                    ax.set_ylabel(f'Frequência', fontsize = 10)
                    ax.set_xticklabels(ax.get_xticks(), fontsize=7) 
                    k = k + 1

        elif colunas == 1:
            fig, axis = plt.subplots(linhas, colunas, figsize = (10, 7), sharey = True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    ax = sns.boxplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', ax = axis[i], orient = 'h')
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 10)
                    ax.set_ylabel(f'Frequência', fontsize = 10)
                    ax.set_xticklabels(ax.get_xticks(), fontsize=7) 
                    k = k + 1
    
        else:
            fig, axis = plt.subplots(linhas, colunas, figsize = (10, 7), sharey = True)
            fig.suptitle(f'{titulo}')
            k = 0
            for i in np.arange(linhas):
                for j in np.arange(colunas):
                    ax = sns.boxplot(x = lista_variaveis[k], data = df, color = '#1FB3E5', ax = axis[i, j], orient = 'h')
                    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize = 10)
                    ax.set_ylabel(f'Frequência', fontsize = 10)
                    ax.set_xticklabels(ax.get_xticks(), fontsize=7) 
                    k = k + 1

def plota_dispersao(df, titulo,  x, y, metodo):
    plt.figure(figsize = (10, 5))
    sns.set(style = 'whitegrid')
    corr1 = str(df[[x, y]].corr(method = metodo).iloc[1, 0].round(2))
    sns.scatterplot(data = df, x = x, y = y, color = '#1FB3E5', sizes = 1, alpha = 0.50, marker = '.')
    plt.text(1, 1, f'Correlacao: {corr1}', fontsize = 12)
    plt.title(f'{titulo}', fontsize = 14)
    plt.xlabel(f'{x}', fontsize = 14)
    plt.ylabel(f'{y}', fontsize = 14)
    plt.ticklabel_format(style = 'plain')
    plt.grid(True, linestyle=':')
    sns.despine()
    plt.tight_layout()
    plt.show()

def roc_curve_graph(classificador, target, y_train, y_predict_train, y_test, y_predict_test, y_predict_proba_train, y_predict_proba_test):

    predict_proba_train = pd.DataFrame(y_predict_proba_train.tolist(), columns=['predict_proba_0', 'predict_proba_1'])
    predict_proba_test = pd.DataFrame(y_predict_proba_test.tolist(), columns=['predict_proba_0', 'predict_proba_1'])

    results_train = y_train[[target]].copy()
    results_train['y_predict_train'] = y_predict_train
    results_train['predict_proba_0'] = list(predict_proba_train['predict_proba_0']) # Probabilidade de ser Bom (classe 0)
    results_train['predict_proba_1'] = list(predict_proba_train['predict_proba_1']) # Probabilidade de ser Mau (classe 1)

    results_test = y_test[[target]].copy()
    results_test['y_predict_test'] = y_predict_test
    results_test['predict_proba_0'] = list(predict_proba_test['predict_proba_0']) # Probabilidade de ser Bom (classe 0)
    results_test['predict_proba_1'] = list(predict_proba_test['predict_proba_1']) # Probabilidade de ser Mau (classe 1)

    # Calculate AUC and ROC for the training set
    y_true_train = results_train[target]
    y_scores_train = results_train['predict_proba_1']
    auc_train = roc_auc_score(y_true_train, y_scores_train)
    fpr_train, tpr_train, thresholds_train = roc_curve(y_true_train, y_scores_train)

    # Calculate AUC and ROC for the test set
    y_true_test = results_test[target]
    y_scores_test = results_test['predict_proba_1']
    auc_test = roc_auc_score(y_true_test, y_scores_test)
    fpr_test, tpr_test, thresholds_test = roc_curve(y_true_test, y_scores_test)

    # Plot ROC curves side by side
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Training set ROC curve
    axs[0].plot(fpr_train, tpr_train, label='ROC Curve (AUC = {:.2f})'.format(auc_train), color='blue')
    axs[0].plot([0, 1], [0, 1], linestyle='--', color='gray')
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    axs[0].set_title(f'ROC Curve - Training Set - {classificador}')
    axs[0].legend()

    # Test set ROC curve
    axs[1].plot(fpr_test, tpr_test, label='ROC Curve (AUC = {:.2f})'.format(auc_test), color='blue')
    axs[1].plot([0, 1], [0, 1], linestyle='--', color='gray')
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].set_title(f'ROC Curve - Test Set - {classificador}')
    axs[1].legend()

    plt.show()

from sklearn.metrics import precision_recall_curve, average_precision_score

def precision_recall_graph(classificador, target, y_train, y_predict_train, y_test, y_predict_test, y_predict_proba_train, y_predict_proba_test):

    predict_proba_train = pd.DataFrame(y_predict_proba_train.tolist(), columns=['predict_proba_0', 'predict_proba_1'])
    predict_proba_test = pd.DataFrame(y_predict_proba_test.tolist(), columns=['predict_proba_0', 'predict_proba_1'])

    results_train = y_train[[target]].copy()
    results_train['y_predict_train'] = y_predict_train
    results_train['predict_proba_0'] = list(predict_proba_train['predict_proba_0']) # Probabilidade de ser Bom (classe 0)
    results_train['predict_proba_1'] = list(predict_proba_train['predict_proba_1']) # Probabilidade de ser Mau (classe 1)

    results_test = y_test[[target]].copy()
    results_test['y_predict_test'] = y_predict_test
    results_test['predict_proba_0'] = list(predict_proba_test['predict_proba_0']) # Probabilidade de ser Bom (classe 0)
    results_test['predict_proba_1'] = list(predict_proba_test['predict_proba_1']) # Probabilidade de ser Mau (classe 1)

    # Calculate Precision-Recall curve and AUC for the training set
    y_true_train = results_train[target]
    y_scores_train = results_train['predict_proba_1']
    precision_train, recall_train, _ = precision_recall_curve(y_true_train, y_scores_train)
    average_precision_train = average_precision_score(y_true_train, y_scores_train)

    # Calculate Precision-Recall curve and AUC for the test set
    y_true_test = results_test[target]
    y_scores_test = results_test['predict_proba_1']
    precision_test, recall_test, _ = precision_recall_curve(y_true_test, y_scores_test)
    average_precision_test = average_precision_score(y_true_test, y_scores_test)

    # Plot Precision-Recall curves side by side
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Training set Precision-Recall curve
    axs[0].plot(recall_train, precision_train, label='Precision-Recall Curve', color='green')
    axs[0].set_xlabel('Recall')
    axs[0].set_ylabel('Precision')
    axs[0].set_title(f'Precision-Recall Curve - Training Set - {classificador}')
    axs[0].legend()

    # Test set Precision-Recall curve
    axs[1].plot(recall_test, precision_test, label='Precision-Recall Curve ', color='green')
    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')
    axs[1].set_title(f'Precision-Recall Curve - Test Set - {classificador}')
    axs[1].legend()

    plt.show()


def verifica_tipo_variavel(df):
    analytics = df.copy()

    qualitativas = [column for column in analytics.columns if 
            (analytics[column].dtype.name == 'object') 
         or ('bad_rate' not in str(analytics[column].name.lower()) and 'mean' not in str(analytics[column].name.lower()) and analytics[column].nunique() <= 2)
         or ('delinq_2yrs' in str(analytics[column].name.lower()))
        ]
    quantitativas = [column for column in analytics.columns if column not in qualitativas]
    # discretas = [column for column in analytics.columns if 
    # (
    #     analytics[column].dtype.name != 'object' 
    #     and analytics[column].nunique() > 2 
    #     and analytics[column].nunique() <= 50 
    #     and 'bad_rate' not in str(analytics[column].name.lower()) 
    #     and 'mean' not in str(analytics[column].name.lower()))
    # ] 
    # quantitativas = [column for column in analytics.columns if 
    # (analytics[column].dtype.name != 'object' and analytics[column].nunique() > 50) 
    #     or ('bad_rate' in str(analytics[column].name.lower())) 
    #     or ('mean' in str(analytics[column].name.lower()))
    # ] 

    qualitativas = pd.DataFrame({'variaveis':qualitativas, 'tipo':'qualitativa'})
    quantitativas = pd.DataFrame({'variaveis':quantitativas, 'tipo':'quantitativas'})
    #discretas = pd.DataFrame({'variaveis':discretas, 'tipo':'discreta'})
    #continuas = pd.DataFrame({'variaveis':continuas, 'tipo':'continua'})

    #variaveis = pd.concat([qualitativas, discretas, continuas])
    variaveis = pd.concat([qualitativas, quantitativas])

    return variaveis

def analisa_correlacao(metodo, df):
    plt.figure(figsize=(14, 7))
    heatmap = sns.heatmap(df.corr(method=metodo), vmin=-1, vmax=1, cmap='magma', annot = True)
    heatmap.set_title(f"Analisando Correlação de {metodo}")
    plt.grid(False)
    plt.box(False)
    plt.tight_layout()
    plt.grid(False)
    plt.show()

def analisa_normalidade(amostra, variavel):

    normaltest_amostra = normaltest(amostra[variavel])
    if normaltest_amostra[1] < 0.05:
        print(f'Pelo Teste de Hipótese, A Hipótese Nula de que a variável "{variavel}" segue uma Distribuição Normal é REJEITADA!')
    else:
        print(f'Pelo Teste de Hipótese, A Hipótese Nula de  que a variável "{variavel}" segue uma Distribuição Normal é ACEITA')

    plt.figure(figsize = (5, 3))
    stats.probplot(amostra[variavel], dist = 'norm', plot = plt)
    plt.title(f'Amostra 1', fontsize = 14)
    plt.grid(False)
    plt.box(False)
    plt.tight_layout()
    plt.show()

def analisa_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IIQ = Q3 - Q1

    outlier_inferior = Q1 - 1.5*IIQ 
    outlier_superior = Q3 + 1.5*IIQ

    return outlier_inferior, outlier_superior

def remove_features_baixa_variancia(target, df, threshold):
    target_column = df[target]
    features = df.drop(target, axis=1)

    selector = VarianceThreshold(threshold=threshold)
    features_filtered = selector.fit_transform(features)

    feature_indices = selector.get_support(indices=True)
    selected_features = features.columns[feature_indices]
    selected_features = selected_features.append(pd.Index([target]))

    return selected_features.tolist()

def remove_features_mutual_information(target, df, threshold):
    x_train, y_train = separa_feature_target(target, df)

    # Calcular a informação mútua entre cada variável e a variável de saída
    mutual_info = mutual_info_classif(x_train, y_train, random_state = 42)

    # Criar um DataFrame com o nome da feature e sua mutual information
    features_selected = pd.DataFrame({'Feature': x_train.columns, 'Mutual Information': mutual_info})
    features_selected = features_selected.loc[features_selected['Mutual Information'] > threshold]
    features_selected = features_selected.sort_values(by='Mutual Information', ascending=False).reset_index(drop=True)

    selected_features = list(features_selected['Feature'])
    selected_features.append(target)
    
    return features_selected, selected_features

def remove_features_feature_importance(target, df, class_weight, threshold):

    # Separa entre Features e Target
    x, y = separa_feature_target(target, df)

    # Criar o modelo de Random Forest
    model = RandomForestClassifier(random_state=42, criterion='log_loss', n_estimators=20, class_weight={0:1, 1:class_weight})

    # Treinar o modelo
    model.fit(x, y)

    # Obter as importâncias das features
    feature_importances = model.feature_importances_

    # Cria o DF

    # Selecionar as features com importância maior que zero

    selected_features = list(x.columns[feature_importances > threshold])
    selected_features.append(target)

    feature_importance = [importance for importance in feature_importances if importance > threshold]

    return selected_features, feature_importance



def teste_hipotese_duas_amostras_independentes(parametrico, amostra1, amostra2, variavel):
    media_amostra_1 = amostra1[variavel].mean()
    media_amostra_2 = amostra2[variavel].mean()
    mediana_amostra_1 = amostra1[variavel].median()
    mediana_amostra_2 = amostra2[variavel].median()

    if parametrico == True: 
        print(f'Média Amostra 1: {media_amostra_1}')
        print(f'Média Amostra 2: {media_amostra_2}')
        stat, p_value = ztest(amostra1[variavel], amostra2[variavel]) 
        if p_value > 0.05:
            print(f'Pelo Teste de Hipótese Z, não há diferença significativa entre as médias da Amostra 1 e Amostra 2')
        else:
            print(f'Pelo Teste de Hipótese Z, há diferença significativa entre as médias da Amostra 1 e Amostra 2')
    else:
        print(f'Mediana Amostra 1: {mediana_amostra_1}')
        print(f'Mediana Amostra 2: {mediana_amostra_2}')
        stat, p_value = stats.mannwhitneyu(amostra1[variavel], amostra2[variavel]) 
        if p_value > 0.05:
            print(f'Pelo Teste de Hipótese de Mann Whitney, não há diferença significativa entre as medianas da Amostra 1 e Amostra 2')
        else:
            print(f'Pelo Teste de Hipótese de Mann Whitney, há diferença significativa entre as medianas da Amostra 1 e Amostra 2')


def teste_hipotese_muitas_amostras_independentes(amostras, variavel):
    medianas = []
    
    for i, amostra in enumerate(amostras):
        mediana_amostra = amostra[variavel].median()
        medianas.append(mediana_amostra)
        print(f'Mediana Amostra {i+1}: {mediana_amostra}')

    stat, p_value = kruskal(*[amostra[variavel] for amostra in amostras])
    
    if p_value > 0.05:
        print(f'Pelo teste de Kruskal-Wallis, não há diferença significativa entre as medianas das amostras')
    else:
        print(f'Pelo teste de Kruskal-Wallis, há diferença significativa entre as medianas das amostras')

def teste_hipotese_duas_variaveis_categoricas(df, variavel1, variavel2):
    # Crie tabelas de contingência
    crosstab = pd.crosstab(df[variavel1], df[variavel2])
    
    # Realize o teste qui-quadrado
    chi2, p, _, _ = chi2_contingency(crosstab)
    
    # Verifique o valor-p
    if p > 0.05:
        print(f'Pelo Teste Qui-Quadrado, não há associação significativa entre {variavel1} e {variavel2}.')
    else:
        print(f'Pelo Teste Qui-Quadrado, há associação significativa entre {variavel1} e {variavel2}.')

def ks(y_proba_0, y_proba_1):
    KS, p_value = stats.ks_2samp(y_proba_0, y_proba_1)

    if p_value > 0.05:
        ks_message = 'Pelo Teste de KS, não há diferença significativa entre as amostras'
    else:
        ks_message = 'Pelo Teste de KS, há diferença significativa entre as amostras'

    return KS, ks_message

def woe(df, feature, target):
    good = df.loc[df[target] == 'BAD'].groupby(feature, as_index = False)[target].count().rename({target:'good'}, axis = 1)
    bad = df.loc[df[target] == 'GOOD'].groupby(feature, as_index = False)[target].count().rename({target:'bad'}, axis = 1)

    woe = good.merge(bad, on = feature, how = 'left')
    woe['percent_good'] = woe['good']/woe['good'].sum()
    woe['percent_bad'] = woe['bad']/woe['bad'].sum()
    woe['woe'] = round(np.log(woe['percent_good']/woe['percent_bad']), 3)
    woe['iv'] = ((woe['percent_good'] - woe['percent_bad'])*np.log(woe['percent_good']/woe['percent_bad'])).sum()

    woe['woe'].fillna(0, inplace = True)
    woe['iv'].fillna(0, inplace = True)

    weight_of_evidence = woe['woe'].unique()
    iv = round(woe['iv'].max(), 2)

    x = df[feature].unique()
    y = woe['woe']
    plt.figure(figsize = (10, 4))
    plt.plot(x, y, marker = 'o', linestyle = '--', linewidth=2, color = '#1FB3E5')
    for label, value in zip(x, y):
        plt.text(x = label, y = value, s = str(value), fontsize = 20, color = 'red', ha='left', va='center', rotation = 45)
    # plt.title(f'WOE of "{feature}" with an Information Value {iv} ', fontsize=14)
    plt.title(f'Weight of Evidence da variável "{feature}"', fontsize=14)
    plt.xlabel('Classes', fontsize=14)
    plt.ylabel('Weight of Evidence', fontsize=14)
    plt.xticks(ha='right', fontsize = 10, rotation = 45)
    
    # return woe

def calculate_ks(y_proba_0, y_proba_1):
    # Calcular as probabilidades acumuladas
    proba_cum_0 = np.cumsum(y_proba_0) / np.sum(y_proba_0)
    proba_cum_1 = np.cumsum(y_proba_1) / np.sum(y_proba_1)

    # Calcular a diferença máxima entre as probabilidades acumuladas
    KS = np.max(np.abs(proba_cum_0 - proba_cum_1))

    return KS

# def metricas_classificacao(classificador, y_train, y_predict_train, y_test, y_predict_test):
#     accuracy = accuracy_score(y_train, y_predict_train)
#     precision = precision_score(y_train, y_predict_train)
#     recall = recall_score(y_train, y_predict_train)
#     roc_curve = roc_auc_score(y_train, y_predict_train)
#     metricas_treino = pd.DataFrame({'Acuracia':accuracy, 'Precisao':precision, 'Recall':recall, 'AUC':roc_curve, 'Etapa':'treino','Classificador':classificador}, index = np.arange(1))
    
#     accuracy = accuracy_score(y_test, y_predict_test)
#     precision = precision_score(y_test, y_predict_test)
#     recall = recall_score(y_test, y_predict_test)
#     roc_curve = roc_auc_score(y_test, y_predict_test)
#     metricas_teste = pd.DataFrame({'Acuracia':accuracy, 'Precisao':precision, 'Recall':recall, 'AUC':roc_curve, 'Etapa':'teste','Classificador':classificador}, index = np.arange(1, 2))
    
#     metricas_finais = pd.concat([metricas_treino, metricas_teste])

#     return metricas_finais

# def metricas_classificacao(classificador, y_train, y_predict_train, y_test, y_predict_test, y_predict_proba_train, y_predict_proba_test):
#     accuracy_train = accuracy_score(y_train, y_predict_train)
#     precision_train = precision_score(y_train, y_predict_train)
#     recall_train = recall_score(y_train, y_predict_train)
#     roc_auc_train = roc_auc_score(y_train, y_predict_proba_train[:, 1])
    
#     metricas_treino = pd.DataFrame({'Acuracia': accuracy_train, 'Precisao': precision_train, 'Recall': recall_train, 'AUC': roc_auc_train, 'Etapa': 'treino', 'Classificador': classificador}, index=[0])
    
#     accuracy_test = accuracy_score(y_test, y_predict_test)
#     precision_test = precision_score(y_test, y_predict_test)
#     recall_test = recall_score(y_test, y_predict_test)
#     roc_auc_test = roc_auc_score(y_test, y_predict_proba_test[:, 1])
    
#     metricas_teste = pd.DataFrame({'Acuracia': accuracy_test, 'Precisao': precision_test, 'Recall': recall_test, 'AUC': roc_auc_test, 'Etapa': 'teste', 'Classificador': classificador}, index=[0])
    
#     metricas_finais = pd.concat([metricas_treino, metricas_teste])

#     return metricas_finais

def metricas_classificacao(classificador, y_train, y_predict_train, y_test, y_predict_test, y_predict_proba_train, y_predict_proba_test):
    accuracy_train = accuracy_score(y_train, y_predict_train)
    precision_train = precision_score(y_train, y_predict_train)
    recall_train = recall_score(y_train, y_predict_train)
    #roc_auc_train = roc_auc_score(y_train, y_predict_train)
    
    #metricas_treino = pd.DataFrame({'Acuracia': accuracy_train, 'Precisao': precision_train, 'Recall': recall_train, 'AUC': roc_auc_train, 'Etapa': 'treino', 'Classificador': classificador}, index=[0])
    metricas_treino = pd.DataFrame({'Acuracia': accuracy_train, 'Precisao': precision_train, 'Recall': recall_train, 'Etapa': 'treino', 'Classificador': classificador}, index=[0])
    
    accuracy_test = accuracy_score(y_test, y_predict_test)
    precision_test = precision_score(y_test, y_predict_test)
    recall_test = recall_score(y_test, y_predict_test)
    #roc_auc_test = roc_auc_score(y_test, y_predict_test)
    
    #metricas_teste = pd.DataFrame({'Acuracia': accuracy_test, 'Precisao': precision_test, 'Recall': recall_test, 'AUC': roc_auc_test, 'Etapa': 'teste', 'Classificador': classificador}, index=[0])
    metricas_teste = pd.DataFrame({'Acuracia': accuracy_test, 'Precisao': precision_test, 'Recall': recall_test, 'Etapa': 'teste', 'Classificador': classificador}, index=[0])
    
    metricas_finais = pd.concat([metricas_treino, metricas_teste])

    return metricas_finais

def metricas_classificacao_modelos_juntos(lista_modelos):
    metricas_modelos = pd.concat(lista_modelos).set_index('Classificador')
    return metricas_modelos

def validacao_cruzada_classificacao(classificador, x_train, y_train, class_weight, n_splits):

    qualitativas_numericas = [column for column in x_train.columns if x_train[column].nunique() <= 5]
    discretas = [column for column in x_train.columns if (x_train[column].nunique() > 5) and (x_train[column].nunique() <= 50)]
    continuas = [column for column in x_train.columns if x_train[column].nunique() > 50]

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = {
        'Regressão Logística': make_pipeline(
            ColumnTransformer([
                ('qualitativas_numericas', make_pipeline(SimpleImputer(strategy='constant')), qualitativas_numericas),
                ('discretas', make_pipeline(SimpleImputer(strategy='median')), discretas),
                ('continuas', make_pipeline(SimpleImputer(strategy='median')), continuas)
                ]),
            LogisticRegression(random_state=42, class_weight={0:1, 1:class_weight}, solver = 'liblinear')
        ),
        'Random Forest': make_pipeline(
            ColumnTransformer([
                ('qualitativas_numericas', make_pipeline(SimpleImputer(strategy='constant')), qualitativas_numericas),
                ('discretas', make_pipeline(SimpleImputer(strategy='median')), discretas),
                ('continuas', make_pipeline(SimpleImputer(strategy='median')), continuas)
                ]),
            RandomForestClassifier(random_state=42, criterion='log_loss', n_estimators=20, max_depth=4, class_weight={0:1, 1:class_weight})
        ),
        'XGBoost': make_pipeline(
            ColumnTransformer([
                ('qualitativas_numericas', make_pipeline(SimpleImputer(strategy='constant')), qualitativas_numericas),
                ('discretas', make_pipeline(SimpleImputer(strategy='median')), discretas),
                ('continuas', make_pipeline(SimpleImputer(strategy='median')), continuas)
                ]),
            XGBClassifier(random_state=42, n_estimators=20, max_depth=5, learning_rate=0.01, eval_metric='logloss', objective='binary:logistic', scale_pos_weight = class_weight)
        )
    }

    if classificador in models:
        model = models[classificador]
    else:
        print('Utilize Regressão Logística, Random Forest ou XGBoost como opções de Classificadores!')
    
    scoring = ['accuracy', 'precision', 'recall', 'roc_auc']
    scores = cross_validate(model, x_train, y_train, cv=kfold, scoring=scoring, return_train_score=False)
    
    metricas_finais = pd.DataFrame({
        'Acuracia': scores['test_accuracy'].mean(),
        'Precisao': scores['test_precision'].mean(),
        'Recall': scores['test_recall'].mean(),
        'AUC':scores['test_roc_auc'].mean(),
        'Etapa': 'validacao_cruzada',
        'Classificador': classificador
    }, index=[1])
    
    return metricas_finais


def metricas_regressao_modelos_juntos(lista_modelos):
    metricas_modelos = pd.concat(lista_modelos).set_index('Regressor')
    return metricas_modelos

def retorno_financeiro(target, y_true, y_predict):
    df = pd.DataFrame({'y_true':y_true[target].values, 'y_predict':y_predict})

    TN = df.loc[(df['y_true'] == 0) & (df['y_predict'] == 0)].shape[0]
    FN = df.loc[(df['y_true'] == 1) & (df['y_predict'] == 0)].shape[0]
    FP = df.loc[(df['y_true'] == 0) & (df['y_predict'] == 1)].shape[0]
    TP = df.loc[(df['y_true'] == 1) & (df['y_predict'] == 1)].shape[0]

    matriz_confusao = np.array(
        [(TN, FP),
        (FN, TP)]
    )
    matriz_custo_beneficios = np.array(
        [(0, 10),
        (0, 90)]
    )
    retorno_financeiro = int(
        (matriz_confusao[0, 0]*matriz_custo_beneficios[0, 0]) - (matriz_confusao[0, 1]*matriz_custo_beneficios[0, 1]) - (matriz_confusao[1, 0]*matriz_custo_beneficios[1, 0]) + (matriz_confusao[1, 1]*matriz_custo_beneficios[1, 1])
        )
    return retorno_financeiro



def separa_feature_target(target, dados):
    x = dados.drop(target, axis = 1)
    y = dados[[target]]

    return x, y


def separa_treino_teste(target, dados, size):
    x = dados.drop(target, axis = 1)
    y = dados[target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= size, random_state = 42)

    df_train = pd.concat([y_train, x_train], axis = 1)
    df_test = pd.concat([y_test, x_test], axis = 1)

    return df_train, df_test

def discretiza_variavel(df, variavel_quant, variavel_qualit, bins, labels, right):
    df[variavel_qualit] = pd.cut(
        df[variavel_quant], 
        bins= bins, 
        labels= labels, 
        right = right
    )
    df.drop(variavel_quant, axis = 1, inplace = True)

def transform_to_deciles(df, variavel_continua):
    # Calcula os limites dos deciles
    decile_limits = [i / 10 for i in range(11)]  # [0.0, 0.1, 0.2, ..., 1.0]
    
    # Aplica a função qcut para transformar a variável em deciles
    deciles = pd.qcut(df[variavel_continua], q=10, labels=False, duplicates='drop')
    
    return deciles

def transform_to_percentiles(df, variavel_continua):
    # Calcula os limites dos percentis de 0,01 a 0,95 em incrementos de 0,01
    percentile_limits = [i / 100 for i in range(1, 100, 10)]  # [0.01, 0.02, ..., 0.95]
    
    # Aplica a função qcut para transformar a variável em percentis
    percentiles = pd.qcut(df[variavel_continua], q=percentile_limits, labels=False, duplicates='drop')
    
    return percentiles


def Classificador(classificador, x_train, y_train, x_test, y_test, class_weight):
    cols = list(x_train.columns)
    imputer = simple_imputer(x_train)
    x_train = pd.DataFrame(imputer.transform(x_train), columns = x_train.columns)
    x_test = pd.DataFrame(imputer.transform(x_test), columns = x_test.columns)

    # Define as colunas categóricas e numéricas
    models = {
        'Regressão Logística': make_pipeline(
            ColumnTransformer([
                ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols),
                ('scaler', make_pipeline(MinMaxScaler()), cols)
            ]),
            LogisticRegression(random_state=42, class_weight={0:1, 1:class_weight}, solver='liblinear')
        ),
        'Random Forest': make_pipeline(
            ColumnTransformer([
                ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols)
            ]),
            RandomForestClassifier(random_state=42, criterion='log_loss', n_estimators=20, max_depth=4, class_weight={0:1, 1:class_weight})
        ),
        'XGBoost': make_pipeline(
            ColumnTransformer([
                ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols)
            ]),
            XGBClassifier(random_state=42, n_estimators=20, max_depth=5, learning_rate=0.01, eval_metric='logloss', objective='binary:logistic', scale_pos_weight=class_weight)
        )
    }

    if classificador in models:
        model = models[classificador]
    else:
        print('Utilize Regressão Logística, Random Forest ou XGBoost como opções de Classificadores!')

    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    y_proba_train = model.predict_proba(x_train)
    y_proba_test = model.predict_proba(x_test)

    return model, y_pred_train, y_pred_test, y_proba_train, y_proba_test


def modelo_otimizado(classificador, x_train, y_train, x_test, y_test):
    # Define as colunas categóricas e numéricas
    qualitativas_numericas = [column for column in x_train.columns if x_train[column].nunique() <= 5]
    discretas = [column for column in x_train.columns if (x_train[column].nunique() > 5) and (x_train[column].nunique() <= 50)]
    continuas = [column for column in x_train.columns if x_train[column].nunique() > 50]

    # Define o ColumnTransformer
    preprocessor = ColumnTransformer([
                ('qualitativas_numericas', make_pipeline(SimpleImputer(strategy='constant')), qualitativas_numericas),
                ('discretas', make_pipeline(SimpleImputer(strategy='median')), discretas),
                ('continuas', make_pipeline(SimpleImputer(strategy='median')), continuas)
    ])

    # Define o modelo de XGBoost com a otimização de hiperparâmetros via BayesSearch
    model = make_pipeline(
        preprocessor,
        BayesSearchCV(
            XGBClassifier(random_state=42, eval_metric='logloss', objective='binary:logistic'),
            {
                'n_estimators': (10, 15, 20, 50), # Número de Árvores construídas
                'max_depth': (4, 5, 7), # Profundidade Máxima de cada Árvore
                'learning_rate': (0.01, 0.05), # Tamanho do passo utilizado no Método do Gradiente Descendente
                'reg_alpha':(0.5, 1), # Valor do Alpha aplicado durante a Regularização Lasso L1 
                'reg_lambda':(0.5, 1), # Valor do Lambda aplicado durante a Regularização Ridge L2
                'gamma':(0.5, 1), # Valor mínimo permitido para um Nó de Árvore ser aceito. Ajuda a controlar o crescimento das Árvores, evitando divisões insignificantes
                'colsample_bytree':(0.5, 1), # Porcentagem de Colunas utilizada para a amostragem aleatória durante a criação das Árvores
                'subsample':(0.5, 1), # Porcentagem de Linhas utilizada para a amostragem aleatória durante a criação das Árvores
                'scale_pos_weight':(8, 10, 12, 14) # Peso atribuído a classe positiva, aumentando a importância da classe minoritária
            },
            n_iter=10,
            random_state=42,
            n_jobs=-1,
            scoring='precision',
            cv=5
        )
    )

    # Treina o modelo
    model.fit(x_train, y_train)

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    y_proba_train = model.predict_proba(x_train)
    y_proba_test = model.predict_proba(x_test)

    return model, y_pred_train, y_pred_test, y_proba_train, y_proba_test, model.named_steps['bayessearchcv'].best_params_


def modelo_corte_probabilidade(classificador, x_train, y_train, x_test, y_test, target):

    # Define as colunas categóricas e numéricas
    qualitativas_numericas = [column for column in x_train.columns if x_train[column].nunique() <= 5]
    discretas = [column for column in x_train.columns if (x_train[column].nunique() > 5) and (x_train[column].nunique() <= 50)]
    continuas = [column for column in x_train.columns if x_train[column].nunique() > 50]
    
    list_threshold = [0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55]
    list_lucro = []
    for threshold in list_threshold:
        # Define o ColumnTransformer
        preprocessor = ColumnTransformer([
                    ('qualitativas_numericas', make_pipeline(SimpleImputer(strategy='constant')), qualitativas_numericas),
                    ('discretas', make_pipeline(SimpleImputer(strategy='median')), discretas),
                    ('continuas', make_pipeline(SimpleImputer(strategy='median')), continuas)
        ])
        model = make_pipeline(
        preprocessor,
        XGBClassifier(
            random_state=42, 
            eval_metric='logloss', 
            objective='binary:logistic', 
            n_estimators = 15, 
            max_depth = 7, 
            learning_rate = 0.029858668143868672,
            reg_alpha = 0.5255672768385259,
            reg_lambda = 0.785388901339449,
            gamma = 0.9600046132186582,
            colsample_bytree = 0.7717015338451563,
            subsample = 0.6928647954923324,
            scale_pos_weight = 8,
            base_score = threshold
        )
        )
        
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        lucro = retorno_financeiro(target, y_test, y_pred)
        list_lucro.append(lucro)
    
    corte_probabilidade = pd.DataFrame({'threshold':list_threshold, 'lucro':list_lucro})
    return corte_probabilidade


def modelo_oficial(classificador, x, y):
    # Define as colunas categóricas e numéricas
    qualitativas_numericas = [column for column in x.columns if x[column].nunique() <= 5]
    discretas = [column for column in x.columns if (x[column].nunique() > 5) and (x[column].nunique() <= 50)]
    continuas = [column for column in x.columns if x[column].nunique() > 50]

    # Define o ColumnTransformer
    preprocessor = ColumnTransformer([
                ('qualitativas_numericas', make_pipeline(SimpleImputer(strategy='constant')), qualitativas_numericas),
                ('discretas', make_pipeline(SimpleImputer(strategy='median')), discretas),
                ('continuas', make_pipeline(SimpleImputer(strategy='median')), continuas)
    ])
    # Define o modelo de XGBoost com a otimização de hiperparâmetros via BayesSearch
    model = make_pipeline(
        preprocessor,
        XGBClassifier(
            random_state=42, 
            eval_metric='logloss', 
            objective='binary:logistic', 
            n_estimators = 15, 
            max_depth = 7, 
            learning_rate = 0.029858668143868672,
            reg_alpha = 0.5255672768385259,
            reg_lambda = 0.785388901339449,
            gamma = 0.9600046132186582,
            colsample_bytree = 0.7717015338451563,
            subsample = 0.6928647954923324,
            scale_pos_weight = 8,
            base_score = 0.54
        )
    )

    # Treina o modelo
    model.fit(x, y)

    y_pred_train = model.predict(x)
    y_proba_train = model.predict_proba(x)

    return model, y_pred_train, y_proba_train
