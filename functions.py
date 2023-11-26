## Bibliotecas De Manipulação de Dados e Visualização
import pandas as pd 
import builtins as builtins
import matplotlib.pyplot as plt
import seaborn as sns 
from IPython.display import display, Image
from tabulate import tabulate
from matplotlib.lines import Line2D

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
from numpy import interp

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
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# Bibliotecas de Métricas de Machine Learning
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, precision_score, recall_score, precision_recall_curve, average_precision_score, f1_score, confusion_matrix, silhouette_score

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

def auc_ks_juntos(classificador, target, 
                                    y_train, y_predict_train, 
                                    y_test, y_predict_test, 
                                    y_predict_proba_train, y_predict_proba_test, 
                                    cv_results):

    predict_proba_train = pd.DataFrame(y_predict_proba_train.tolist(), columns=['predict_proba_0', 'predict_proba_1'])
    predict_proba_test = pd.DataFrame(y_predict_proba_test.tolist(), columns=['predict_proba_0', 'predict_proba_1'])

    # Inicialize as variáveis x_max_ks e y_max_ks fora dos blocos condicionais
    x_max_ks_train, y_max_ks_train = 0, 0
    x_max_ks_test, y_max_ks_test = 0, 0
    x_max_ks_cv, y_max_ks_cv = 0, 0

    ### Treino
    results_train = y_train[[target]].copy()
    results_train['y_predict_train'] = y_predict_train
    results_train['predict_proba_0'] = list(predict_proba_train['predict_proba_0']) # Probabilidade de ser Bom (classe 0)
    results_train['predict_proba_1'] = list(predict_proba_train['predict_proba_1']) # Probabilidade de ser Mau (classe 1)

    results_train_sorted = results_train.sort_values(by='predict_proba_1', ascending=False)
    results_train_sorted['Cumulative N Population'] = range(1, results_train_sorted.shape[0] + 1)
    results_train_sorted['Cumulative N Good'] = results_train_sorted[target].cumsum()
    results_train_sorted['Cumulative N Bad'] = results_train_sorted['Cumulative N Population'] - results_train_sorted['Cumulative N Good']
    results_train_sorted['Cumulative Perc Population'] = results_train_sorted['Cumulative N Population'] / results_train_sorted.shape[0]
    results_train_sorted['Cumulative Perc Good'] = results_train_sorted['Cumulative N Good'] / results_train_sorted[target].sum()
    results_train_sorted['Cumulative Perc Bad'] = results_train_sorted['Cumulative N Bad'] / (results_train_sorted.shape[0] - results_train_sorted[target].sum())

    max_ks_index_train = np.argmax(results_train_sorted['Cumulative Perc Good'] - results_train_sorted['Cumulative Perc Bad'])
    x_max_ks_train = results_train_sorted['Cumulative Perc Population'].iloc[max_ks_index_train]
    y_max_ks_train = results_train_sorted['Cumulative Perc Good'].iloc[max_ks_index_train]
    y_min_ks_train = results_train_sorted['Cumulative Perc Bad'].iloc[max_ks_index_train]

        ###### Calculate AUC and ROC for the training set
    y_true_train = results_train[target]
    y_scores_train = results_train['predict_proba_1']
    auc_train = roc_auc_score(y_true_train, y_scores_train)
    fpr_train, tpr_train, thresholds_train = roc_curve(y_true_train, y_scores_train)
        ###### Calculate KS curve for the training set
    KS_train = round(np.max(results_train_sorted['Cumulative Perc Good'] - results_train_sorted['Cumulative Perc Bad']), 2)

    ### Teste
    results_test = y_test[[target]].copy()
    results_test['y_predict_test'] = y_predict_test
    results_test['predict_proba_0'] = list(predict_proba_test['predict_proba_0']) # Probabilidade de ser Bom (classe 0)
    results_test['predict_proba_1'] = list(predict_proba_test['predict_proba_1']) # Probabilidade de ser Mau (classe 1)

    results_test_sorted = results_test.sort_values(by='predict_proba_1', ascending=False)
    results_test_sorted['Cumulative N Population'] = range(1, results_test_sorted.shape[0] + 1)
    results_test_sorted['Cumulative N Good'] = results_test_sorted[target].cumsum()
    results_test_sorted['Cumulative N Bad'] = results_test_sorted['Cumulative N Population'] - results_test_sorted['Cumulative N Good']
    results_test_sorted['Cumulative Perc Population'] = results_test_sorted['Cumulative N Population'] / results_test_sorted.shape[0]
    results_test_sorted['Cumulative Perc Good'] = results_test_sorted['Cumulative N Good'] / results_test_sorted[target].sum()
    results_test_sorted['Cumulative Perc Bad'] = results_test_sorted['Cumulative N Bad'] / (results_test_sorted.shape[0] - results_test_sorted[target].sum())

    max_ks_index_test = np.argmax(results_test_sorted['Cumulative Perc Good'] - results_test_sorted['Cumulative Perc Bad'])
    x_max_ks_test = results_test_sorted['Cumulative Perc Population'].iloc[max_ks_index_test]
    y_max_ks_test = results_test_sorted['Cumulative Perc Good'].iloc[max_ks_index_test]
    y_min_ks_test = results_test_sorted['Cumulative Perc Bad'].iloc[max_ks_index_test]


            ###### Calculate AUC and ROC for the test set
    y_true_test = results_test[target]
    y_scores_test = results_test['predict_proba_1']
    auc_test = roc_auc_score(y_true_test, y_scores_test)
    fpr_test, tpr_test, thresholds_test = roc_curve(y_true_test, y_scores_test)
            ###### Calculate KS curve for the test set
    KS_test = round(np.max(results_test_sorted['Cumulative Perc Good'] - results_test_sorted['Cumulative Perc Bad']), 2)

    # Cross-validation set
    auc_scores_cv = []
    ks_scores_cv = []
    roc_curves_cv = []
    ks_curves_cv = []
    for fold_results in cv_results:
        results_cv = fold_results[[target]].copy()
        results_cv['y_predict_cv'] = fold_results['y_predict']
        results_cv['predict_proba_0'] = fold_results['predict_proba_0']
        results_cv['predict_proba_1'] = fold_results['predict_proba_1']

        y_true_cv = results_cv[target]
        y_scores_cv = results_cv['predict_proba_1']

        # Aggregate ROC curves
        fpr_cv, tpr_cv, _ = roc_curve(y_true_cv, y_scores_cv)
        roc_curves_cv.append((fpr_cv, tpr_cv))

        # Aggregate AUC scores
        auc_cv = roc_auc_score(y_true_cv, y_scores_cv)
        auc_scores_cv.append(auc_cv)

        # Aggregate KS scores
        results_cv_sorted = results_cv.sort_values(by='predict_proba_1', ascending=False)
        results_cv_sorted['Cumulative N Population'] = range(1, results_cv_sorted.shape[0] + 1)
        results_cv_sorted['Cumulative N Good'] = results_cv_sorted[target].cumsum()
        results_cv_sorted['Cumulative N Bad'] = results_cv_sorted['Cumulative N Population'] - results_cv_sorted['Cumulative N Good']
        results_cv_sorted['Cumulative Perc Population'] = results_cv_sorted['Cumulative N Population'] / results_cv_sorted.shape[0]
        results_cv_sorted['Cumulative Perc Good'] = results_cv_sorted['Cumulative N Good'] / results_cv_sorted[target].sum()
        results_cv_sorted['Cumulative Perc Bad'] = results_cv_sorted['Cumulative N Bad'] / (results_cv_sorted.shape[0] - results_cv_sorted[target].sum())
        ks_cv = np.max(results_cv_sorted['Cumulative Perc Good'] - results_cv_sorted['Cumulative Perc Bad'])
        ks_scores_cv.append(ks_cv)

    # Calculate average ROC, AUC and KS scores across folds
    auc_cv_mean = np.mean(auc_scores_cv)
    ks_cv_mean = np.mean(ks_scores_cv)
    mean_fpr_cv = np.linspace(0, 1, 100)  # You can adjust the number of points for a smoother curve
    mean_tpr_cv = np.mean([interp(mean_fpr_cv, fpr, tpr) for fpr, tpr in roc_curves_cv], axis=0)
    mean_ks_cv = np.mean(ks_curves_cv, axis=0)

    max_ks_index_cv = np.argmax(results_cv_sorted['Cumulative Perc Good'] - results_cv_sorted['Cumulative Perc Bad'])
    x_max_ks_cv = results_cv_sorted['Cumulative Perc Population'].iloc[max_ks_index_cv]
    y_max_ks_cv = results_cv_sorted['Cumulative Perc Good'].iloc[max_ks_index_cv]
    y_min_ks_cv = results_cv_sorted['Cumulative Perc Bad'].iloc[max_ks_index_cv]

    KS_cv = round(np.max(results_cv_sorted['Cumulative Perc Good'] - results_cv_sorted['Cumulative Perc Bad']), 2)

    # Plot ROC and KS curves side by side
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Training set ROC curve
    axs[0].plot(fpr_train, tpr_train, label='Train ROC Curve (AUC = {:.2f})'.format(auc_train), color='blue')
    axs[0].fill_between(fpr_train, 0, tpr_train, color='gray', alpha=0.3)  # Preencha a área sob a curva ROC
    axs[0].plot([0, 1], [0, 1], linestyle='--', color='black')
    axs[0].set_xlabel('False Positive Rate', fontsize = 14)
    axs[0].set_ylabel('True Positive Rate', fontsize = 14)
    axs[0].set_title(f'ROC Curve - {classificador}', fontsize = 14)

    # Test set ROC curve
    axs[0].plot(fpr_test, tpr_test, label='Test ROC Curve (AUC = {:.2f})'.format(auc_test), color='red')
    axs[0].fill_between(fpr_test, 0, tpr_test, color='gray', alpha=0.3)  # Preencha a área sob a curva ROC

    # Cross-validation set ROC cruve
    axs[0].plot(mean_fpr_cv, mean_tpr_cv, label='CV ROC Curve (AUC = {:.2f})'.format(auc_cv_mean), color='green')
    axs[0].fill_between(mean_fpr_cv, 0, mean_tpr_cv, color='gray', alpha=0.3)

    # Adicione a legenda personalizada com cores para a curva ROC
    roc_legend_labels = [
        {'label': 'Train ROC Curve (AUC = {:.2f})'.format(auc_train), 'color': 'blue', 'marker': 'o'},
        {'label': 'Test ROC Curve (AUC = {:.2f})'.format(auc_test), 'color': 'red', 'marker': 's'},
        {'label': 'Test ROC Curve (AUC = {:.2f})'.format(auc_test), 'color': 'green', 'marker': '^'}
    ]

    # Criar marcadores personalizados para a legenda ROC
    roc_legend_handles = [Line2D([0], [0], marker=label_info['marker'], color='w', markerfacecolor=label_info['color'], markersize=10) for label_info in roc_legend_labels]

    # Adicione a legenda personalizada ao gráfico da curva ROC
    roc_legend = axs[0].legend(handles=roc_legend_handles, labels=[label_info['label'] for label_info in roc_legend_labels], loc='upper right', bbox_to_anchor=(0.9, 0.4), fontsize='11')
    roc_legend.set_title('ROC AUC', prop={'size': '11'})


    # Train set KS curve
    axs[1].plot(results_train_sorted['Cumulative Perc Population'], results_train_sorted['Cumulative Perc Good'], label='Train Positive Class (Class 1)', color='blue')
    axs[1].plot(results_train_sorted['Cumulative Perc Population'], results_train_sorted['Cumulative Perc Bad'], label='Train Negative Class (Class 0)', color='blue')
    axs[1].plot([x_max_ks_train, x_max_ks_train], [y_min_ks_train, y_max_ks_train], color='black', linestyle='--')
    axs[1].fill_between(results_train_sorted['Cumulative Perc Population'], results_train_sorted['Cumulative Perc Good'], results_train_sorted['Cumulative Perc Bad'], color='gray', alpha=0.5)
    axs[1].text(x=results_train_sorted['Cumulative Perc Population'].iloc[np.argmax(results_train_sorted['Cumulative Perc Good'] - results_train_sorted['Cumulative Perc Bad'])],
                y=(y_min_ks_train + results_train_sorted['Cumulative Perc Good'].iloc[np.argmax(results_train_sorted['Cumulative Perc Good'] - results_train_sorted['Cumulative Perc Bad'])]) / 2,
                s=str(KS_train), fontsize = 14, color='blue', ha='left', va='center', rotation=45)
    axs[1].set_xlabel('Cumulative Percentage of Population', fontsize = 14)
    axs[1].set_ylabel('Cumulative Percentage', fontsize = 14)
    axs[1].set_title(f'KS Plot - {classificador}', fontsize = 14)

    # Test set KS curve
    axs[1].plot(results_test_sorted['Cumulative Perc Population'], results_test_sorted['Cumulative Perc Good'], label='Test Positive Class (Class 1)', color='red')
    axs[1].plot(results_test_sorted['Cumulative Perc Population'], results_test_sorted['Cumulative Perc Bad'], label='Test Negative Class (Class 0)', color='red')
    axs[1].plot([x_max_ks_test, x_max_ks_test], [y_min_ks_test, y_max_ks_test], color='black', linestyle='--')
    axs[1].fill_between(results_test_sorted['Cumulative Perc Population'], results_test_sorted['Cumulative Perc Good'], results_test_sorted['Cumulative Perc Bad'], color='gray', alpha=0.5)
    axs[1].text(x=results_test_sorted['Cumulative Perc Population'].iloc[np.argmax(results_test_sorted['Cumulative Perc Good'] - results_test_sorted['Cumulative Perc Bad'])],
                y=(y_min_ks_test + results_test_sorted['Cumulative Perc Good'].iloc[np.argmax(results_test_sorted['Cumulative Perc Good'] - results_test_sorted['Cumulative Perc Bad'])]) / 2,
                s=str(KS_test), fontsize = 14, color='red', ha='left', va='center', rotation=45)
    axs[1].set_xlabel('Cumulative Percentage of Population', fontsize = 14)
    axs[1].set_ylabel('Cumulative Percentage', fontsize = 14)
    axs[1].set_title(f'KS Plot - {classificador}', fontsize = 14)

    # Cross-validation set KS curve
    axs[1].plot(results_cv_sorted['Cumulative Perc Population'], results_cv_sorted['Cumulative Perc Good'], label='CV Positive Class (Class 1)', color='green')
    axs[1].plot(results_cv_sorted['Cumulative Perc Population'], results_cv_sorted['Cumulative Perc Bad'], label='CV Negative Class (Class 0)', color='green')
    axs[1].plot([x_max_ks_cv, x_max_ks_cv], [y_min_ks_cv, y_max_ks_cv], color='black', linestyle='--')
    axs[1].fill_between(results_cv_sorted['Cumulative Perc Population'], results_cv_sorted['Cumulative Perc Good'], results_cv_sorted['Cumulative Perc Bad'], color='gray', alpha=0.5)
    axs[1].text(x=x_max_ks_cv,
                y=(y_min_ks_cv + y_max_ks_cv) / 2,
                s=str(KS_cv), fontsize = 14, color='green', ha='left', va='center', rotation=45)
    axs[1].set_xlabel('Cumulative Percentage of Population', fontsize = 14)
    axs[1].set_ylabel('Cumulative Percentage', fontsize = 14)
    axs[1].set_title(f'KS Plot - {classificador}', fontsize = 14)


    # Adicione a legenda personalizada com cores
    ks_legend_labels = [
        {'label': f'Treino (KS: {KS_train})', 'color': 'blue', 'marker': 'o'},
        {'label': f'Teste (KS: {KS_test})', 'color': 'red', 'marker': 's'},
        {'label': f'CV (KS: {KS_cv})', 'color': 'green', 'marker': '^'}
    ]

    # Criar marcadores personalizados para a legenda
    legend_handles = [Line2D([0], [0], marker=label_info['marker'], color='w', markerfacecolor=label_info['color'], markersize=10) for label_info in ks_legend_labels]

    ks_legend = axs[1].legend(handles=legend_handles, labels=[label_info['label'] for label_info in ks_legend_labels], loc='upper right', bbox_to_anchor=(0.9, 0.4), fontsize='11')
    ks_legend.set_title('KS', prop={'size': '11'})

    plt.tight_layout()
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

    # Selecionar as features com importância maior que zero
    selected_features = list(x.columns[feature_importances > threshold])
    selected_features.append(target)

    feature_importance_df = pd.DataFrame({
        'feature': x.columns,
        'importance': feature_importances
    }).sort_values(by = 'importance', ascending = False)
    feature_importance_df = feature_importance_df.loc[feature_importance_df['importance'] > 0]
    feature_importance_df['importance'] = feature_importance_df['importance']*100

    return selected_features, feature_importance_df



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

def ks_test(y_proba_0, y_proba_1):
    KS, p_value = stats.ks_2samp(y_proba_0, y_proba_1)

    if p_value > 0.05:
        ks_message = 'Pelo Teste de KS, não há diferença significativa entre as amostras'
    else:
        ks_message = 'Pelo Teste de KS, há diferença significativa entre as amostras'

    return KS, ks_message

def plot_linear_separability(feature_1, feature_2, x_train, y_train, target):
    # Crie um DataFrame para facilitar a visualização com seaborn
    df = pd.DataFrame(x_train, columns=[feature_1, feature_2])
    df['Target'] = y_train[target]

    # Configure o estilo do seaborn para uma boa estética
    sns.set(style="whitegrid")

    # Plote um gráfico de dispersão com diferentes cores para cada classe
    sns.scatterplot(x=feature_1, y=feature_2, hue='Target', data=df, palette="bright")

    # Adicione uma linha de separação linear (hiperplano)
    plt.title('Linear Separability')
    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
    plt.legend(loc='best')
    plt.show()


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

def metricas_classificacao(classificador, y_train, y_predict_train, y_test, y_predict_test, y_predict_proba_train, y_predict_proba_test):

    predict_proba_train = pd.DataFrame(y_predict_proba_train.tolist(), columns=['predict_proba_0', 'predict_proba_1'])
    predict_proba_test = pd.DataFrame(y_predict_proba_test.tolist(), columns=['predict_proba_0', 'predict_proba_1'])

    # Treino
    accuracy_train = accuracy_score(y_train, y_predict_train)
    precision_train = precision_score(y_train, y_predict_train)
    recall_train = recall_score(y_train, y_predict_train)
    f1_train = f1_score(y_train, y_predict_train)
    roc_auc_train = roc_auc_score(y_train['loan_status'], predict_proba_train['predict_proba_1'])
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train['loan_status'], predict_proba_train['predict_proba_1'])
    ks_train = max(tpr_train - fpr_train)
    metricas_treino = pd.DataFrame({'Acuracia': accuracy_train, 'Precisao': precision_train, 'Recall': recall_train, 'F1-Score': f1_train, 'AUC': roc_auc_train, 'KS': ks_train, 'Etapa': 'treino', 'Classificador': classificador}, index=[0])
    
    # Teste
    accuracy_test = accuracy_score(y_test, y_predict_test)
    precision_test = precision_score(y_test, y_predict_test)
    recall_test = recall_score(y_test, y_predict_test)
    f1_test = f1_score(y_test, y_predict_test)
    roc_auc_test = roc_auc_score(y_test['loan_status'], predict_proba_test['predict_proba_1'])
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test['loan_status'], predict_proba_test['predict_proba_1'])
    ks_test = max(tpr_test - fpr_test)
    metricas_teste = pd.DataFrame({'Acuracia': accuracy_test, 'Precisao': precision_test, 'Recall': recall_test, 'F1-Score': f1_test, 'AUC': roc_auc_test, 'KS': ks_test, 'Etapa': 'teste', 'Classificador': classificador}, index=[0])
    
    # Consolidando
    metricas_finais = pd.concat([metricas_treino, metricas_teste])

    return metricas_finais

def metricas_classificacao_modelos_juntos(lista_modelos):
    metricas_modelos = pd.concat(lista_modelos)#.set_index('Classificador')
    # Redefina o índice para torná-lo exclusivo
    df = metricas_modelos.reset_index(drop=True)
    df = df.round(2)

    # Função para formatar as células com base na Etapa
    def color_etapa(val):
        color = 'black'
        if val == 'treino':
            color = 'blue'
        elif val == 'teste':
            color = 'red'
        return f'color: {color}; font-weight: bold;'

    # Função para formatar os valores com até duas casas decimais
    def format_values(val):
        if isinstance(val, (int, float)):
            return f'{val:.2f}'
        return val

    # Estilizando o DataFrame
    styled_df = df.style\
        .format(format_values)\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: white; font-size: 14px', subset=pd.IndexSlice[:, :])\
        .applymap(color_etapa, subset=pd.IndexSlice[:, :])\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px', subset=pd.IndexSlice[:, 'Acuracia':'F1-Score'])\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px', subset=pd.IndexSlice[:, 'Etapa'])\
        .set_table_styles([
            {'selector': 'thead', 'props': [('color', 'black'), ('font-weight', 'bold'), ('background-color', 'lightgray')]}
        ])

    # Mostrando o DataFrame estilizado
    styled_df
    return styled_df

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

def simple_imputer(df):

    df_aux = df.copy()
    imputer = SimpleImputer(strategy = 'median')
    imputer.fit(df_aux)

    return imputer


def Classificador(classificador, x_train, y_train, x_test, y_test, class_weight):

    def simple_imputer(df):

        df_aux = df.copy()
        imputer = SimpleImputer(strategy = 'median')
        imputer.fit(df_aux)

        return imputer
    
    cols = list(x_train.columns)
    imputer = simple_imputer(x_train)
    x_train = pd.DataFrame(imputer.transform(x_train), columns = x_train.columns)
    x_test = pd.DataFrame(imputer.transform(x_test), columns = x_test.columns)

    # Ajuste automático do class_weight para o MLPClassifier
    if classificador == 'Multilayer Perceptron':
        class_weight = get_class_weight(y_train, target_class=1)

    # Define as colunas categóricas e numéricas
    models = {
        'Regressão Logística': make_pipeline(
            ColumnTransformer([
                ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols),
                ('scaler', make_pipeline(MinMaxScaler()), cols)
            ]),
        LogisticRegression(
            random_state=42, # Semente aleatória para reproducibilidade dos resultados
            class_weight={0: 1, 1: class_weight}, # Peso atribuído às classes. Pode ser útil para lidar com conjuntos de dados desbalanceados.
            C=1, # Parâmetro de regularização inversa. Controla a força da regularização.
            penalty='l2', # Tipo de regularização. 'l1', 'l2', 'elasticnet', ou 'none'.
            max_iter=50, # Número máximo de iterações para a convergência do otimizador.
            solver='liblinear' # Algoritmo de otimização. 'newton-cg', 'lbfgs', 'liblinear' (gradiente descendente), 'sag' (Stochastic gradient descent), 'saga' (Stochastic gradient descent que suporta reg L1).
            )
        ),
        'Naive Bayes': make_pipeline(
            ColumnTransformer([
                ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols)
            ]),
            GaussianNB(priors = [0.88, 0.12]) # Probabilidade a Priori
        ),
        'KNN Classifier': make_pipeline(
            ColumnTransformer([
                ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols),
                ('scaler', make_pipeline(MinMaxScaler()), cols)
            ]),
            KNeighborsClassifier(n_neighbors=5)  # Escolha o número adequado de vizinhos
        ),
        'SVM': make_pipeline(
            ColumnTransformer([
                ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols),
                ('scaler', make_pipeline(MinMaxScaler()), cols)
            ]),
            SVC(kernel='linear', class_weight={0: 1, 1: class_weight}, random_state=42)
        ),
        'Random Forest': make_pipeline(
            ColumnTransformer([
                ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols)
            ]),
        RandomForestClassifier(
            random_state=42,            # Semente aleatória para reproducibilidade dos resultados
            criterion='log_loss',       # Critério usado para medir a qualidade de uma divisão
            n_estimators=50,           # Número de árvores na floresta (equivalente ao n_estimators no XGBoost)
            max_depth = 6,                # Profundidade máxima de cada árvore
            class_weight={0:1, 1:class_weight},  # Peso das classes em casos desequilibrados
            min_samples_split=2,        # O número mínimo de amostras necessárias para dividir um nó interno
            min_samples_leaf=1,         # O número mínimo de amostras necessárias para ser um nó folha
            max_features='auto',        # O número máximo de características a serem consideradas para a melhor divisão
            max_leaf_nodes=None,        # O número máximo de folhas que uma árvore pode ter
            bootstrap=True               # Se deve ou não amostrar com substituição ao construir árvores
            )
        ),
        'XGBoost': make_pipeline(
            ColumnTransformer([
                ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols)
            ]),
        XGBClassifier(
            random_state=42,            # Semente aleatória para reproducibilidade dos resultados
            n_estimators=50,           # Número de árvores no modelo (equivalente ao n_estimators na Random Forest)
            max_depth = 6,                # Profundidade máxima de cada árvore
            learning_rate = 0.04,         # Taxa de aprendizado - controla a contribuição de cada árvore
            eval_metric='logloss',      # Métrica de avaliação durante o treinamento, 'logloss' é comum para problemas de classificação binária
            objective='binary:logistic',# Define o objetivo do modelo, 'binary:logistic' para classificação binária
            scale_pos_weight=class_weight,  # Peso das classes positivas em casos desequilibrados
            reg_alpha=1,                # Termo de regularização L1 (penalidade nos pesos)
            reg_lambda=0,               # Termo de regularização L2 (penalidade nos quadrados dos pesos)
            gamma=1,                    # Controle de poda da árvore, maior gamma leva a menos crescimento da árvore
            colsample_bytree=0.5,       # Fração de características a serem consideradas ao construir cada árvore
            subsample=0.5,              # Fração de amostras a serem usadas para treinar cada árvore
            base_score=0.5              # Threshold de Probabilidade de Decisão do Classificador (geralmente é 0.5 para problemas de classificação binária)
            )
        ),
        'Multilayer Perceptron': make_pipeline(
            ColumnTransformer([
                ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols),
                ('scaler', make_pipeline(MinMaxScaler()), cols)
            ]),
        MLPClassifier(
            hidden_layer_sizes=(100, ),  # Número de camadas ocultas
            activation='relu',           # Função de ativação
            solver='adam',               # Otimizador
            max_iter=200,                # Número máximo de iterações
            random_state=42,
            class_weight={0: 1, 1: class_weight},  # Peso para classe minoritária
            alpha=0.0001,  # Exemplo de ajuste do termo de regularização
            learning_rate='adaptive'  # Exemplo de ajuste da taxa de aprendizado
        )
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


def validacao_cruzada_classificacao(classificador, df, target_column, n_splits, class_weight):

    def numero_de_anos_emprego_atual(df):
        df['emp_length'] = (df['emp_length'].replace({'< 1 year':0, '1 year':1, '2 years':2, '3 years':3, '4 years':4, '5 years':5, '6 years':6, '7 years':7, '8 years':8, '9 years':9,'10+ years':10}).fillna(0))
        df['emp_length'] = df['emp_length'].apply(lambda x:int(x))
        df['emp_length'] = np.where(df['emp_length'] <= 3, '3_YEARS', 
                            np.where(df['emp_length'] <= 6, '6_YEARS',
                            np.where(df['emp_length'] <= 9, '9_YEARS',
                            '10_YEARS+')))
        return df['emp_length']

    def numero_de_registros_negativos(df):

        df = df[['loan_status', 'pub_rec']].copy()
        df[['pub_rec']] = np.where(df[['pub_rec']] == 0, 'sem_registros_negativos', 'com_registros_negativos')

        return df['pub_rec']

    def consulta_de_credito_nos_ultimos_6_meses(df):
        df = df[['loan_status', 'inq_last_6mths']].copy()
        df[['inq_last_6mths']] = np.where(df[['inq_last_6mths']] == 0, 'sem_consultas', 'com_consultas')

        return df['inq_last_6mths']

    def compromento_de_renda(df): 
        df_aux = df[['annual_inc', 'installment', 'loan_amnt', 'term', 'int_rate', 'loan_status']].copy()
        df_aux['term'] = np.where(df_aux['term'] == ' 36 months', 36, 60)
        df_aux['loan_amnt_with_int_rate'] = df_aux['installment']*df_aux['term']
        df_aux['annual_payment'] = np.where(df_aux['term'] == ' 36 months', df_aux['loan_amnt_with_int_rate']/3, df_aux['loan_amnt_with_int_rate']/5)
        df_aux['annual_income_commitment_rate'] = ((df_aux['annual_payment']/df_aux['annual_inc'])*100).round(2)
        
        return df_aux['annual_income_commitment_rate']

    def numero_incidencias_inadimplencia_vencidas_30d(df):
        df_aux = df[['loan_status', 'delinq_2yrs']].copy()
        df_aux['delinq_2yrs'] = np.where(df_aux[['delinq_2yrs']] == 0, 'sem_inadimplencia_vencida', 'com_inadimplencia_vencida')

        return df_aux['delinq_2yrs']

    def n_meses_produto_credito_atual(df):
        df = df.copy()
        df['issue_d'] = pd.to_datetime(df['issue_d'], format = '%b-%y')
        df['mths_since_issue_d'] = round(pd.to_numeric((pd.to_datetime('2023-09-20') - df['issue_d'])/np.timedelta64(1, 'M')))
        df['mths_since_issue_d'] = df['mths_since_issue_d'].fillna(df['mths_since_issue_d'].median())
        df['mths_since_issue_d'] = np.where(df['mths_since_issue_d'] < 0, df['mths_since_issue_d'].median(), df['mths_since_issue_d'])
        df['mths_since_issue_d'] = df['mths_since_issue_d'].apply(lambda x:int(x))
        df['issue_d'] = df['mths_since_issue_d']

        return df['issue_d']

    def n_meses_primeiro_produto_credito(df):
        df = df.copy()
        df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format = '%b-%y')
        df['mths_since_earliest_cr_line'] = round(pd.to_numeric((pd.to_datetime('2023-09-20') - df['earliest_cr_line'])/np.timedelta64(1, 'M')))
        df['mths_since_earliest_cr_line'] = df['mths_since_earliest_cr_line'].fillna(df['mths_since_earliest_cr_line'].median())
        df['mths_since_earliest_cr_line'] = np.where(df['mths_since_earliest_cr_line'] < 0, df['mths_since_earliest_cr_line'].median(), df['mths_since_earliest_cr_line'])
        df['mths_since_earliest_cr_line'] = df['mths_since_earliest_cr_line'].apply(lambda x:int(x))
        df['earliest_cr_line'] = df['mths_since_earliest_cr_line']
        
        return df['earliest_cr_line']

    def formato_features_binarias(df):
        df['term'] = np.where(df['term'] == ' 36 months', 0, 1)
        df['delinq_2yrs'] = np.where(df['delinq_2yrs'] == 'sem_inadimplencia_vencida', 0, 1)
        df['initial_list_status'] = np.where(df['initial_list_status'] == 'f', 0, 1)
        df['pymnt_plan'] = np.where(df['pymnt_plan'] == 'n', 0, 1)
        df['verification_status'] = np.where(df['verification_status'] == 'Source Verified', 0, 1)
        df['inq_last_6mths'] = np.where(df['inq_last_6mths'] == 'com_consultas', 0, 1)

        return df

    def taxa_de_bad_por_categoria(df, tipo):
        categoricas = ['term', 'grade', 'sub_grade', 'purpose', 'policy_code', 'initial_list_status', 'pymnt_plan', 'emp_length', 'home_ownership', 'verification_status', 'addr_state', 'pub_rec', 'inq_last_6mths']
        df_aux_2 = df.copy()
        if tipo == 'Criação':
            for cat in categoricas:
                df_aux = df[[f'{cat}', 'loan_status']].copy()
                good = pd.DataFrame(df_aux.loc[df_aux['loan_status'] == 0].groupby(f'{cat}', as_index = False)['loan_status'].count()).rename({'loan_status':'qt_good'}, axis = 1)
                bad = pd.DataFrame(df_aux.loc[df_aux['loan_status'] == 1].groupby(f'{cat}', as_index = False)['loan_status'].count()).rename({'loan_status':'qt_bad'}, axis = 1)
                df_aux = good.merge(bad, on = f'{cat}', how = 'left')
                df_aux['qt_total'] = df_aux['qt_good'] + df_aux['qt_bad']
                df_aux[f'bad_rate_{cat}'] = ((df_aux['qt_bad']/df_aux['qt_total'])*100).round(2)
                df_aux[f'bad_rate_{cat}'] = df_aux[f'bad_rate_{cat}'].apply(lambda x:float(x))
                df_aux = df_aux[[f'{cat}', f'bad_rate_{cat}']].drop_duplicates().sort_values(by = f'bad_rate_{cat}', ascending = True)
                df_aux.to_csv(f'features/bad_rate_{cat}.csv', index = False)
                df_aux_2 = df_aux_2.merge(df_aux[[f'{cat}', f'bad_rate_{cat}']], on = f'{cat}', how = 'left')
        else:
            for cat in categoricas:
                ft = pd.read_csv(f'features/bad_rate_{cat}.csv')
                replace_dict = dict(zip(ft[f'{cat}'], ft[f'bad_rate_{cat}']))
                df_aux_2[f'bad_rate_{cat}'] = df_aux_2[f'{cat}'].replace(replace_dict)

        return df_aux_2

    def media_categoria_variavel_quantitativa(df, tipo):
        df_aux_2 = df.copy()
        categoricas = ['term', 'grade', 'sub_grade', 'purpose', 'delinq_2yrs', 'policy_code', 'initial_list_status', 'pymnt_plan', 'emp_length', 'home_ownership', 'verification_status', 'addr_state', 'pub_rec', 'inq_last_6mths']
        quantitativas = ['loan_amnt', 'int_rate', 'annual_inc', 'annual_income_commitment_rate', 'tot_cur_bal', 'total_rev_hi_lim', 'revol_util', 'open_acc', 'total_acc']
        if tipo == 'Criação':
            for cat in categoricas:
                for quant in quantitativas:
                    df_aux = df[[f'{cat}', f'{quant}']].copy()
                    df_aux = pd.DataFrame(df_aux.groupby(f'{cat}', as_index = False)[f'{quant}'].mean()).rename({f'{quant}':f'mean_{cat}_{quant}'}, axis = 1)
                    df_aux[f'mean_{cat}_{quant}'] = df_aux[f'mean_{cat}_{quant}'].apply(lambda x:float(x))
                    df_aux[[f'{cat}', f'mean_{cat}_{quant}']].drop_duplicates().sort_values(by = f'mean_{cat}_{quant}', ascending = True)
                    df_aux.to_csv(f'features/mean_{cat}_{quant}.csv', index = False)
                    df_aux_2 = df_aux_2.merge(df_aux[[f'{cat}', f'mean_{cat}_{quant}']], on = f'{cat}', how = 'left')
        else:
            for cat in categoricas:
                for quant in quantitativas:
                    ft = pd.read_csv(f'features/mean_{cat}_{quant}.csv')
                    replace_dict = dict(zip(ft[f'{cat}'], ft[f'mean_{cat}_{quant}']))
                    df_aux_2[f'mean_{cat}_{quant}'] = df_aux_2[f'{cat}'].replace(replace_dict)

        return df_aux_2


    def simple_imputer(df):

        df_aux = df.copy()
        imputer = SimpleImputer(strategy = 'median')
        imputer.fit(df_aux)

        return imputer

    columns_selected = ['loan_status', 'term','grade','sub_grade','purpose', 'delinq_2yrs', 'loan_amnt','int_rate','issue_d','policy_code','pymnt_plan','initial_list_status','installment','emp_length','home_ownership',
    'verification_status','annual_inc','addr_state', 'tot_cur_bal','total_rev_hi_lim','revol_bal','revol_util','open_acc','total_acc','pub_rec','inq_last_6mths','earliest_cr_line','mths_since_last_record', 'mths_since_last_major_derog',
    'mths_since_last_delinq']

    df_raw = df[columns_selected].copy()

    # Feature Selection

    features_selected = pd.read_csv('features/features_selected.csv')
    features_selected = features_selected.loc[features_selected['importance'] > 1] 
    features_selected = list(features_selected['feature'].unique()) + ['loan_status']

    # Inicializar o KFold para dividir os dados
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Listas para armazenar as métricas para cada fold
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    auc_scores = []  # Lista para armazenar os valores de AUC
    ks_scores = []   # Lista para armazenar os valores de KS
    cv_results = []  # Lista para armazenar os resultados de validação cruzada

    # Loop pelos folds
    for train_idx, test_idx in kfold.split(df_raw):
        # Criar DataFrames de treino e teste
        df_train = df_raw.iloc[train_idx]
        df_test = df_raw.iloc[test_idx]

        # Criação das Features sem Data Leakage
        df_train['emp_length'] = numero_de_anos_emprego_atual(df_train)
        df_train['pub_rec'] = numero_de_registros_negativos(df_train)
        df_train['inq_last_6mths'] = consulta_de_credito_nos_ultimos_6_meses(df_train)
        df_train['annual_income_commitment_rate'] = compromento_de_renda(df_train)
        df_train['delinq_2yrs'] = numero_incidencias_inadimplencia_vencidas_30d(df_train)
        df_train['issue_d'] = n_meses_produto_credito_atual(df_train)
        df_train['earliest_cr_line'] = n_meses_primeiro_produto_credito(df_train)
        df_train = formato_features_binarias(df_train)
        df_train = taxa_de_bad_por_categoria(df_train, 'escoragem')
        df_train = media_categoria_variavel_quantitativa(df_train, 'escoragem')

        df_test['emp_length'] = numero_de_anos_emprego_atual(df_test)
        df_test['pub_rec'] = numero_de_registros_negativos(df_test)
        df_test['inq_last_6mths'] = consulta_de_credito_nos_ultimos_6_meses(df_test)
        df_test['annual_income_commitment_rate'] = compromento_de_renda(df_test)
        df_test['delinq_2yrs'] = numero_incidencias_inadimplencia_vencidas_30d(df_test)
        df_test['issue_d'] = n_meses_produto_credito_atual(df_test)
        df_test['earliest_cr_line'] = n_meses_primeiro_produto_credito(df_test)
        df_test = formato_features_binarias(df_test)
        df_test = taxa_de_bad_por_categoria(df_test, 'escoragem')
        df_test = media_categoria_variavel_quantitativa(df_test, 'escoragem')

        # Filtragem das Features que passaram no Feature Selection
        df_train = df_train[features_selected]
        df_test = df_test[features_selected]

        # Separação Feature e Target
        x_train, y_train = separa_feature_target('loan_status', df_train)
        x_test, y_test = separa_feature_target('loan_status', df_test)
        
        # Imputer
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
                LogisticRegression(
                    random_state=42, # Semente aleatória para reproducibilidade dos resultados
                    class_weight={0: 1, 1: class_weight}, # Peso atribuído às classes. Pode ser útil para lidar com conjuntos de dados desbalanceados.
                    C=2, # Parâmetro de regularização inversa. Controla a força da regularização.
                    penalty='l2', # Tipo de regularização. 'l1', 'l2', 'elasticnet', ou 'none'.
                    max_iter=50, # Número máximo de iterações para a convergência do otimizador.
                    solver='liblinear' # Algoritmo de otimização. 'newton-cg', 'lbfgs', 'liblinear' (gradiente descendente), 'sag' (Stochastic gradient descent), 'saga' (Stochastic gradient descent que suporta reg L1).
                )
            ),
            'Random Forest': make_pipeline(
                ColumnTransformer([
                    ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols)
                ]),
                RandomForestClassifier(
                    random_state=42,            # Semente aleatória para reproducibilidade dos resultados
                    criterion='log_loss',       # Critério usado para medir a qualidade de uma divisão
                    n_estimators=50,           # Número de árvores na floresta (equivalente ao n_estimators no XGBoost)
                    max_depth = 6,                # Profundidade máxima de cada árvore
                    class_weight={0:1, 1:class_weight},  # Peso das classes em casos desequilibrados
                    min_samples_split=2,        # O número mínimo de amostras necessárias para dividir um nó interno
                    min_samples_leaf=1,         # O número mínimo de amostras necessárias para ser um nó folha
                    max_features='auto',        # O número máximo de características a serem consideradas para a melhor divisão
                    max_leaf_nodes=None,        # O número máximo de folhas que uma árvore pode ter
                    bootstrap=True               # Se deve ou não amostrar com substituição ao construir árvores
                )
            ),
            'XGBoost': make_pipeline(
                ColumnTransformer([
                    ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols)
                ]),
                XGBClassifier(
                    random_state=42,            # Semente aleatória para reproducibilidade dos resultados
                    n_estimators=50,           # Número de árvores no modelo (equivalente ao n_estimators na Random Forest)
                    max_depth = 6,                # Profundidade máxima de cada árvore
                    learning_rate = 0.04,         # Taxa de aprendizado - controla a contribuição de cada árvore
                    eval_metric='logloss',      # Métrica de avaliação durante o treinamento, 'logloss' é comum para problemas de classificação binária
                    objective='binary:logistic',# Define o objetivo do modelo, 'binary:logistic' para classificação binária
                    scale_pos_weight=class_weight,  # Peso das classes positivas em casos desequilibrados
                    reg_alpha=1,                # Termo de regularização L1 (penalidade nos pesos)
                    reg_lambda=0,               # Termo de regularização L2 (penalidade nos quadrados dos pesos)
                    gamma=1,                    # Controle de poda da árvore, maior gamma leva a menos crescimento da árvore
                    colsample_bytree=0.5,       # Fração de características a serem consideradas ao construir cada árvore
                    subsample=0.5,              # Fração de amostras a serem usadas para treinar cada árvore
                    base_score=0.5              # Threshold de Probabilidade de Decisão do Classificador (geralmente é 0.5 para problemas de classificação binária)
                )
            )
        }

        if classificador in models:
            model = models[classificador]
        else:
            print('Utilize Regressão Logística, Random Forest ou XGBoost como opções de Classificadores!')

        # Treinar o modelo usando os dados de treinamento
        model.fit(x_train, y_train)

        # Obter as probabilidades previstas para ambas as classes
        y_proba = model.predict_proba(x_test)

        # Fazer as previsões usando o modelo nos dados de teste
        y_pred = model.predict(x_test)

        # Calcular as métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        y_proba = model.predict_proba(x_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        ks = max(tpr - fpr)

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        auc_scores.append(roc_auc)
        ks_scores.append(ks)

        # Adicionar resultados de validação cruzada ao DataFrame
        fold_results = pd.DataFrame({
            'loan_status': y_test['loan_status'].values,
            'y_predict': y_pred,
            'predict_proba_0': y_proba[:, 0],  # Probabilidade da classe 0
            'predict_proba_1': y_proba[:, 1]  # Probabilidade da classe 1
        })
        cv_results.append(fold_results)


    # Calcular a média das métricas para todos os folds
    mean_accuracy = np.mean(accuracy_scores)
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_f1 = np.mean(f1_scores)
    mean_auc = np.mean(auc_scores),
    mean_ks = np.mean(ks_scores)

    # Criar um DataFrame com as métricas
    metricas_finais = pd.DataFrame({
        'Acuracia': mean_accuracy,
        'Precisao': mean_precision,
        'Recall': mean_recall,
        'F1-Score': mean_f1,
        'AUC':mean_auc,
        'KS': mean_ks,
        'Etapa': 'validacao_cruzada',
        'Classificador': classificador
    }, index=[1])

    return metricas_finais, cv_results


def modelo_otimizado(classificador, x_train, y_train, x_test, y_test):
    def simple_imputer(df):

        df_aux = df.copy()
        imputer = SimpleImputer(strategy = 'median')
        imputer.fit(df_aux)

        return imputer
    
    cols = list(x_train.columns)
    imputer = simple_imputer(x_train)
    x_train = pd.DataFrame(imputer.transform(x_train), columns = x_train.columns)
    x_test = pd.DataFrame(imputer.transform(x_test), columns = x_test.columns)

    # Define o ColumnTransformer
    preprocessor = ColumnTransformer([
                ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols),
                ('scaler', make_pipeline(MinMaxScaler()), cols)
            ])

    # Define o modelo de XGBoost com a otimização de hiperparâmetros via BayesSearch
    model = make_pipeline(
        preprocessor,
        BayesSearchCV(
            XGBClassifier(random_state=42, eval_metric='logloss', objective='binary:logistic'),
            {
                'n_estimators': (10, 15, 20, 50), # Número de Árvores construídas
                'max_depth': (4, 5, 7, 8, 9, 10), # Profundidade Máxima de cada Árvore
                'learning_rate': (0.01, 0.05), # Tamanho do passo utilizado no Método do Gradiente Descendente
                'reg_alpha':(0.5, 1), # Valor do Alpha aplicado durante a Regularização Lasso L1 
                'reg_lambda':(0.5, 1), # Valor do Lambda aplicado durante a Regularização Ridge L2
                'gamma':(0.5, 1), # Valor mínimo permitido para um Nó de Árvore ser aceito. Ajuda a controlar o crescimento das Árvores, evitando divisões insignificantes
                'colsample_bytree':(0.5, 1), # Porcentagem de Colunas utilizada para a amostragem aleatória durante a criação das Árvores
                'subsample':(0.5, 1), # Porcentagem de Linhas utilizada para a amostragem aleatória durante a criação das Árvores
                'scale_pos_weight':(3, 5, 8, 10, 12, 14), # Peso atribuído a classe positiva, aumentando a importância da classe minoritária
                'base_score':(0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
            },
            n_iter=10,
            random_state=42,
            n_jobs=-1,
            scoring='roc_auc',
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
