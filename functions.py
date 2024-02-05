## Bibliotecas De Manipulação de Dados e Visualização
import pandas as pd 
import builtins as builtins
import matplotlib.pyplot as plt
import seaborn as sns 
from IPython.display import display, Image
from tabulate import tabulate
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

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
from hyperopt import fmin, tpe, hp, Trials
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate, cross_val_predict
from sklearn.feature_selection import VarianceThreshold, chi2, mutual_info_classif

# Bibliotecas de Pré-Processamento e Pipeline
from category_encoders import BinaryEncoder
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer

# Bibliotecas de Modelos de Machine Learning
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# Bibliotecas de Métricas de Machine Learning
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, precision_score, recall_score, precision_recall_curve, average_precision_score, f1_score, log_loss, brier_score_loss, confusion_matrix, silhouette_score

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
        {'label': 'CV ROC Curve (AUC = {:.2f})'.format(auc_test), 'color': 'green', 'marker': '^'}
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

def auc_ks_final(classificador, target, y, y_predict, y_predict_proba):

    predict_proba = pd.DataFrame(y_predict_proba.tolist(), columns=['predict_proba_0', 'predict_proba_1'])

    # Inicialize as variáveis x_max_ks e y_max_ks fora dos blocos condicionais
    x_max_ks, y_max_ks = 0, 0

    ### Etapa Final
    results = y[[target]].copy()
    results['y_predict'] = y_predict
    results['predict_proba_0'] = list(predict_proba['predict_proba_0']) # Probabilidade de ser Bom (classe 0)
    results['predict_proba_1'] = list(predict_proba['predict_proba_1']) # Probabilidade de ser Mau (classe 1)

    results_sorted = results.sort_values(by='predict_proba_1', ascending=False)
    results_sorted['Cumulative N Population'] = range(1, results_sorted.shape[0] + 1)
    results_sorted['Cumulative N Good'] = results_sorted[target].cumsum()
    results_sorted['Cumulative N Bad'] = results_sorted['Cumulative N Population'] - results_sorted['Cumulative N Good']
    results_sorted['Cumulative Perc Population'] = results_sorted['Cumulative N Population'] / results_sorted.shape[0]
    results_sorted['Cumulative Perc Good'] = results_sorted['Cumulative N Good'] / results_sorted[target].sum()
    results_sorted['Cumulative Perc Bad'] = results_sorted['Cumulative N Bad'] / (results_sorted.shape[0] - results_sorted[target].sum())

    max_ks_index = np.argmax(results_sorted['Cumulative Perc Good'] - results_sorted['Cumulative Perc Bad'])
    x_max_ks = results_sorted['Cumulative Perc Population'].iloc[max_ks_index]
    y_max_ks = results_sorted['Cumulative Perc Good'].iloc[max_ks_index]
    y_min_ks = results_sorted['Cumulative Perc Bad'].iloc[max_ks_index]

        ###### Calculate AUC and ROC for the training set
    y_true = results[target]
    y_scores = results['predict_proba_1']
    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        ###### Calculate KS curve for the training set
    KS = round(np.max(results_sorted['Cumulative Perc Good'] - results_sorted['Cumulative Perc Bad']), 2)

    # Plot ROC and KS curves side by side
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Training set ROC curve
    axs[0].plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(auc), color='blue')
    axs[0].fill_between(fpr, 0, tpr, color='gray', alpha=0.3)  # Preencha a área sob a curva ROC
    axs[0].plot([0, 1], [0, 1], linestyle='--', color='black')
    axs[0].set_xlabel('False Positive Rate', fontsize = 14)
    axs[0].set_ylabel('True Positive Rate', fontsize = 14)
    axs[0].set_title(f'ROC Curve - {classificador}', fontsize = 14)

    # Adicione a legenda personalizada com cores para a curva ROC
    roc_legend_labels = [
        {'label': 'ROC Curve (AUC = {:.2f})'.format(auc), 'color': 'blue', 'marker': 'o'},
    ]

    # Criar marcadores personalizados para a legenda ROC
    roc_legend_handles = [Line2D([0], [0], marker=label_info['marker'], color='w', markerfacecolor=label_info['color'], markersize=10) for label_info in roc_legend_labels]

    # Adicione a legenda personalizada ao gráfico da curva ROC
    roc_legend = axs[0].legend(handles=roc_legend_handles, labels=[label_info['label'] for label_info in roc_legend_labels], loc='upper right', bbox_to_anchor=(0.9, 0.4), fontsize='11')
    roc_legend.set_title('ROC AUC', prop={'size': '11'})


    # Train set KS curve
    axs[1].plot(results_sorted['Cumulative Perc Population'], results_sorted['Cumulative Perc Good'], label='Train Positive Class (Class 1)', color='blue')
    axs[1].plot(results_sorted['Cumulative Perc Population'], results_sorted['Cumulative Perc Bad'], label='Train Negative Class (Class 0)', color='blue')
    axs[1].plot([x_max_ks, x_max_ks], [y_min_ks, y_max_ks], color='black', linestyle='--')
    axs[1].fill_between(results_sorted['Cumulative Perc Population'], results_sorted['Cumulative Perc Good'], results_sorted['Cumulative Perc Bad'], color='gray', alpha=0.5)
    axs[1].text(x=results_sorted['Cumulative Perc Population'].iloc[np.argmax(results_sorted['Cumulative Perc Good'] - results_sorted['Cumulative Perc Bad'])],
                y=(y_min_ks + results_sorted['Cumulative Perc Good'].iloc[np.argmax(results_sorted['Cumulative Perc Good'] - results_sorted['Cumulative Perc Bad'])]) / 2,
                s=str(KS), fontsize = 14, color='blue', ha='left', va='center', rotation=45)
    axs[1].set_xlabel('Cumulative Percentage of Population', fontsize = 14)
    axs[1].set_ylabel('Cumulative Percentage', fontsize = 14)
    axs[1].set_title(f'KS Plot - {classificador}', fontsize = 14)

    # Adicione a legenda personalizada com cores
    ks_legend_labels = [
        {'label': f'(KS: {KS})', 'color': 'blue', 'marker': 'o'},
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
    model = RandomForestClassifier(random_state=42, criterion='entropy', n_estimators=20, class_weight={0:1, 1:class_weight})

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

    woe.sort_values(by = 'woe', ascending = True, inplace = True)

    weight_of_evidence = woe['woe'].unique()
    iv = round(woe['iv'].max(), 2)

    x = list(df[feature].unique())
    x.sort()
    y = list(woe['woe'].values)
    plt.figure(figsize = (10, 4))
    plt.plot(x, y, marker = 'o', linestyle = '--', linewidth=2, color = '#1FB3E5')
    for label, value in zip(x, y):
        plt.text(x = label, y = value, s = str(value), fontsize = 20, color = 'red', ha='left', va='center', rotation = 45)
    # plt.title(f'WOE of "{feature}" with an Information Value {iv} ', fontsize=14)
    plt.title(f'Weight of Evidence da variável "{feature}"', fontsize=14)
    plt.xlabel('Classes', fontsize=14)
    plt.ylabel('Weight of Evidence', fontsize=14)
    plt.xticks(ha='right', fontsize = 10, rotation = 45)

def iv(df, feature, target):
    good = df.loc[df[target] == 0].groupby(feature, as_index = False)[target].count().rename({target:'good'}, axis = 1)
    bad = df.loc[df[target] == 1].groupby(feature, as_index = False)[target].count().rename({target:'bad'}, axis = 1)

    woe = good.merge(bad, on = feature, how = 'left')
    woe['percent_good'] = woe['good']/woe['good'].sum()
    woe['percent_bad'] = woe['bad']/woe['bad'].sum()
    woe['woe'] = round(np.log(woe['percent_good']/woe['percent_bad']), 3)
    woe['iv'] = ((woe['percent_good'] - woe['percent_bad'])*np.log(woe['percent_good']/woe['percent_bad'])).sum()

    woe['woe'].fillna(0, inplace = True)
    woe['iv'].fillna(0, inplace = True)

    weight_of_evidence = woe['woe'].unique()
    iv = round(woe['iv'].max(), 2)

    dicionario = {feature:iv}

    iv_df = pd.DataFrame(list(dicionario.items()), columns=['Feature', 'IV'])
    
    return iv_df

def metricas_classificacao(classificador, y_train, y_predict_train, y_test, y_predict_test, y_predict_proba_train, y_predict_proba_test):

    predict_proba_train = pd.DataFrame(y_predict_proba_train.tolist(), columns=['predict_proba_0', 'predict_proba_1'])
    predict_proba_test = pd.DataFrame(y_predict_proba_test.tolist(), columns=['predict_proba_0', 'predict_proba_1'])

    # Treino
    accuracy_train = accuracy_score(y_train, y_predict_train)
    precision_train = precision_score(y_train, y_predict_train)
    recall_train = recall_score(y_train, y_predict_train)
    f1_train = f1_score(y_train, y_predict_train)
    roc_auc_train = roc_auc_score(y_train['situacao_do_emprestimo'], predict_proba_train['predict_proba_1'])
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train['situacao_do_emprestimo'], predict_proba_train['predict_proba_1'])
    ks_train = max(tpr_train - fpr_train)
    metricas_treino = pd.DataFrame({'Acuracia': accuracy_train, 'Precisao': precision_train, 'Recall': recall_train, 'F1-Score': f1_train, 'AUC': roc_auc_train, 'KS': ks_train, 'Etapa': 'treino', 'Classificador': classificador}, index=[0])
    
    # Teste
    accuracy_test = accuracy_score(y_test, y_predict_test)
    precision_test = precision_score(y_test, y_predict_test)
    recall_test = recall_score(y_test, y_predict_test)
    f1_test = f1_score(y_test, y_predict_test)
    roc_auc_test = roc_auc_score(y_test['situacao_do_emprestimo'], predict_proba_test['predict_proba_1'])
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test['situacao_do_emprestimo'], predict_proba_test['predict_proba_1'])
    ks_test = max(tpr_test - fpr_test)
    metricas_teste = pd.DataFrame({'Acuracia': accuracy_test, 'Precisao': precision_test, 'Recall': recall_test, 'F1-Score': f1_test, 'AUC': roc_auc_test, 'KS': ks_test, 'Etapa': 'teste', 'Classificador': classificador}, index=[0])
    
    # Consolidando
    metricas_finais = pd.concat([metricas_treino, metricas_teste])

    return metricas_finais

def metricas_classificacao_modelos_juntos(lista_modelos):
    if len(lista_modelos) > 0:
        metricas_modelos = pd.concat(lista_modelos)#.set_index('Classificador')
    else:
        metricas_modelos = lista_modelos[0]
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

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

def metricas_classificacao_final(classificador, df, y, y_predict, y_predict_proba):

    predict_proba = pd.DataFrame(y_predict_proba.tolist(), columns=['predict_proba_0', 'predict_proba_1'])

    # Amostra Final
    accuracy = accuracy_score(y, y_predict)
    precision = precision_score(y, y_predict)
    recall = recall_score(y, y_predict)
    f1 = f1_score(y, y_predict)
    roc_auc = roc_auc_score(y['situacao_do_emprestimo'], predict_proba['predict_proba_1'])
    fpr, tpr, thresholds = roc_curve(y['situacao_do_emprestimo'], predict_proba['predict_proba_1'])
    ks = max(tpr - fpr)
    total, retorno_financeiro_por_caso, valor_de_exposicao_total, return_on_portfolio = retorno_financeiro(df, y_predict)
    total = 'R$' + str(int(round(total/1000000, 0))) + ' MM'
    valor_de_exposicao_total = 'R$' + str(float(round(valor_de_exposicao_total/1000000000, 3))) + 'B'
    rocp = str(return_on_portfolio) + '%'
    metricas_finais = pd.DataFrame({
        # 'Acuracia': accuracy, 
        # 'Precisao': precision, 
        # 'Recall': recall, 
        # 'F1-Score': f1, 
        # 'AUC': roc_auc, 
        # 'KS': ks, 
        'Etapa': 'Amostra Final', 
        'Método': classificador, 
        'Valor Total de Exposição': valor_de_exposicao_total,
        'Retorno Financeiro': total,
        'Return on Credit Portfolio (ROCP)': rocp
    }, index=[0])

    df = metricas_finais.reset_index(drop=True)
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
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: white; font-size: 14px', subset=pd.IndexSlice[:, 'Etapa':])\
        .applymap(color_etapa, subset=pd.IndexSlice[:, 'Etapa':])\
        .set_table_styles([
            {'selector': 'thead', 'props': [('color', 'black'), ('font-weight', 'bold'), ('background-color', 'lightgray')]}
        ])

    # Mostrando o DataFrame estilizado
    return styled_df


# Exemplo de chamada da função
# Substitua os argumentos pelos seus dados reais
# result = metricas_classificacao_final(classificador, df, y, y_predict, y_predict_proba)
# result.show()  # Certifique-se de ajustar conforme necessário, dependendo do ambiente em que você está executando o código.

def retorno_financeiro(df, y_predict):

    df_aux = df.copy()
    df_aux['qt_parcelas'] = np.where(df_aux['qt_parcelas'] == ' 36 months', 36, 60)
    df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'] = df_aux['qt_parcelas'] * df_aux['pagamento_mensal']
    df_aux['y_predict'] = y_predict

    TN = df_aux.loc[(df_aux['situacao_do_emprestimo'] == 0) & (df_aux['y_predict'] == 0)].shape[0] # O CARA É BOM E MEU MODELO FALA QUE ELE É BOM
    FN = df_aux.loc[(df_aux['situacao_do_emprestimo'] == 1) & (df_aux['y_predict'] == 0)].shape[0] # O CARA É MAU E MEU MODELO FALA QUE ELE É BOM
    FP = df_aux.loc[(df_aux['situacao_do_emprestimo'] == 0) & (df_aux['y_predict'] == 1)].shape[0] # O CARA É BOM E MEU MODELO FALA QUE É MAU
    TP = df_aux.loc[(df_aux['situacao_do_emprestimo'] == 1) & (df_aux['y_predict'] == 1)].shape[0] # O CARA É MAU E O MEU MODELO FALA QUE É MAU

    df_aux['caso'] = np.where((df_aux['situacao_do_emprestimo'] == 0) & (df_aux['y_predict'] == 0), 'Verdadeiro Negativo (Cliente Bom | Modelo classifica como Bom) - Ganho a Diferença entre Valor Bruto e Valor com Juros', # Ganha a Diferença entre Valor Bruto e Valor com Juros
                        np.where((df_aux['situacao_do_emprestimo'] == 1) & (df_aux['y_predict'] == 0), 'Falso Negativo (Cliente Mau | Modelo classifica como Bom) - Perco o valor emprestado', # Perde o valor emprestado
                        np.where((df_aux['situacao_do_emprestimo'] == 0) & (df_aux['y_predict'] == 1), 'Falso Positivo (Cliente Bom | Modelo classifica como Mau) - Deixo de ganhar a diferença entre Valor Bruto e Valor com Juros', # Deixo de ganhar a diferença entre Valor Bruto e Valor com Juros
                        'Verdadeiro Positivo (Cliente Mau | Modelo classifica como Mau) - Não ganho nada' # Não ganho nada
    )))

    df_aux['retorno_financeiro'] = np.where((df_aux['situacao_do_emprestimo'] == 0) & (df_aux['y_predict'] == 0), df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'] - df_aux['valor_emprestimo_solicitado'], # Ganha a Diferença entre Valor Bruto e Valor com Juros
                        np.where((df_aux['situacao_do_emprestimo'] == 1) & (df_aux['y_predict'] == 0), df_aux['valor_emprestimo_solicitado']*(-1), # Perde o valor emprestado
                        np.where((df_aux['situacao_do_emprestimo'] == 0) & (df_aux['y_predict'] == 1), 0, # Deixo de ganhar a diferença entre Valor Bruto e Valor com Juros (df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'] - df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'])*(-1)
                        0 # Não ganho nada
    )))

    valor_de_exposicao_total = int(df_aux['valor_emprestimo_solicitado'].sum())
    retorno_financeiro = int(df_aux['retorno_financeiro'].sum())
    valor_conquistado = valor_de_exposicao_total + retorno_financeiro
    return_on_portfolio = round((retorno_financeiro/valor_de_exposicao_total)*100, 2)
    retorno_financeiro_por_caso = df_aux.groupby('caso', as_index = False)['retorno_financeiro'].sum().sort_values(by = 'retorno_financeiro', ascending = False)

    # Crie um DataFrame a partir dos hiperparâmetros
    df = retorno_financeiro_por_caso.reset_index(drop=True)
    df = df.round(2)

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
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: white; font-size: 14px')\
        .applymap(color_etapa, subset=pd.IndexSlice[:, :])\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px')\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px')\
        .set_table_styles([
            {'selector': 'thead', 'props': [('color', 'black'), ('font-weight', 'bold'), ('background-color', 'lightgray')]}
        ])

    return retorno_financeiro, styled_df, valor_de_exposicao_total, return_on_portfolio


# def retorno_financeiro(df, target, y, y_predict):

#     df_aux = df.copy()

#     TN = df_aux.loc[(df_aux['y_true'] == 0) & (df_aux['y_predict'] == 0)].shape[0] # O CARA É BOM E MEU MODELO FALA QUE ELE É BOM
#     FN = df_aux.loc[(df_aux['y_true'] == 1) & (df_aux['y_predict'] == 0)].shape[0] # O CARA É MAU E MEU MODELO FALA QUE ELE É BOM
#     FP = df_aux.loc[(df_aux['y_true'] == 0) & (df_aux['y_predict'] == 1)].shape[0] # O CARA É BOM E MEU MODELO FALA QUE É MAU
#     TP = df_aux.loc[(df_aux['y_true'] == 1) & (df_aux['y_predict'] == 1)].shape[0] # O CARA É MAU E O MEU MODELO FALA QUE É MAU

#     df_aux['caso'] = np.where((df_aux['y_true'] == 0) & (df_aux['y_predict'] == 0), 'Verdadeiro Negativo (Cliente Bom | Modelo classifica como Bom) - Ganho a Diferença entre Valor Bruto e Valor com Juros', # Ganha a Diferença entre Valor Bruto e Valor com Juros
#                         np.where((df_aux['y_true'] == 1) & (df_aux['y_predict'] == 0), 'Falso Negativo (Cliente Mau | Modelo classifica como Bom) - Perco o valor emprestado', # Perde o valor emprestado
#                         np.where((df_aux['y_true'] == 0) & (df_aux['y_predict'] == 1), 'Falso Positivo (Cliente Bom | Modelo classifica como Mau) - Deixo de ganhar a diferença entre Valor Bruto e Valor com Juros', # Deixo de ganhar a diferença entre Valor Bruto e Valor com Juros
#                         'Verdadeiro Positivo (Cliente Mau | Modelo classifica como Mau) - Não ganho nada' # Não ganho nada
#     )))

#     df_aux['retorno_financeiro'] = np.where((df_aux['y_true'] == 0) & (df_aux['y_predict'] == 0), df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'] - df_aux['valor_emprestimo_solicitado'], # Ganha a Diferença entre Valor Bruto e Valor com Juros
#                         np.where((df_aux['y_true'] == 1) & (df_aux['y_predict'] == 0), df_aux['valor_emprestimo_solicitado']*(-1), # Perde o valor emprestado
#                         np.where((df_aux['y_true'] == 0) & (df_aux['y_predict'] == 1), 0, # Deixo de ganhar a diferença entre Valor Bruto e Valor com Juros (df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'] - df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'])*(-1)
#                         0 # Não ganho nada
#     )))

#     valor_de_exposicao_total = int(df_aux['valor_emprestimo_solicitado'].sum())
#     retorno_financeiro = int(df_aux['retorno_financeiro'].sum())
#     valor_conquistado = valor_de_exposicao_total + retorno_financeiro
#     return_on_portfolio = round((retorno_financeiro/valor_de_exposicao_total)*100, 2)
#     retorno_financeiro_por_caso = df_aux.groupby('caso', as_index = False)['retorno_financeiro'].sum().sort_values(by = 'retorno_financeiro', ascending = False)

#     # Crie um DataFrame a partir dos hiperparâmetros
#     df = retorno_financeiro_por_caso.reset_index(drop=True)
#     df = df.round(2)

#     def color_etapa(val):
#         color = 'black'
#         if val == 'treino':
#             color = 'blue'
#         elif val == 'teste':
#             color = 'red'
#         return f'color: {color}; font-weight: bold;'

#     # Função para formatar os valores com até duas casas decimais
#     def format_values(val):
#         if isinstance(val, (int, float)):
#             return f'{val:.2f}'
#         return val

#     # Estilizando o DataFrame
#     styled_df = df.style\
#         .format(format_values)\
#         .applymap(lambda x: 'color: black; font-weight: bold; background-color: white; font-size: 14px')\
#         .applymap(color_etapa, subset=pd.IndexSlice[:, :])\
#         .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px')\
#         .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px')\
#         .set_table_styles([
#             {'selector': 'thead', 'props': [('color', 'black'), ('font-weight', 'bold'), ('background-color', 'lightgray')]}
#         ])

#     return retorno_financeiro, styled_df, valor_de_exposicao_total, return_on_portfolio


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
    
def transform_to_quantiles(df, variavel_continua):

    decile_limits = [df[variavel_continua].quantile(i / 10) for i in range(10)]  # Calcula os pontos de corte
    
    return decile_limits

# Exemplo de uso:
# Suponha que você tenha um DataFrame df e uma coluna chamada 'variavel_contínua'
# df['decile_values'] = transform_to_decile_values(df, 'variavel_contínua')


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
            KNeighborsClassifier(n_neighbors=3)  # Escolha o número adequado de vizinhos
        ),
        'SVM': make_pipeline(
            ColumnTransformer([
                ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols),
                ('scaler', make_pipeline(MinMaxScaler()), cols)
            ]),
            SVC(kernel='linear', class_weight={0: 1, 1: class_weight}, cache_size=1000, probability=True, random_state=42)
        ),
        'Random Forest': make_pipeline(
            ColumnTransformer([
                ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols)
            ]),
        RandomForestClassifier(
            random_state=42,            # Semente aleatória para reproducibilidade dos resultados
            criterion='entropy',       # Critério usado para medir a qualidade de uma divisão
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
        # 'MLP': make_pipeline(
        #     ColumnTransformer([
        #         ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols),
        #         ('scaler', make_pipeline(MinMaxScaler()), cols)
        #     ]),
        #     MLPClassifier(
        #         hidden_layer_sizes=(100,),  # Número de neurônios em cada camada oculta, ajuste conforme necessário
        #         activation='relu',  # Função de ativação para as camadas ocultas
        #         solver='adam',  # Algoritmo de otimização
        #         alpha=0.0001,  # Termo de regularização
        #         batch_size='auto',  # Tamanho do lote para otimização em lote, 'auto' ajusta automaticamente
        #         learning_rate='constant',  # Taxa de aprendizado
        #         learning_rate_init=0.001,  # Taxa de aprendizado inicial
        #         max_iter=200,  # Número máximo de iterações
        #         random_state=42
        #     )
        # )
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
        df['qt_anos_mesmo_emprego'] = (df['qt_anos_mesmo_emprego'].replace({'< 1 year':0, '1 year':1, '2 years':2, '3 years':3, '4 years':4, '5 years':5, '6 years':6, '7 years':7, '8 years':8, '9 years':9,'10+ years':10}).fillna(0))
        df['qt_anos_mesmo_emprego'] = df['qt_anos_mesmo_emprego'].apply(lambda x:int(x))
        df['qt_anos_mesmo_emprego'] = np.where(df['qt_anos_mesmo_emprego'] <= 3, '3_YEARS', 
                            np.where(df['qt_anos_mesmo_emprego'] <= 6, '6_YEARS',
                            np.where(df['qt_anos_mesmo_emprego'] <= 9, '9_YEARS',
                            '10_YEARS+')))
        return df['qt_anos_mesmo_emprego']

    def numero_de_registros_negativos(df):

        df = df[['situacao_do_emprestimo', 'registros_publicos_depreciativos']].copy()
        df[['registros_publicos_depreciativos']] = np.where(df[['registros_publicos_depreciativos']] == 0, 'sem_registros_negativos', 'com_registros_negativos')

        return df['registros_publicos_depreciativos']

    def consulta_de_credito_nos_ultimos_6_meses(df):
        df = df[['situacao_do_emprestimo', 'consultas_credito_6meses']].copy()
        df[['consultas_credito_6meses']] = np.where(df[['consultas_credito_6meses']] == 0, 'sem_consultas', 'com_consultas')

        return df['consultas_credito_6meses']

    def compromento_de_renda(df): 
        df_aux = df[['faturamento_anual', 'pagamento_mensal', 'valor_emprestimo_solicitado', 'qt_parcelas', 'taxa_de_juros', 'situacao_do_emprestimo']].copy()
        df_aux['qt_parcelas'] = np.where(df_aux['qt_parcelas'] == ' 36 months', 36, 60)
        df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'] = df_aux['pagamento_mensal']*df_aux['qt_parcelas']
        df_aux['pagamento_anual'] = np.where(df_aux['qt_parcelas'] == ' 36 months', df_aux['valor_emprestimo_solicitado_com_taxa_de_juros']/3, df_aux['valor_emprestimo_solicitado_com_taxa_de_juros']/5)
        df_aux['comprometimento_de_renda_anual'] = ((df_aux['pagamento_anual']/df_aux['faturamento_anual'])*100).round(2)
        
        return df_aux['comprometimento_de_renda_anual']

    def numero_incidencias_inadimplencia_vencidas_30d(df):
        df_aux = df[['situacao_do_emprestimo', 'inadimplencia_vencida_30dias']].copy()
        df_aux['inadimplencia_vencida_30dias'] = np.where(df_aux[['inadimplencia_vencida_30dias']] == 0, 'sem_inadimplencia_vencida', 'com_inadimplencia_vencida')

        return df_aux['inadimplencia_vencida_30dias']

    def n_meses_produto_credito_atual(df):
        df = df.copy()
        df['data_financiamento_emprestimo'] = pd.to_datetime(df['data_financiamento_emprestimo'], format = '%b-%y')
        df['mths_since_data_financiamento_emprestimo'] = round(pd.to_numeric((pd.to_datetime('2023-09-20') - df['data_financiamento_emprestimo'])/np.timedelta64(1, 'M')))
        df['mths_since_data_financiamento_emprestimo'] = df['mths_since_data_financiamento_emprestimo'].fillna(df['mths_since_data_financiamento_emprestimo'].median())
        df['mths_since_data_financiamento_emprestimo'] = np.where(df['mths_since_data_financiamento_emprestimo'] < 0, df['mths_since_data_financiamento_emprestimo'].median(), df['mths_since_data_financiamento_emprestimo'])
        df['mths_since_data_financiamento_emprestimo'] = df['mths_since_data_financiamento_emprestimo'].apply(lambda x:int(x))
        df['data_financiamento_emprestimo'] = df['mths_since_data_financiamento_emprestimo']

        return df['data_financiamento_emprestimo']

    def n_meses_primeiro_produto_credito(df):
        df = df.copy()
        df['data_contratacao_primeiro_produto_credito'] = pd.to_datetime(df['data_contratacao_primeiro_produto_credito'], format = '%b-%y')
        df['mths_since_data_contratacao_primeiro_produto_credito'] = round(pd.to_numeric((pd.to_datetime('2023-09-20') - df['data_contratacao_primeiro_produto_credito'])/np.timedelta64(1, 'M')))
        df['mths_since_data_contratacao_primeiro_produto_credito'] = df['mths_since_data_contratacao_primeiro_produto_credito'].fillna(df['mths_since_data_contratacao_primeiro_produto_credito'].median())
        df['mths_since_data_contratacao_primeiro_produto_credito'] = np.where(df['mths_since_data_contratacao_primeiro_produto_credito'] < 0, df['mths_since_data_contratacao_primeiro_produto_credito'].median(), df['mths_since_data_contratacao_primeiro_produto_credito'])
        df['mths_since_data_contratacao_primeiro_produto_credito'] = df['mths_since_data_contratacao_primeiro_produto_credito'].apply(lambda x:int(x))
        df['data_contratacao_primeiro_produto_credito'] = df['mths_since_data_contratacao_primeiro_produto_credito']
        
        return df['data_contratacao_primeiro_produto_credito']

    def produto_disponivel_publicamente(df):
        df_aux = df[['situacao_do_emprestimo', 'produto_disponivel_publicamente']].copy()
        df_aux['produto_disponivel_publicamente'] = np.where(df_aux[['produto_disponivel_publicamente']] == 0, 'sem_disponibilidade_publica', 'com_disponibilidade_publica')

        return df_aux['produto_disponivel_publicamente']

    def formato_features_binarias(df):
        df['qt_parcelas'] = np.where(df['qt_parcelas'] == ' 36 months', 0, 1)
        df['registros_publicos_depreciativos'] = np.where(df['qt_parcelas'] == 'sem_registros_negativos', 0, 1)
        df['inadimplencia_vencida_30dias'] = np.where(df['inadimplencia_vencida_30dias'] == 'sem_inadimplencia_vencida', 0, 1)
        df['tipo_de_concessao_do_credor'] = np.where(df['tipo_de_concessao_do_credor'] == 'f', 0, 1)
        df['plano_de_pagamento'] = np.where(df['plano_de_pagamento'] == 'n', 0, 1)
        df['renda_comprovada'] = np.where(df['renda_comprovada'] == 'Source Verified', 0, 1)
        df['consultas_credito_6meses'] = np.where(df['consultas_credito_6meses'] == 'sem_consultas', 0, 1)
        df['produto_disponivel_publicamente'] = np.where(df['consultas_credito_6meses'] == 'sem_disponibilidade_publica', 0, 1)

        return df

    def target_encoder_bad_rate(df, tipo):
        categoricas = ['grau_de_emprestimo', 'subclasse_de_emprestimo', 'produto_de_credito', 'qt_anos_mesmo_emprego', 'status_propriedade_residencial', 'estado']
        df_aux_2 = df.copy()
        if tipo == 'Criação':
            for cat in categoricas:
                df_aux = df[[f'{cat}', 'situacao_do_emprestimo']].copy()
                good = pd.DataFrame(df_aux.loc[df_aux['situacao_do_emprestimo'] == 0].groupby(f'{cat}', as_index = False)['situacao_do_emprestimo'].count()).rename({'situacao_do_emprestimo':'qt_good'}, axis = 1)
                bad = pd.DataFrame(df_aux.loc[df_aux['situacao_do_emprestimo'] == 1].groupby(f'{cat}', as_index = False)['situacao_do_emprestimo'].count()).rename({'situacao_do_emprestimo':'qt_bad'}, axis = 1)
                df_aux = good.merge(bad, on = f'{cat}', how = 'left')
                df_aux['qt_total'] = df_aux['qt_good'] + df_aux['qt_bad']
                df_aux[f'{cat}_enc'] = ((df_aux['qt_bad']/df_aux['qt_total'])*100).round(2)
                df_aux[f'{cat}_enc'] = df_aux[f'{cat}_enc'].apply(lambda x:float(x))
                df_aux = df_aux[[f'{cat}', f'{cat}_enc']].drop_duplicates().sort_values(by = f'{cat}_enc', ascending = True)
                df_aux.to_csv(f'features/{cat}_enc.csv', index = False)
                df_aux_2 = df_aux_2.merge(df_aux[[f'{cat}', f'{cat}_enc']], on = f'{cat}', how = 'left')
                df_aux_2.drop(f'{cat}', axis = 1, inplace = True)
        else:
            for cat in categoricas:
                ft = pd.read_csv(f'features/{cat}_enc.csv')
                replace_dict = dict(zip(ft[f'{cat}'], ft[f'{cat}_enc']))
                df_aux_2[f'{cat}_enc'] = df_aux_2[f'{cat}'].replace(replace_dict)

        return df_aux_2


    def simple_imputer(df):

        df_aux = df.copy()
        imputer = SimpleImputer(strategy = 'median')
        imputer.fit(df_aux)

        return imputer

    columns_selected = ['situacao_do_emprestimo', 'qt_parcelas','grau_de_emprestimo','subclasse_de_emprestimo','produto_de_credito', 'inadimplencia_vencida_30dias', 'valor_emprestimo_solicitado','taxa_de_juros','data_financiamento_emprestimo','produto_disponivel_publicamente','plano_de_pagamento','tipo_de_concessao_do_credor','pagamento_mensal','qt_anos_mesmo_emprego','status_propriedade_residencial',
    'renda_comprovada','faturamento_anual','estado', 'limite_total_produtos_credito','limite_total_rotativos','limite_rotativos_utilizado','taxa_utilizacao_limite_rotativos','qt_produtos_credito_contratados_atualmente','qt_produtos_credito_contratados_historicamente','registros_publicos_depreciativos','consultas_credito_6meses','data_contratacao_primeiro_produto_credito','qt_meses_desde_ultimo_registro_publico', 'qt_meses_classificacao_mais_recente_90dias',
    'qt_meses_ultima_inadimplencia']

    df_raw = df[columns_selected].copy()

    # Feature Selection

    features_selected = pd.read_csv('features/features_selected.csv')
    features_selected = features_selected.loc[features_selected['importance'] > 1] 
    features_selected = list(features_selected['feature'].unique()) + ['situacao_do_emprestimo']

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
        df_train['qt_anos_mesmo_emprego'] = numero_de_anos_emprego_atual(df_train)
        df_train['registros_publicos_depreciativos'] = numero_de_registros_negativos(df_train)
        df_train['consultas_credito_6meses'] = consulta_de_credito_nos_ultimos_6_meses(df_train)
        df_train['comprometimento_de_renda_anual'] = compromento_de_renda(df_train)
        df_train['inadimplencia_vencida_30dias'] = numero_incidencias_inadimplencia_vencidas_30d(df_train)
        df_train['data_financiamento_emprestimo'] = n_meses_produto_credito_atual(df_train)
        df_train['data_contratacao_primeiro_produto_credito'] = n_meses_primeiro_produto_credito(df_train)
        df_train = formato_features_binarias(df_train)
        df_train = target_encoder_bad_rate(df_train, 'escoragem')

        df_test['qt_anos_mesmo_emprego'] = numero_de_anos_emprego_atual(df_test)
        df_test['registros_publicos_depreciativos'] = numero_de_registros_negativos(df_test)
        df_test['consultas_credito_6meses'] = consulta_de_credito_nos_ultimos_6_meses(df_test)
        df_test['comprometimento_de_renda_anual'] = compromento_de_renda(df_test)
        df_test['inadimplencia_vencida_30dias'] = numero_incidencias_inadimplencia_vencidas_30d(df_test)
        df_test['data_financiamento_emprestimo'] = n_meses_produto_credito_atual(df_test)
        df_test['data_contratacao_primeiro_produto_credito'] = n_meses_primeiro_produto_credito(df_test)
        df_test = formato_features_binarias(df_test)
        df_test = target_encoder_bad_rate(df_test, 'escoragem')

        # Filtragem das Features que passaram no Feature Selection
        df_train = df_train[features_selected]
        df_test = df_test[features_selected]

        # Separação Feature e Target
        x_train, y_train = separa_feature_target('situacao_do_emprestimo', df_train)
        x_test, y_test = separa_feature_target('situacao_do_emprestimo', df_test)
        
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
                KNeighborsClassifier(n_neighbors=3)  # Escolha o número adequado de vizinhos
            ),
            'SVM': make_pipeline(
                ColumnTransformer([
                    ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols),
                    ('scaler', make_pipeline(MinMaxScaler()), cols)
                ]),
                SVC(kernel='linear', 
                    class_weight={0: 1, 1: class_weight}, 
                    cache_size=1000,
                    probability=True,
                    random_state=42
                    )
            ),
            'Random Forest': make_pipeline(
                ColumnTransformer([
                    ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols)
                ]),
            RandomForestClassifier(
                random_state=42,            # Semente aleatória para reproducibilidade dos resultados
                criterion='entropy',       # Critério usado para medir a qualidade de uma divisão
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
                tree_method = 'gpu_hist',
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
            # 'MLP': make_pipeline(
            #     ColumnTransformer([
            #         ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols),
            #         ('scaler', make_pipeline(MinMaxScaler()), cols)
            #     ]),
            #     MLPClassifier(
            #         hidden_layer_sizes=(100,),  # Número de neurônios em cada camada oculta, ajuste conforme necessário
            #         activation='relu',  # Função de ativação para as camadas ocultas
            #         solver='adam',  # Algoritmo de otimização
            #         alpha=0.0001,  # Termo de regularização
            #         batch_size='auto',  # Tamanho do lote para otimização em lote, 'auto' ajusta automaticamente
            #         learning_rate='constant',  # Taxa de aprendizado
            #         learning_rate_init=0.001,  # Taxa de aprendizado inicial
            #         max_iter=200,  # Número máximo de iterações
            #         random_state=42
            #     )
            # )
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
            'situacao_do_emprestimo': y_test['situacao_do_emprestimo'].values,
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

# def modelo_otimizado_hyperopt_(classificador, target, x_train, y_train, x_test, y_test):

#     cols = list(x_train.columns)

#     # Define o ColumnTransformer
#     preprocessor = ColumnTransformer([
#         ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols),
#         ('scaler', make_pipeline(MinMaxScaler()), cols)
#     ])

#     # Divide o conjunto de treinamento em treinamento e validação
#     x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

#     # Função de avaliação para o Hyperopt
#     def objective(params):
#         model = make_pipeline(
#             preprocessor,
#             XGBClassifier(
#                 random_state=42,
#                 eval_metric='logloss',
#                 objective='binary:logistic',
#                 **params
#             )
#         )

#         # Treina o modelo com Early Stopping usando o conjunto de validação
#         model.fit(
#             x_train, 
#             y_train.values.ravel(),
#             eval_set=[(x_val, y_val.values.ravel())],  # Conjunto de dados utilizado para otimização
#             early_stopping_rounds=10,  # Número de iterações sem melhoria no conjunto de validação para parar o treinamento
#             verbose=False
#         )

#         # Obtém a melhor iteração do modelo
#         best_iteration = model.named_steps['xgbclassifier'].best_iteration

#         # Use a métrica de validação cruzada para otimização
#         score = model.named_steps['xgbclassifier'].best_score
#         return -score  # Hyperopt minimiza a função objetivo, então multiplicamos por -1

#     # Espaço de busca para os hiperparâmetros
#     space = {
#         'n_estimators': hp.choice('n_estimators', [10, 20, 50, 100]),
#         'max_depth': hp.choice('max_depth', [4, 5, 7, 8, 9, 10]),
#         'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
#         'reg_alpha': hp.uniform('reg_alpha', 0.5, 1),
#         'reg_lambda': hp.uniform('reg_lambda', 0.5, 1),
#         'gamma': hp.uniform('gamma', 0.5, 1),
#         'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
#         'subsample': hp.uniform('subsample', 0.5, 1),
#         'scale_pos_weight': hp.choice('scale_pos_weight', [3, 5, 8, 10, 12, 14]),
#         'base_score': hp.uniform('base_score', 0.30, 0.90)
#     }

#     # Executa a otimização
#     trials = Trials()
#     best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials, verbose=0)


#     # Cria o modelo final com os melhores hiperparâmetros encontrados
#     best_model = XGBClassifier(
#         n_estimators=int(best_params['n_estimators']),
#         max_depth=int(best_params['max_depth']),
#         learning_rate=best_params['learning_rate'],
#         subsample=best_params['subsample'],
#         reg_alpha=best_params['reg_alpha'],
#         reg_lambda=best_params['reg_lambda'],
#         gamma=best_params['gamma'],
#         colsample_bytree=best_params['colsample_bytree'],
#         scale_pos_weight=int(best_params['scale_pos_weight']),
#         base_score=best_params['base_score'],
#         random_state=42
#     )

#     # Treina o modelo final no conjunto completo de treinamento
#     best_model.fit(x_train, y_train.values.ravel())

#     y_pred_train = best_model.predict(x_train)
#     y_pred_test = best_model.predict(x_test)

#     y_proba_train = best_model.predict_proba(x_train)
#     y_proba_test = best_model.predict_proba(x_test)

#     return best_model, y_pred_train, y_pred_test, y_proba_train, y_proba_test, best_params


def otimizacao(classificador, x_train, y_train, x_test, y_test):
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
            XGBClassifier(random_state=42, tree_method = 'gpu_hist', eval_metric='logloss', objective='binary:logistic'),
            {
                'n_estimators': (99, 100), # Número de Árvores construídas
                'max_depth': (7, 8, 9), # Profundidade Máxima de cada Árvore
                'learning_rate': (0.03, 0.05), # Tamanho do passo utilizado no Método do Gradiente Descendente
                'reg_alpha':(0.5, 1), # Valor do Alpha aplicado durante a Regularização Lasso L1 
                'reg_lambda':(0.5, 1), # Valor do Lambda aplicado durante a Regularização Ridge L2
                'gamma':(0.5, 1), # Valor mínimo permitido para um Nó de Árvore ser aceito. Ajuda a controlar o crescimento das Árvores, evitando divisões insignificantes
                'colsample_bytree':(0.5, 1), # Porcentagem de Colunas utilizada para a amostragem aleatória durante a criação das Árvores
                'subsample':(0.5, 1), # Porcentagem de Linhas utilizada para a amostragem aleatória durante a criação das Árvores
                'scale_pos_weight':(4, 5, 6, 7, 8), # Peso atribuído a classe positiva, aumentando a importância da classe minoritária
                #'base_score':(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9) # Threshold de Probabilidade de decisão do modelo
            },
            n_iter=10,
            random_state=42,
            n_jobs=-1,
            scoring='recall',
            cv=5
        )
    )

    # Treina o modelo
    model.fit(x_train, y_train)

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    y_proba_train = model.predict_proba(x_train)
    y_proba_test = model.predict_proba(x_test)

    melhores_hiperparametros = model.named_steps['bayessearchcv'].best_params_
    hiperparametros = pd.DataFrame([melhores_hiperparametros])

    best_hiperpams = []
    for chave, valor in melhores_hiperparametros.items():
        best_hiperpams.append([chave, valor])

    pivot = pd.DataFrame(best_hiperpams).T
    pivot.columns = pivot.iloc[0]
    pivot = pivot.drop(0)

    # Crie um DataFrame a partir dos hiperparâmetros
    df = hiperparametros.reset_index(drop=True)
    df = df.round(2)

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
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: white; font-size: 14px')\
        .applymap(color_etapa, subset=pd.IndexSlice[:, :])\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px')\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px')\
        .set_table_styles([
            {'selector': 'thead', 'props': [('color', 'black'), ('font-weight', 'bold'), ('background-color', 'lightgray')]}
        ])

    return model, y_pred_train, y_pred_test, y_proba_train, y_proba_test, styled_df, pivot

def validacao_cruzada_classificacao_otimizada(classificador, df, target_column, n_splits, best_hiperpams):

    def numero_de_anos_emprego_atual(df):
        df['qt_anos_mesmo_emprego'] = (df['qt_anos_mesmo_emprego'].replace({'< 1 year':0, '1 year':1, '2 years':2, '3 years':3, '4 years':4, '5 years':5, '6 years':6, '7 years':7, '8 years':8, '9 years':9,'10+ years':10}).fillna(0))
        df['qt_anos_mesmo_emprego'] = df['qt_anos_mesmo_emprego'].apply(lambda x:int(x))
        df['qt_anos_mesmo_emprego'] = np.where(df['qt_anos_mesmo_emprego'] <= 3, '3_YEARS', 
                            np.where(df['qt_anos_mesmo_emprego'] <= 6, '6_YEARS',
                            np.where(df['qt_anos_mesmo_emprego'] <= 9, '9_YEARS',
                            '10_YEARS+')))
        return df['qt_anos_mesmo_emprego']

    def numero_de_registros_negativos(df):

        df = df[['situacao_do_emprestimo', 'registros_publicos_depreciativos']].copy()
        df[['registros_publicos_depreciativos']] = np.where(df[['registros_publicos_depreciativos']] == 0, 'sem_registros_negativos', 'com_registros_negativos')

        return df['registros_publicos_depreciativos']

    def consulta_de_credito_nos_ultimos_6_meses(df):
        df = df[['situacao_do_emprestimo', 'consultas_credito_6meses']].copy()
        df[['consultas_credito_6meses']] = np.where(df[['consultas_credito_6meses']] == 0, 'sem_consultas', 'com_consultas')

        return df['consultas_credito_6meses']

    def compromento_de_renda(df): 
        df_aux = df[['faturamento_anual', 'pagamento_mensal', 'valor_emprestimo_solicitado', 'qt_parcelas', 'taxa_de_juros', 'situacao_do_emprestimo']].copy()
        df_aux['qt_parcelas'] = np.where(df_aux['qt_parcelas'] == ' 36 months', 36, 60)
        df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'] = df_aux['pagamento_mensal']*df_aux['qt_parcelas']
        df_aux['pagamento_anual'] = np.where(df_aux['qt_parcelas'] == ' 36 months', df_aux['valor_emprestimo_solicitado_com_taxa_de_juros']/3, df_aux['valor_emprestimo_solicitado_com_taxa_de_juros']/5)
        df_aux['comprometimento_de_renda_anual'] = ((df_aux['pagamento_anual']/df_aux['faturamento_anual'])*100).round(2)
        
        return df_aux['comprometimento_de_renda_anual']

    def numero_incidencias_inadimplencia_vencidas_30d(df):
        df_aux = df[['situacao_do_emprestimo', 'inadimplencia_vencida_30dias']].copy()
        df_aux['inadimplencia_vencida_30dias'] = np.where(df_aux[['inadimplencia_vencida_30dias']] == 0, 'sem_inadimplencia_vencida', 'com_inadimplencia_vencida')

        return df_aux['inadimplencia_vencida_30dias']

    def n_meses_produto_credito_atual(df):
        df = df.copy()
        df['data_financiamento_emprestimo'] = pd.to_datetime(df['data_financiamento_emprestimo'], format = '%b-%y')
        df['mths_since_data_financiamento_emprestimo'] = round(pd.to_numeric((pd.to_datetime('2023-09-20') - df['data_financiamento_emprestimo'])/np.timedelta64(1, 'M')))
        df['mths_since_data_financiamento_emprestimo'] = df['mths_since_data_financiamento_emprestimo'].fillna(df['mths_since_data_financiamento_emprestimo'].median())
        df['mths_since_data_financiamento_emprestimo'] = np.where(df['mths_since_data_financiamento_emprestimo'] < 0, df['mths_since_data_financiamento_emprestimo'].median(), df['mths_since_data_financiamento_emprestimo'])
        df['mths_since_data_financiamento_emprestimo'] = df['mths_since_data_financiamento_emprestimo'].apply(lambda x:int(x))
        df['data_financiamento_emprestimo'] = df['mths_since_data_financiamento_emprestimo']

        return df['data_financiamento_emprestimo']

    def n_meses_primeiro_produto_credito(df):
        df = df.copy()
        df['data_contratacao_primeiro_produto_credito'] = pd.to_datetime(df['data_contratacao_primeiro_produto_credito'], format = '%b-%y')
        df['mths_since_data_contratacao_primeiro_produto_credito'] = round(pd.to_numeric((pd.to_datetime('2023-09-20') - df['data_contratacao_primeiro_produto_credito'])/np.timedelta64(1, 'M')))
        df['mths_since_data_contratacao_primeiro_produto_credito'] = df['mths_since_data_contratacao_primeiro_produto_credito'].fillna(df['mths_since_data_contratacao_primeiro_produto_credito'].median())
        df['mths_since_data_contratacao_primeiro_produto_credito'] = np.where(df['mths_since_data_contratacao_primeiro_produto_credito'] < 0, df['mths_since_data_contratacao_primeiro_produto_credito'].median(), df['mths_since_data_contratacao_primeiro_produto_credito'])
        df['mths_since_data_contratacao_primeiro_produto_credito'] = df['mths_since_data_contratacao_primeiro_produto_credito'].apply(lambda x:int(x))
        df['data_contratacao_primeiro_produto_credito'] = df['mths_since_data_contratacao_primeiro_produto_credito']
        
        return df['data_contratacao_primeiro_produto_credito']

    def produto_disponivel_publicamente(df):
        df_aux = df[['situacao_do_emprestimo', 'produto_disponivel_publicamente']].copy()
        df_aux['produto_disponivel_publicamente'] = np.where(df_aux[['produto_disponivel_publicamente']] == 0, 'sem_disponibilidade_publica', 'com_disponibilidade_publica')

        return df_aux['produto_disponivel_publicamente']

    def formato_features_binarias(df):
        df['qt_parcelas'] = np.where(df['qt_parcelas'] == ' 36 months', 0, 1)
        df['registros_publicos_depreciativos'] = np.where(df['qt_parcelas'] == 'sem_registros_negativos', 0, 1)
        df['inadimplencia_vencida_30dias'] = np.where(df['inadimplencia_vencida_30dias'] == 'sem_inadimplencia_vencida', 0, 1)
        df['tipo_de_concessao_do_credor'] = np.where(df['tipo_de_concessao_do_credor'] == 'f', 0, 1)
        df['plano_de_pagamento'] = np.where(df['plano_de_pagamento'] == 'n', 0, 1)
        df['renda_comprovada'] = np.where(df['renda_comprovada'] == 'Source Verified', 0, 1)
        df['consultas_credito_6meses'] = np.where(df['consultas_credito_6meses'] == 'sem_consultas', 0, 1)
        df['produto_disponivel_publicamente'] = np.where(df['consultas_credito_6meses'] == 'sem_disponibilidade_publica', 0, 1)

        return df

    def target_encoder_bad_rate(df, tipo):
        categoricas = ['grau_de_emprestimo', 'subclasse_de_emprestimo', 'produto_de_credito', 'qt_anos_mesmo_emprego', 'status_propriedade_residencial', 'estado']
        df_aux_2 = df.copy()
        if tipo == 'Criação':
            for cat in categoricas:
                df_aux = df[[f'{cat}', 'situacao_do_emprestimo']].copy()
                good = pd.DataFrame(df_aux.loc[df_aux['situacao_do_emprestimo'] == 0].groupby(f'{cat}', as_index = False)['situacao_do_emprestimo'].count()).rename({'situacao_do_emprestimo':'qt_good'}, axis = 1)
                bad = pd.DataFrame(df_aux.loc[df_aux['situacao_do_emprestimo'] == 1].groupby(f'{cat}', as_index = False)['situacao_do_emprestimo'].count()).rename({'situacao_do_emprestimo':'qt_bad'}, axis = 1)
                df_aux = good.merge(bad, on = f'{cat}', how = 'left')
                df_aux['qt_total'] = df_aux['qt_good'] + df_aux['qt_bad']
                df_aux[f'{cat}_enc'] = ((df_aux['qt_bad']/df_aux['qt_total'])*100).round(2)
                df_aux[f'{cat}_enc'] = df_aux[f'{cat}_enc'].apply(lambda x:float(x))
                df_aux = df_aux[[f'{cat}', f'{cat}_enc']].drop_duplicates().sort_values(by = f'{cat}_enc', ascending = True)
                df_aux.to_csv(f'features/{cat}_enc.csv', index = False)
                df_aux_2 = df_aux_2.merge(df_aux[[f'{cat}', f'{cat}_enc']], on = f'{cat}', how = 'left')
                df_aux_2.drop(f'{cat}', axis = 1, inplace = True)
        else:
            for cat in categoricas:
                ft = pd.read_csv(f'features/{cat}_enc.csv')
                replace_dict = dict(zip(ft[f'{cat}'], ft[f'{cat}_enc']))
                df_aux_2[f'{cat}_enc'] = df_aux_2[f'{cat}'].replace(replace_dict)

        return df_aux_2



    def simple_imputer(df):

        df_aux = df.copy()
        imputer = SimpleImputer(strategy = 'median')
        imputer.fit(df_aux)

        return imputer

    columns_selected = ['situacao_do_emprestimo', 'qt_parcelas','grau_de_emprestimo','subclasse_de_emprestimo','produto_de_credito', 'inadimplencia_vencida_30dias', 'valor_emprestimo_solicitado', 'taxa_de_juros','data_financiamento_emprestimo','produto_disponivel_publicamente','plano_de_pagamento','tipo_de_concessao_do_credor','pagamento_mensal','qt_anos_mesmo_emprego','status_propriedade_residencial',
    'renda_comprovada','faturamento_anual','estado', 'limite_total_produtos_credito','limite_total_rotativos','limite_rotativos_utilizado','taxa_utilizacao_limite_rotativos','qt_produtos_credito_contratados_atualmente','qt_produtos_credito_contratados_historicamente','registros_publicos_depreciativos','consultas_credito_6meses','data_contratacao_primeiro_produto_credito','qt_meses_desde_ultimo_registro_publico', 'qt_meses_classificacao_mais_recente_90dias',
    'qt_meses_ultima_inadimplencia']


    df_raw = df[columns_selected].copy()

    # Feature Selection

    features_selected = pd.read_csv('features/features_selected.csv')
    features_selected = features_selected.loc[features_selected['importance'] > 1] 
    features_selected = list(features_selected['feature'].unique()) + ['situacao_do_emprestimo']

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
        df_train['qt_anos_mesmo_emprego'] = numero_de_anos_emprego_atual(df_train)
        df_train['registros_publicos_depreciativos'] = numero_de_registros_negativos(df_train)
        df_train['consultas_credito_6meses'] = consulta_de_credito_nos_ultimos_6_meses(df_train)
        df_train['comprometimento_de_renda_anual'] = compromento_de_renda(df_train)
        df_train['inadimplencia_vencida_30dias'] = numero_incidencias_inadimplencia_vencidas_30d(df_train)
        df_train['data_financiamento_emprestimo'] = n_meses_produto_credito_atual(df_train)
        df_train['data_contratacao_primeiro_produto_credito'] = n_meses_primeiro_produto_credito(df_train)
        df_train = formato_features_binarias(df_train)
        df_train = target_encoder_bad_rate(df_train, 'escoragem')

        df_test['qt_anos_mesmo_emprego'] = numero_de_anos_emprego_atual(df_test)
        df_test['registros_publicos_depreciativos'] = numero_de_registros_negativos(df_test)
        df_test['consultas_credito_6meses'] = consulta_de_credito_nos_ultimos_6_meses(df_test)
        df_test['comprometimento_de_renda_anual'] = compromento_de_renda(df_test)
        df_test['inadimplencia_vencida_30dias'] = numero_incidencias_inadimplencia_vencidas_30d(df_test)
        df_test['data_financiamento_emprestimo'] = n_meses_produto_credito_atual(df_test)
        df_test['data_contratacao_primeiro_produto_credito'] = n_meses_primeiro_produto_credito(df_test)
        df_test = formato_features_binarias(df_test)
        df_test = target_encoder_bad_rate(df_test, 'escoragem')

        # Filtragem das Features que passaram no Feature Selection
        df_train = df_train[features_selected]
        df_test = df_test[features_selected]

        # Separação Feature e Target
        x_train, y_train = separa_feature_target('situacao_do_emprestimo', df_train)
        x_test, y_test = separa_feature_target('situacao_do_emprestimo', df_test)
        
        # Imputer
        cols = list(x_train.columns)
        imputer = simple_imputer(x_train)
        x_train = pd.DataFrame(imputer.transform(x_train), columns = x_train.columns)
        x_test = pd.DataFrame(imputer.transform(x_test), columns = x_test.columns)


        # Melhores Hiperparâmetros
        melhores_hiperparametros = best_hiperpams
        colsample_bytree = melhores_hiperparametros['colsample_bytree'][1]
        gamma = melhores_hiperparametros['gamma'][1]
        learning_rate = melhores_hiperparametros['learning_rate'][1]
        max_depth = melhores_hiperparametros['max_depth'][1]
        n_estimators = melhores_hiperparametros['n_estimators'][1]
        reg_alpha = melhores_hiperparametros['reg_alpha'][1]
        reg_lambda = melhores_hiperparametros['reg_lambda'][1]
        scale_pos_weight = melhores_hiperparametros['scale_pos_weight'][1]
        subsample = melhores_hiperparametros['subsample'][1]

        # Define as colunas categóricas e numéricas
        model = make_pipeline(
                ColumnTransformer([
                    ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols)
                ]),
                
                # XGBClassifier(
                #     random_state=42, # Semente aleatória para reproducibilidade dos resultados
                #     tree_method = 'gpu_hist',
                #     eval_metric='logloss', # Métrica de avaliação durante o treinamento, 'logloss' é comum para problemas de classificação binária
                #     objective='binary:logistic', # Define o objetivo do modelo, 'binary:logistic' para classificação binária
                #     n_estimators = n_estimators, # Número de árvores no modelo (equivalente ao n_estimators na Random Forest)
                #     max_depth = max_depth, # Profundidade máxima de cada árvore
                #     learning_rate = learning_rate, # Taxa de aprendizado - controla a contribuição de cada árvore
                #     reg_alpha = reg_alpha, # Termo de regularização L1 (penalidade nos pesos)
                #     reg_lambda = reg_lambda, # Termo de regularização L2 (penalidade nos quadrados dos pesos)
                #     gamma = gamma, # Controle de poda da árvore, maior gamma leva a menos crescimento da árvore
                #     colsample_bytree = colsample_bytree, # Fração de características a serem consideradas ao construir cada árvore
                #     subsample = subsample, # Fração de amostras a serem usadas para treinar cada árvore
                #     scale_pos_weight = scale_pos_weight, # Peso das classes positivas em casos desequilibrados
                #     base_score = 0.5 # Threshold de Probabilidade de Decisão do Classificador (geralmente é 0.5 para problemas de classificação binária)
                # )
            XGBClassifier(
                    random_state=42, # Semente aleatória para reproducibilidade dos resultados
                    tree_method = 'gpu_hist', # Treino usando GPU
                    eval_metric='logloss', # Métrica de avaliação durante o treinamento, 'logloss' é comum para problemas de classificação binária
                    objective='binary:logistic', # Define o objetivo do modelo, 'binary:logistic' para classificação binária
                    n_estimators = 99, # Número de árvores no modelo (equivalente ao n_estimators na Random Forest)
                    max_depth = 8, # Profundidade máxima de cada árvore
                    learning_rate = 0.03209718317105407, # Taxa de aprendizado - controla a contribuição de cada árvore
                    reg_alpha = 0.7268326719031495, # Termo de regularização L1 (penalidade nos pesos)
                    reg_lambda = 0.5777240270252717, # Termo de regularização L2 (penalidade nos quadrados dos pesos)
                    gamma = 0.9593612608346885, # Controle de poda da árvore, maior gamma leva a menos crescimento da árvore
                    colsample_bytree = 0.7224162561505759, # Fração de características a serem consideradas ao construir cada árvore
                    subsample = 0.7786702115169006, # Fração de amostras a serem usadas para treinar cada árvore
                    scale_pos_weight = 7, # Peso das classes positivas em casos desequilibrados
                    base_score = 0.5 # Threshold de Probabilidade de Decisão do Classificador (geralmente é 0.5 para problemas de classificação binária)
                )
            )

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
            'situacao_do_emprestimo': y_test['situacao_do_emprestimo'].values,
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



def modelo_corte_probabilidade(df_model, df_retorno_financeiro, target, x, y):

    def simple_imputer(df_model):

        df_aux = df_model.copy()
        imputer = SimpleImputer(strategy = 'median')
        imputer.fit(df_aux)

        return imputer
    
    cols = list(x.columns)
    imputer = simple_imputer(x)
    x = pd.DataFrame(imputer.transform(x), columns = x.columns)
    
    list_threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    list_lucro = []
    for threshold in list_threshold:
        # Define o ColumnTransformer
        preprocessor = ColumnTransformer([
                    ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols),
                    ('scaler', make_pipeline(MinMaxScaler()), cols)
                ])

        model = make_pipeline(
        preprocessor,
            XGBClassifier(
            random_state=42, # Semente aleatória para reproducibilidade dos resultados
            tree_method = 'gpu_hist', # Treino usando GPU
            eval_metric='logloss', # Métrica de avaliação durante o treinamento, 'logloss' é comum para problemas de classificação binária
            objective='binary:logistic', # Define o objetivo do modelo, 'binary:logistic' para classificação binária
            n_estimators = 99, # Número de árvores no modelo (equivalente ao n_estimators na Random Forest)
            max_depth = 8, # Profundidade máxima de cada árvore
            learning_rate = 0.03209718317105407, # Taxa de aprendizado - controla a contribuição de cada árvore
            reg_alpha = 0.7268326719031495, # Termo de regularização L1 (penalidade nos pesos)
            reg_lambda = 0.5777240270252717, # Termo de regularização L2 (penalidade nos quadrados dos pesos)
            gamma = 0.9593612608346885, # Controle de poda da árvore, maior gamma leva a menos crescimento da árvore
            colsample_bytree = 0.7224162561505759, # Fração de características a serem consideradas ao construir cada árvore
            subsample = 0.7786702115169006, # Fração de amostras a serem usadas para treinar cada árvore
            scale_pos_weight = 7, # Peso das classes positivas em casos desequilibrados
            base_score = 0.5 # Threshold de Probabilidade de Decisão do Classificador (geralmente é 0.5 para problemas de classificação binária)
            )
        )
        
        model.fit(x, y)

        y_pred = model.predict(x)
        y_predict_proba = model.predict_proba(x)

        teste_threshold = pd.DataFrame({'y_predict':y_pred, 'Proba Good':y_predict_proba[:, 0]})
        teste_threshold['y_predict_threshold'] = np.where(teste_threshold['Proba Good'] <= threshold, 1, 0)
        y_pred = teste_threshold['y_predict_threshold'].values

        lucro = retorno_financeiro(df_retorno_financeiro, y_pred)[0]
        list_lucro.append(lucro)
    
    corte_probabilidade = pd.DataFrame({'threshold':list_threshold, 'lucro':list_lucro})


    # teste_threshold = pd.DataFrame({'y_true':y_test['situacao_do_emprestimo'].values, 'y_predict_test':y_predict_test_otimizado, 'Proba Good':y_proba_test_otimizado[:, 0], 'Proba Bad':y_proba_test_otimizado[:, 1]})
    # teste_threshold['y_predict_threshold'] = np.where(teste_threshold['Proba Good'] <= 0.4, 1, 0)


    # display(teste_threshold.groupby('y_true', as_index = False)['Proba Good'].count())
    # display(teste_threshold.groupby('y_predict_test', as_index = False)['Proba Good'].count())
    # display(teste_threshold.groupby('y_predict_threshold', as_index = False)['Proba Good'].count())
    return corte_probabilidade



def modelo_oficial(classificador, x, y):
    def simple_imputer(df_model):

        df_aux = df_model.copy()
        imputer = SimpleImputer(strategy = 'median')
        imputer.fit(df_aux)

        return imputer
    
    cols = list(x.columns)
    imputer = simple_imputer(x)
    x = pd.DataFrame(imputer.transform(x), columns = x.columns)

    # Define o ColumnTransformer
    preprocessor = ColumnTransformer([
                ('imputer', make_pipeline(SimpleImputer(strategy='median')), cols),
                ('scaler', make_pipeline(MinMaxScaler()), cols)
            ])

    model = make_pipeline(
    preprocessor,
        XGBClassifier(
        random_state=42, # Semente aleatória para reproducibilidade dos resultados
        tree_method = 'gpu_hist', # Treino usando GPU
        eval_metric='logloss', # Métrica de avaliação durante o treinamento, 'logloss' é comum para problemas de classificação binária
        objective='binary:logistic', # Define o objetivo do modelo, 'binary:logistic' para classificação binária
        n_estimators = 99, # Número de árvores no modelo (equivalente ao n_estimators na Random Forest)
        max_depth = 8, # Profundidade máxima de cada árvore
        learning_rate = 0.03209718317105407, # Taxa de aprendizado - controla a contribuição de cada árvore
        reg_alpha = 0.7268326719031495, # Termo de regularização L1 (penalidade nos pesos)
        reg_lambda = 0.5777240270252717, # Termo de regularização L2 (penalidade nos quadrados dos pesos)
        gamma = 0.9593612608346885, # Controle de poda da árvore, maior gamma leva a menos crescimento da árvore
        colsample_bytree = 0.7224162561505759, # Fração de características a serem consideradas ao construir cada árvore
        subsample = 0.7786702115169006, # Fração de amostras a serem usadas para treinar cada árvore
        scale_pos_weight = 7, # Peso das classes positivas em casos desequilibrados
        base_score = 0.5 # Threshold de Probabilidade de Decisão do Classificador (geralmente é 0.5 para problemas de classificação binária)
        )
    )

    # Treina o modelo oficial
    model.fit(x, y)
    salvar_modelo_pickle(model, 'models/clf_final.pkl')

def escoragem(x, y):
    clf_final = carregar_modelo_pickle('models/clf_final.pkl')

    y_pred = clf_final.predict(x)
    y_proba= clf_final.predict_proba(x)
    teste_threshold = pd.DataFrame({'y_predict':y_pred, 'Proba Good':y_proba[:, 0]})
    teste_threshold['y_predict_threshold'] = np.where(teste_threshold['Proba Good'] <= 0.3, 1, 0)
    y_pred = teste_threshold['y_predict_threshold'].values

    return y_pred, y_proba


def salvar_modelo_pickle(modelo, caminho_arquivo):
    """
    Salva um modelo em um arquivo pickle.

    Parâmetros:
    - modelo: O modelo treinado que você deseja salvar.
    - caminho_arquivo: O caminho do arquivo onde o modelo será salvo.
    """
    with open(caminho_arquivo, 'wb') as arquivo:
        pickle.dump(modelo, arquivo)
    print(f"Modelo salvo em {caminho_arquivo}")

def carregar_modelo_pickle(caminho_arquivo):
    """
    Carrega um modelo salvo de um arquivo pickle.

    Parâmetros:
    - caminho_arquivo: O caminho do arquivo onde o modelo foi salvo.

    Retorna:
    - O modelo carregado.
    """
    with open(caminho_arquivo, 'rb') as arquivo:
        modelo_carregado = pickle.load(arquivo)
    print(f"Modelo carregado de {caminho_arquivo}")
    return modelo_carregado


def calibracao_probabilidade():  
    # Modelo Otimizado
    model_otimizado, y_predict_train_otimizado, y_predict_test_otimizado, y_predict_proba_train_otimizado, y_predict_proba_test_otimizado = modelo_otimizado('Bayes Search + Threshold Proba + XGBoost', x_train, y_train, x_test, y_test)

    # Modelo de Calibração
    calibrated_clf = CalibratedClassifierCV(model_otimizado, cv=5, method='isotonic')
    calibrated_clf.fit(x_train, y_train)

    y_predict_proba_ajustada_train = calibrated_clf.predict_proba(x_train)
    y_predict_proba_ajustada_test = calibrated_clf.predict_proba(x_test)

    predict_proba_train = pd.DataFrame(y_predict_proba_train_otimizado.tolist(), columns=['predict_proba_0', 'predict_proba_1'])
    predict_proba_test = pd.DataFrame(y_predict_proba_test_otimizado.tolist(), columns=['predict_proba_0', 'predict_proba_1'])

    predict_proba_ajustada_train = pd.DataFrame(y_predict_proba_ajustada_train.tolist(), columns=['predict_proba_0', 'predict_proba_1'])
    predict_proba_ajustada_test = pd.DataFrame(y_predict_proba_ajustada_test.tolist(), columns=['predict_proba_0', 'predict_proba_1'])

    # Definição das Probabilidades

    probabilities_train = predict_proba_train['predict_proba_1'] # Obtenha as probabilidades previstas
    prob_true_train, prob_pred_train = calibration_curve(y_train, probabilities_train, n_bins=10) # Calcule a curva de calibração
    brier_score_train = brier_score_loss(y_train, probabilities_train) # Calcule o Brier Score (uma métrica de calibração)

    probabilities_test = predict_proba_test['predict_proba_1'] # Obtenha as probabilidades previstas
    prob_true_test, prob_pred_test = calibration_curve(y_test, probabilities_test, n_bins=10) # Calcule a curva de calibração
    brier_score_test = brier_score_loss(y_test, probabilities_test) # Calcule o Brier Score (uma métrica de calibração)

    probabilities_ajustada_train = predict_proba_ajustada_train['predict_proba_1'] # Obtenha as probabilidades previstas
    prob_true_ajustada_train, prob_pred_ajustada_train = calibration_curve(y_train, probabilities_ajustada_train, n_bins=10) # Calcule a curva de calibração
    brier_score_ajustada_train = brier_score_loss(y_train, probabilities_ajustada_train) # Calcule o Brier Score (uma métrica de calibração)

    probabilities_ajustada_test = predict_proba_ajustada_test['predict_proba_1'] # Obtenha as probabilidades previstas
    prob_true_ajustada_test, prob_pred_ajustada_test = calibration_curve(y_test, probabilities_ajustada_test, n_bins=10) # Calcule a curva de calibração
    brier_score_ajustada_test = brier_score_loss(y_test, probabilities_ajustada_test) # Calcule o Brier Score (uma métrica de calibração)

    y_predict_ajustada_train_best_clf = calibrated_clf.predict(x_train)
    y_predict_ajustada_test_best_clf = calibrated_clf.predict(x_test)

    y_predict_proba_ajustada_train_best_clf = calibrated_clf.predict_proba(x_train)
    y_predict_proba_ajustada_test_best_clf = calibrated_clf.predict_proba(x_test)


    # Métricas Otimizadas

    metricas_before_calibration_ajustada = metricas_classificacao('Threshold Proba (0.1) + Bayes Search + XGBoost', y_train, y_predict_train_otimizado, y_test, y_predict_test_otimizado, y_predict_proba_train_otimizado, y_predict_proba_test_otimizado)
    metricas_after_calibration_ajustada = metricas_classificacao('Calibration + Threshold Proba (0.1) + Bayes Search + XGBoost', y_train, y_predict_ajustada_train_best_clf, y_test, y_predict_ajustada_test_best_clf, y_predict_proba_ajustada_train_best_clf, y_predict_proba_ajustada_test_best_clf)


    print('Métricas Finais')
    metricas_finais = metricas_classificacao_modelos_juntos(
        [
            metricas_before_calibration_ajustada,
            metricas_after_calibration_ajustada
        ]
    )
    display(metricas_finais)

    # Plote a curva de calibração
    plt.figure(figsize=(8, 8))
    plt.plot(prob_pred_train, prob_true_train, marker='o', label='Probability Curve Before Calibration- Train', color = 'blue')
    plt.plot(prob_pred_test, prob_true_test, marker='o', label='Probability Curve Before Calibration - Test', color = 'red')
    plt.plot(prob_pred_ajustada_train, prob_true_ajustada_train, marker='o', label='Probability Curve After Calibration - Train', color = 'green')
    plt.plot(prob_pred_ajustada_test, prob_true_ajustada_test, marker='o', label='Probability Curve After Calibration - Test', color = 'purple')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.title(f'Calibration Curve')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.legend()
    plt.show()


def politica_de_credito(df):    
    df_aux = df.copy()
    df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'] = df_aux['pagamento_mensal']*df_aux['qt_parcelas']

    df_aux = df_aux[['situacao_do_emprestimo', 'estado', 'valor_emprestimo_solicitado', 'valor_emprestimo_solicitado_com_taxa_de_juros',
                    'taxa_de_juros', 'faturamento_anual', 'comprometimento_de_renda_anual', 'subclasse_de_emprestimo', 'grau_de_emprestimo']].copy()

    for num in df_aux.drop(['situacao_do_emprestimo', 'estado', 'valor_emprestimo_solicitado', 'valor_emprestimo_solicitado_com_taxa_de_juros', 'subclasse_de_emprestimo', 'grau_de_emprestimo'], axis = 1):
        df_aux[f'{num}'].fillna(df_aux[num].median(), inplace = True)
    df_aux.rename({'estado':'qt_clientes'}, axis = 1, inplace = True)


    for col in ['comprometimento_de_renda_anual', 'faturamento_anual', 'taxa_de_juros']:
        df_bad_rate = pd.read_excel(f"Credit_Policy/{col}_value_pair.xlsx").sort_values(by = f'{col}_value', ascending = True)
        df_aux[f'{col}'] = np.where(df_aux[f'{col}'] <= df_bad_rate[f'{col}_value'].values[0], df_bad_rate[f'{col}_enc'].values[0], 
                           np.where(df_aux[f'{col}'] <= df_bad_rate[f'{col}_value'].values[1], df_bad_rate[f'{col}_enc'].values[1], 
                           np.where(df_aux[f'{col}'] <= df_bad_rate[f'{col}_value'].values[2], df_bad_rate[f'{col}_enc'].values[2], 
                           np.where(df_aux[f'{col}'] <= df_bad_rate[f'{col}_value'].values[3], df_bad_rate[f'{col}_enc'].values[3], 
                           np.where(df_aux[f'{col}'] <= df_bad_rate[f'{col}_value'].values[4], df_bad_rate[f'{col}_enc'].values[4], 
                           np.where(df_aux[f'{col}'] <= df_bad_rate[f'{col}_value'].values[5], df_bad_rate[f'{col}_enc'].values[5], 
                           np.where(df_aux[f'{col}'] <= df_bad_rate[f'{col}_value'].values[6], df_bad_rate[f'{col}_enc'].values[6], 
                           np.where(df_aux[f'{col}'] <= df_bad_rate[f'{col}_value'].values[7], df_bad_rate[f'{col}_enc'].values[7], 
                           np.where(df_aux[f'{col}'] <= df_bad_rate[f'{col}_value'].values[8], df_bad_rate[f'{col}_enc'].values[8], 
                           df_bad_rate[f'{col}_enc'].values[9])))))))))

    for col in ['grau_de_emprestimo']:
        df_bad_rate = pd.read_excel(f"Credit_Policy/{col}_value_pair.xlsx").sort_values(by = f'{col}_value', ascending = True)
        df_aux[f'{col}'] = np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[0], df_bad_rate[f'{col}_enc'].values[0], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[1], df_bad_rate[f'{col}_enc'].values[1], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[2], df_bad_rate[f'{col}_enc'].values[2], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[3], df_bad_rate[f'{col}_enc'].values[3], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[4], df_bad_rate[f'{col}_enc'].values[4], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[5], df_bad_rate[f'{col}_enc'].values[5],         
                           df_bad_rate[f'{col}_enc'].values[6]))))))

    for col in ['subclasse_de_emprestimo']:
        df_bad_rate = pd.read_excel(f"Credit_Policy/{col}_value_pair.xlsx").sort_values(by = f'{col}_value', ascending = True)
        df_aux[f'{col}'] = np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[0], df_bad_rate[f'{col}_enc'].values[0], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[1], df_bad_rate[f'{col}_enc'].values[1], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[2], df_bad_rate[f'{col}_enc'].values[2], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[3], df_bad_rate[f'{col}_enc'].values[3], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[4], df_bad_rate[f'{col}_enc'].values[4], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[5], df_bad_rate[f'{col}_enc'].values[5], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[6], df_bad_rate[f'{col}_enc'].values[6], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[7], df_bad_rate[f'{col}_enc'].values[7], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[8], df_bad_rate[f'{col}_enc'].values[8], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[9], df_bad_rate[f'{col}_enc'].values[9], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[10], df_bad_rate[f'{col}_enc'].values[10], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[11], df_bad_rate[f'{col}_enc'].values[11], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[12], df_bad_rate[f'{col}_enc'].values[12], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[13], df_bad_rate[f'{col}_enc'].values[13], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[14], df_bad_rate[f'{col}_enc'].values[14], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[15], df_bad_rate[f'{col}_enc'].values[15], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[16], df_bad_rate[f'{col}_enc'].values[16], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[17], df_bad_rate[f'{col}_enc'].values[17], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[18], df_bad_rate[f'{col}_enc'].values[18], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[19], df_bad_rate[f'{col}_enc'].values[19], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[20], df_bad_rate[f'{col}_enc'].values[20], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[21], df_bad_rate[f'{col}_enc'].values[21], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[22], df_bad_rate[f'{col}_enc'].values[22], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[23], df_bad_rate[f'{col}_enc'].values[23], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[24], df_bad_rate[f'{col}_enc'].values[24], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[25], df_bad_rate[f'{col}_enc'].values[25],   
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[26], df_bad_rate[f'{col}_enc'].values[26], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[27], df_bad_rate[f'{col}_enc'].values[27], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[28], df_bad_rate[f'{col}_enc'].values[28], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[29], df_bad_rate[f'{col}_enc'].values[29], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[30], df_bad_rate[f'{col}_enc'].values[30], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[31], df_bad_rate[f'{col}_enc'].values[31], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[32], df_bad_rate[f'{col}_enc'].values[32], 
                           np.where(df_aux[f'{col}'] == df_bad_rate[f'{col}_value'].values[33], df_bad_rate[f'{col}_enc'].values[33],            
                           df_bad_rate[f'{col}_enc'].values[34]))))))))))))))))))))))))))))))))))

    df_aux['predict_proba_1'] = df_aux[[
        'taxa_de_juros', 'subclasse_de_emprestimo',
        'grau_de_emprestimo',
        'comprometimento_de_renda_anual', 'faturamento_anual']].sum(axis = 1)
    df_aux['predict_proba_1'] = np.where(df_aux['predict_proba_1'] >= 100, 100, df_aux['predict_proba_1'])
    df_aux['predict_proba_1'] = df_aux['predict_proba_1']/100
    df_aux['predict_proba_0'] = 1 - df_aux['predict_proba_1']
    df_aux['y_predict'] = np.where(df_aux['predict_proba_0']  <= 0.3, 1, 0)
    df_aux.rename({'situacao_do_emprestimo':'y_true'}, axis = 1, inplace = True)

    df_aux.head()

    y_true = df_aux[['y_true']]
    y_predict = df_aux[['y_predict']]
    y_predict_proba_0 = df_aux['predict_proba_0'].values
    y_predict_proba_1 = df_aux['predict_proba_1'].values
    y_predict_proba = pd.DataFrame({'predict_proba_0':y_predict_proba_0, 'predict_proba_1':y_predict_proba_1})

    df_predict = pd.DataFrame(
            {
                'y_true':df_aux['y_true'].values,
                'y_predict':df_aux['y_predict'].values,
                'y_predict_proba_0':y_predict_proba_0,
                'y_predict_proba_1':y_predict_proba_1,
            }
        )

    return y_true, y_predict, y_predict_proba, df_predict

def corte_probabilidade_politica(df_politica):
    list_threshold = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    list_lucro = []
    for threshold in list_threshold:
        df_politica['predict_proba_1'] = df_politica[[
            'taxa_de_juros_enc', 'subclasse_de_emprestimo_enc',
            'grau_de_emprestimo_enc',
            'comprometimento_de_renda_anual_enc', 'faturamento_anual_enc']].sum(axis = 1)
        df_politica['predict_proba_1'] = np.where(df_politica['predict_proba_1'] >= 100, 100, df_politica['predict_proba_1'])
        df_politica['predict_proba_1'] = df_politica['predict_proba_1']/100
        df_politica['predict_proba_0'] = 1 - df_politica['predict_proba_1']
        df_politica['y_predict'] = np.where(df_politica['predict_proba_0']  <= threshold, 1, 0)
        y_predict = df_politica['y_predict'].values

        lucro = retorno_financeiro_politica_credito(df_politica, y_predict)[0]
        list_lucro.append(lucro)
        df_politica.drop(['predict_proba_1', 'predict_proba_0', 'y_predict'], axis = 1, inplace = True)
    
    corte_probabilidade = pd.DataFrame({'threshold':list_threshold, 'lucro':list_lucro})
    return corte_probabilidade

def metricas_politica_credito(Politica, y_train, y_predict_train, y_test, y_predict_test, y_predict_proba_train, y_predict_proba_test):

    predict_proba_train = y_predict_proba_train
    predict_proba_test = y_predict_proba_test

    # Treino
    accuracy_train = accuracy_score(y_train, y_predict_train)
    precision_train = precision_score(y_train, y_predict_train)
    recall_train = recall_score(y_train, y_predict_train)
    f1_train = f1_score(y_train, y_predict_train)
    roc_auc_train = roc_auc_score(y_train['y_true'], predict_proba_train['predict_proba_1'])
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train['y_true'], predict_proba_train['predict_proba_1'])
    ks_train = max(tpr_train - fpr_train)
    metricas_treino = pd.DataFrame({'Acuracia': accuracy_train, 'Precisao': precision_train, 'Recall': recall_train, 'F1-Score': f1_train, 'AUC': roc_auc_train, 'KS': ks_train, 'Etapa': 'treino', 'Politica': Politica}, index=[0])
    
    # Teste
    accuracy_test = accuracy_score(y_test, y_predict_test)
    precision_test = precision_score(y_test, y_predict_test)
    recall_test = recall_score(y_test, y_predict_test)
    f1_test = f1_score(y_test, y_predict_test)
    roc_auc_test = roc_auc_score(y_test['y_true'], predict_proba_test['predict_proba_1'])
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test['y_true'], predict_proba_test['predict_proba_1'])
    ks_test = max(tpr_test - fpr_test)
    metricas_teste = pd.DataFrame({'Acuracia': accuracy_test, 'Precisao': precision_test, 'Recall': recall_test, 'F1-Score': f1_test, 'AUC': roc_auc_test, 'KS': ks_test, 'Etapa': 'teste', 'Politica': Politica}, index=[0])
    
    # Consolidando
    metricas_finais = pd.concat([metricas_treino, metricas_teste])

    return metricas_finais

def auc_ks_politica(Politica, target, 
                                    y_train, y_predict_train, 
                                    y_test, y_predict_test, 
                                    predict_proba_train, predict_proba_test):

    # Inicialize as variáveis x_max_ks e y_max_ks fora dos blocos condicionais
    x_max_ks_train, y_max_ks_train = 0, 0
    x_max_ks_test, y_max_ks_test = 0, 0

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

    # Plot ROC and KS curves side by side
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Training set ROC curve
    axs[0].plot(fpr_train, tpr_train, label='Train ROC Curve (AUC = {:.2f})'.format(auc_train), color='purple')
    axs[0].fill_between(fpr_train, 0, tpr_train, color='gray', alpha=0.3)  # Preencha a área sob a curva ROC
    axs[0].plot([0, 1], [0, 1], linestyle='--', color='black')
    axs[0].set_xlabel('False Positive Rate', fontsize = 14)
    axs[0].set_ylabel('True Positive Rate', fontsize = 14)
    axs[0].set_title(f'ROC Curve - {Politica}', fontsize = 14)

    # Test set ROC curve
    axs[0].plot(fpr_test, tpr_test, label='Test ROC Curve (AUC = {:.2f})'.format(auc_test), color='orange')
    axs[0].fill_between(fpr_test, 0, tpr_test, color='gray', alpha=0.3)  # Preencha a área sob a curva ROC

    # Adicione a legenda personalizada com cores para a curva ROC
    roc_legend_labels = [
        {'label': 'Train ROC Curve (AUC = {:.2f})'.format(auc_train), 'color': 'purple', 'marker': 'o'},
        {'label': 'Test ROC Curve (AUC = {:.2f})'.format(auc_test), 'color': 'orange', 'marker': 's'},
    ]

    # Criar marcadores personalizados para a legenda ROC
    roc_legend_handles = [Line2D([0], [0], marker=label_info['marker'], color='w', markerfacecolor=label_info['color'], markersize=10) for label_info in roc_legend_labels]

    # Adicione a legenda personalizada ao gráfico da curva ROC
    roc_legend = axs[0].legend(handles=roc_legend_handles, labels=[label_info['label'] for label_info in roc_legend_labels], loc='upper right', bbox_to_anchor=(0.9, 0.4), fontsize='11')
    roc_legend.set_title('ROC AUC', prop={'size': '11'})


    # Train set KS curve
    axs[1].plot(results_train_sorted['Cumulative Perc Population'], results_train_sorted['Cumulative Perc Good'], label='Train Positive Class (Class 1)', color='purple')
    axs[1].plot(results_train_sorted['Cumulative Perc Population'], results_train_sorted['Cumulative Perc Bad'], label='Train Negative Class (Class 0)', color='purple')
    axs[1].plot([x_max_ks_train, x_max_ks_train], [y_min_ks_train, y_max_ks_train], color='black', linestyle='--')
    axs[1].fill_between(results_train_sorted['Cumulative Perc Population'], results_train_sorted['Cumulative Perc Good'], results_train_sorted['Cumulative Perc Bad'], color='gray', alpha=0.5)
    axs[1].text(x=results_train_sorted['Cumulative Perc Population'].iloc[np.argmax(results_train_sorted['Cumulative Perc Good'] - results_train_sorted['Cumulative Perc Bad'])],
                y=(y_min_ks_train + results_train_sorted['Cumulative Perc Good'].iloc[np.argmax(results_train_sorted['Cumulative Perc Good'] - results_train_sorted['Cumulative Perc Bad'])]) / 2,
                s=str(KS_train), fontsize = 14, color='purple', ha='left', va='center', rotation=45)
    axs[1].set_xlabel('Cumulative Percentage of Population', fontsize = 14)
    axs[1].set_ylabel('Cumulative Percentage', fontsize = 14)
    axs[1].set_title(f'KS Plot - {Politica}', fontsize = 14)

    # Test set KS curve
    axs[1].plot(results_test_sorted['Cumulative Perc Population'], results_test_sorted['Cumulative Perc Good'], label='Test Positive Class (Class 1)', color='orange')
    axs[1].plot(results_test_sorted['Cumulative Perc Population'], results_test_sorted['Cumulative Perc Bad'], label='Test Negative Class (Class 0)', color='orange')
    axs[1].plot([x_max_ks_test, x_max_ks_test], [y_min_ks_test, y_max_ks_test], color='black', linestyle='--')
    axs[1].fill_between(results_test_sorted['Cumulative Perc Population'], results_test_sorted['Cumulative Perc Good'], results_test_sorted['Cumulative Perc Bad'], color='gray', alpha=0.5)
    axs[1].text(x=results_test_sorted['Cumulative Perc Population'].iloc[np.argmax(results_test_sorted['Cumulative Perc Good'] - results_test_sorted['Cumulative Perc Bad'])],
                y=(y_min_ks_test + results_test_sorted['Cumulative Perc Good'].iloc[np.argmax(results_test_sorted['Cumulative Perc Good'] - results_test_sorted['Cumulative Perc Bad'])]) / 2,
                s=str(KS_test), fontsize = 14, color='orange', ha='left', va='center', rotation=45)
    axs[1].set_xlabel('Cumulative Percentage of Population', fontsize = 14)
    axs[1].set_ylabel('Cumulative Percentage', fontsize = 14)
    axs[1].set_title(f'KS Plot - {Politica}', fontsize = 14)

    # Adicione a legenda personalizada com cores
    ks_legend_labels = [
        {'label': f'Treino (KS: {KS_train})', 'color': 'purple', 'marker': 'o'},
        {'label': f'Teste (KS: {KS_test})', 'color': 'orange', 'marker': 's'},
    ]

    # Criar marcadores personalizados para a legenda
    legend_handles = [Line2D([0], [0], marker=label_info['marker'], color='w', markerfacecolor=label_info['color'], markersize=10) for label_info in ks_legend_labels]

    ks_legend = axs[1].legend(handles=legend_handles, labels=[label_info['label'] for label_info in ks_legend_labels], loc='upper right', bbox_to_anchor=(0.9, 0.4), fontsize='11')
    ks_legend.set_title('KS', prop={'size': '11'})

    plt.tight_layout()
    plt.show()

def retorno_financeiro_politica_credito(df, y_predict):

    df_aux = df.copy()
    df_aux['y_predict'] = y_predict

    TN = df_aux.loc[(df_aux['situacao_do_emprestimo'] == 0) & (df_aux['y_predict'] == 0)].shape[0] # O CARA É BOM E MEU MODELO FALA QUE ELE É BOM
    FN = df_aux.loc[(df_aux['situacao_do_emprestimo'] == 1) & (df_aux['y_predict'] == 0)].shape[0] # O CARA É MAU E MEU MODELO FALA QUE ELE É BOM
    FP = df_aux.loc[(df_aux['situacao_do_emprestimo'] == 0) & (df_aux['y_predict'] == 1)].shape[0] # O CARA É BOM E MEU MODELO FALA QUE É MAU
    TP = df_aux.loc[(df_aux['situacao_do_emprestimo'] == 1) & (df_aux['y_predict'] == 1)].shape[0] # O CARA É MAU E O MEU MODELO FALA QUE É MAU

    df_aux['caso'] = np.where((df_aux['situacao_do_emprestimo'] == 0) & (df_aux['y_predict'] == 0), 'Verdadeiro Negativo (Cliente Bom | Modelo classifica como Bom) - Ganho a Diferença entre Valor Bruto e Valor com Juros', # Ganha a Diferença entre Valor Bruto e Valor com Juros
                        np.where((df_aux['situacao_do_emprestimo'] == 1) & (df_aux['y_predict'] == 0), 'Falso Negativo (Cliente Mau | Modelo classifica como Bom) - Perco o valor emprestado', # Perde o valor emprestado
                        np.where((df_aux['situacao_do_emprestimo'] == 0) & (df_aux['y_predict'] == 1), 'Falso Positivo (Cliente Bom | Modelo classifica como Mau) - Deixo de ganhar a diferença entre Valor Bruto e Valor com Juros', # Deixo de ganhar a diferença entre Valor Bruto e Valor com Juros
                        'Verdadeiro Positivo (Cliente Mau | Modelo classifica como Mau) - Não ganho nada' # Não ganho nada
    )))

    df_aux['retorno_financeiro'] = np.where((df_aux['situacao_do_emprestimo'] == 0) & (df_aux['y_predict'] == 0), df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'] - df_aux['valor_emprestimo_solicitado'], # Ganha a Diferença entre Valor Bruto e Valor com Juros
                        np.where((df_aux['situacao_do_emprestimo'] == 1) & (df_aux['y_predict'] == 0), df_aux['valor_emprestimo_solicitado']*(-1), # Perde o valor emprestado
                        np.where((df_aux['situacao_do_emprestimo'] == 0) & (df_aux['y_predict'] == 1), 0, # Deixo de ganhar a diferença entre Valor Bruto e Valor com Juros (df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'] - df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'])*(-1)
                        0 # Não ganho nada
    )))

    valor_de_exposicao_total = int(df_aux['valor_emprestimo_solicitado'].sum())
    retorno_financeiro = int(df_aux['retorno_financeiro'].sum())
    valor_conquistado = valor_de_exposicao_total + retorno_financeiro
    return_on_portfolio = round((retorno_financeiro/valor_de_exposicao_total)*100, 2)
    retorno_financeiro_por_caso = df_aux.groupby('caso', as_index = False)['retorno_financeiro'].sum().sort_values(by = 'retorno_financeiro', ascending = False)

    # Crie um DataFrame a partir dos hiperparâmetros
    df = retorno_financeiro_por_caso.reset_index(drop=True)
    df = df.round(2)

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
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: white; font-size: 14px')\
        .applymap(color_etapa, subset=pd.IndexSlice[:, :])\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px')\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px')\
        .set_table_styles([
            {'selector': 'thead', 'props': [('color', 'black'), ('font-weight', 'bold'), ('background-color', 'lightgray')]}
        ])

    return retorno_financeiro, styled_df, valor_de_exposicao_total, return_on_portfolio

def retorno_financeiro_politica(df, y_true, y_predict):

    df_aux = df[['qt_parcelas', 'pagamento_mensal', 'valor_emprestimo_solicitado']].copy()
    df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'] = df_aux['pagamento_mensal']*df_aux['qt_parcelas']
    df_aux['y_true'] = y_true['y_true'].values
    df_aux['y_predict'] = y_predict['y_predict'].values

    df_aux['caso'] = np.where((df_aux['y_true'].values == 0) & (df_aux['y_predict'].values == 0),'Verdadeiro Negativo (Cliente Bom | Modelo classifica como Bom) - Ganho a Diferença entre Valor Bruto e Valor com Juros',
                    np.where((df_aux['y_true'].values == 1) & (df_aux['y_predict'].values == 0),'Falso Negativo (Cliente Mau | Modelo classifica como Bom) - Perco o valor emprestado',
                    np.where((df_aux['y_true'].values == 0) & (df_aux['y_predict'].values == 1),'Falso Positivo (Cliente Bom | Modelo classifica como Mau) - Deixo de ganhar a diferença entre Valor Bruto e Valor com Juros','Verdadeiro Positivo (Cliente Mau | Modelo classifica como Mau) - Não ganho nada'
    )))


    df_aux['retorno_financeiro'] = np.where((df_aux['y_true'].values == 0) & (df_aux['y_predict'].values == 0),df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'] - df_aux['valor_emprestimo_solicitado'],
                                np.where((df_aux['y_true'].values == 1) & (df_aux['y_predict'].values == 0),df_aux['valor_emprestimo_solicitado'] * (-1),
                                np.where((df_aux['y_true'].values == 0) & (df_aux['y_predict'].values == 1),0,0
    )))

    valor_de_exposicao_total = int(df_aux['valor_emprestimo_solicitado'].sum())
    retorno_financeiro = int(df_aux['retorno_financeiro'].sum())
    valor_conquistado = valor_de_exposicao_total + retorno_financeiro
    return_on_portfolio = round((retorno_financeiro/valor_de_exposicao_total)*100, 2)
    retorno_financeiro_por_caso = df_aux.groupby('caso', as_index = False)['retorno_financeiro'].sum().sort_values(by = 'retorno_financeiro', ascending = False)

    # Crie um DataFrame a partir dos hiperparâmetros
    df = retorno_financeiro_por_caso.reset_index(drop=True)
    df = df.round(2)

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
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: white; font-size: 14px')\
        .applymap(color_etapa, subset=pd.IndexSlice[:, :])\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px')\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px')\
        .set_table_styles([
            {'selector': 'thead', 'props': [('color', 'black'), ('font-weight', 'bold'), ('background-color', 'lightgray')]}
        ])

    return retorno_financeiro, styled_df, valor_de_exposicao_total, return_on_portfolio

def metricas_politica_final(Politica, df, df_grana, y, y_predict, y_predict_proba):

    # Amostra Final
    accuracy = accuracy_score(y, y_predict)
    precision = precision_score(y, y_predict)
    recall = recall_score(y, y_predict)
    f1 = f1_score(y, y_predict)
    roc_auc = roc_auc_score(y['y_true'], y_predict_proba['y_predict_proba_1'])
    fpr, tpr, thresholds = roc_curve(y['y_true'], y_predict_proba['y_predict_proba_1'])
    ks = max(tpr - fpr)
    total, retorno_financeiro_por_caso, valor_de_exposicao_total, return_on_portfolio = retorno_financeiro_politica(df_grana, y, y_predict)
    total = 'R$' + str(int(round(total/1000000, 0))) + ' MM'
    valor_de_exposicao_total = 'R$' + str(float(round(valor_de_exposicao_total/1000000000, 3))) + 'B'
    rocp = str(return_on_portfolio) + '%'
    metricas_finais = pd.DataFrame({
        # 'Acuracia': accuracy, 
        # 'Precisao': precision, 
        # 'Recall': recall, 
        # 'F1-Score': f1, 
        # 'AUC': roc_auc, 
        # 'KS': ks, 
        'Etapa': 'Amostra Final', 
        'Método': Politica, 
        'Valor Total de Exposição': valor_de_exposicao_total,
        'Retorno Financeiro': total,
        'Return on Credit Portfolio (ROCP)': rocp
    }, index=[0])

    df = metricas_finais.reset_index(drop=True)
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
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: white; font-size: 14px', subset=pd.IndexSlice[:, 'Etapa':])\
        .applymap(color_etapa, subset=pd.IndexSlice[:, 'Etapa':])\
        .set_table_styles([
            {'selector': 'thead', 'props': [('color', 'black'), ('font-weight', 'bold'), ('background-color', 'lightgray')]}
        ])

    # Mostrando o DataFrame estilizado
    return styled_df

def retorno_financeiro_swap_in_swap_out(df):

    df_aux = df.copy()

    TN = df_aux.loc[(df_aux['y_true'] == 0) & (df_aux['y_predict_test_best_clf'] == 0)].shape[0] # O CARA É BOM E MEU MODELO FALA QUE ELE É BOM
    FN = df_aux.loc[(df_aux['y_true'] == 1) & (df_aux['y_predict_test_best_clf'] == 0)].shape[0] # O CARA É MAU E MEU MODELO FALA QUE ELE É BOM
    FP = df_aux.loc[(df_aux['y_true'] == 0) & (df_aux['y_predict_test_best_clf'] == 1)].shape[0] # O CARA É BOM E MEU MODELO FALA QUE É MAU
    TP = df_aux.loc[(df_aux['y_true'] == 1) & (df_aux['y_predict_test_best_clf'] == 1)].shape[0] # O CARA É MAU E O MEU MODELO FALA QUE É MAU

    df_aux['caso'] = np.where((df_aux['y_true'] == 0) & (df_aux['y_predict_test_best_clf'] == 0), 'Verdadeiro Negativo (Cliente Bom | Modelo classifica como Bom) - Ganho a Diferença entre Valor Bruto e Valor com Juros', # Ganha a Diferença entre Valor Bruto e Valor com Juros
                        np.where((df_aux['y_true'] == 1) & (df_aux['y_predict_test_best_clf'] == 0), 'Falso Negativo (Cliente Mau | Modelo classifica como Bom) - Perco o valor emprestado', # Perde o valor emprestado
                        np.where((df_aux['y_true'] == 0) & (df_aux['y_predict_test_best_clf'] == 1), 'Falso Positivo (Cliente Bom | Modelo classifica como Mau) - Deixo de ganhar a diferença entre Valor Bruto e Valor com Juros', # Deixo de ganhar a diferença entre Valor Bruto e Valor com Juros
                        'Verdadeiro Positivo (Cliente Mau | Modelo classifica como Mau) - Não ganho nada' # Não ganho nada
    )))

    df_aux['retorno_financeiro'] = np.where((df_aux['y_true'] == 0) & (df_aux['y_predict_test_best_clf'] == 0), df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'] - df_aux['valor_emprestimo_solicitado'], # Ganha a Diferença entre Valor Bruto e Valor com Juros
                        np.where((df_aux['y_true'] == 1) & (df_aux['y_predict_test_best_clf'] == 0), df_aux['valor_emprestimo_solicitado']*(-1), # Perde o valor emprestado
                        np.where((df_aux['y_true'] == 0) & (df_aux['y_predict_test_best_clf'] == 1), 0, # Deixo de ganhar a diferença entre Valor Bruto e Valor com Juros (df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'] - df_aux['valor_emprestimo_solicitado_com_taxa_de_juros'])*(-1)
                        0 # Não ganho nada
    )))

    valor_de_exposicao_total = int(df_aux['valor_emprestimo_solicitado'].sum())
    retorno_financeiro = int(df_aux['retorno_financeiro'].sum())
    valor_conquistado = valor_de_exposicao_total + retorno_financeiro
    return_on_portfolio = round((retorno_financeiro/valor_de_exposicao_total)*100, 2)
    retorno_financeiro_por_caso = df_aux.groupby('caso', as_index = False)['retorno_financeiro'].sum().sort_values(by = 'retorno_financeiro', ascending = False)

    # Crie um DataFrame a partir dos hiperparâmetros
    df = retorno_financeiro_por_caso.reset_index(drop=True)
    df = df.round(2)

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
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: white; font-size: 14px')\
        .applymap(color_etapa, subset=pd.IndexSlice[:, :])\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px')\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px')\
        .set_table_styles([
            {'selector': 'thead', 'props': [('color', 'black'), ('font-weight', 'bold'), ('background-color', 'lightgray')]}
        ])

    return retorno_financeiro, styled_df, valor_de_exposicao_total, return_on_portfolio