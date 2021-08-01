from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import iplot,plot
import dataSource as ds
import dataVisualization as dv
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# =============================================================================
#  Class Traitement
# =============================================================================

class Traitement:
    """
        La classe Traitement permet de construire les données X,y 
        d'apprentissage et de test.
    """
    
    def __init__(self, df, l_attrs_x, labels, freq_train=1000, freq_test=400, preprocessor=None):
        #DataFrame
        self.df = df
        #features
        self.l_attrs = l_attrs_x
        #targets
        self.labels = labels
        #preprocessor
        self.preprocessor = preprocessor
        #fréquences
        self.freq_train = freq_train
        self.freq_test = freq_test

        #Données d'apprentissage/test pour chaque modèle
        self.l_Xtrain = []
        self.l_Ytrain = []
        self.l_Xtest = []
        self.l_Ytest = []
        
        
    # Fonction de construction des données de train/test
    def set_data_train_test(self, train_size=0.8):
        X_train, X_test, y_train, y_test = ds.create_data_xy(self.df, train_size, self.freq_train, self.freq_test)
        
        #On vide les anciennes données s'il y en a
        self.l_Xtrain = []
        self.l_Ytrain = []
        self.l_Xtest = []
        self.l_Ytest = []
        
        for attrs in self.l_attrs:
            self.l_Xtrain.append(X_train[attrs])
            self.l_Xtest.append(X_test[attrs])
            self.l_Ytrain.append(y_train[self.labels])
            self.l_Ytest.append(y_test[self.labels])


# =============================================================================
#  Class Evaluation 
# =============================================================================

class Evaluation :
    """
        La classe Evaluation permet d'entrainer des modèles à partir de la
        classe Traitement et d'en afficher des résultats.
    """
    
    def __init__(self, models, traitement):
        self.models = models
        self.traitement = traitement
        self.preprocessor = self.traitement.preprocessor

        self.l_Xtrain = self.traitement.l_Xtrain
        self.l_Ytrain = self.traitement.l_Ytrain
        self.l_Xtest = self.traitement.l_Xtest
        self.l_Ytest = self.traitement.l_Ytest
        self.labels = self.traitement.labels
        
        #Ajout du preprocessor à la pipeline s'il y en a un
        if self.preprocessor is not None :
            self.models_pip = [make_pipeline(self.preprocessor[i], self.models[i]) for i in range(len(self.models))]
        else: 
            self.models_pip = self.models
    
        # for mi in range(len(models)):
        #     if type(models[mi]).__name__ == 'model_physique1':
        #         temp = self.l_Xtest[mi].copy()
        #         temp['index'] = temp.index
        #         f = temp.groupby(['Trip']).nth(1).reset_index()['index'].values
        #         self.l_Ytest[mi].drop(f, inplace=True)
        
        
    
    def fit(self):
        """
            Fonction qui entraine tous nos modèles.
        """
        for i in range(len(self.models)):
            self.models_pip[i].fit(self.l_Xtrain[i], self.l_Ytrain[i])
            
    def score(self):
        """
            Fonction retournant une liste de scores sur les données de test
            pour chaque modèle.
        """
        return [self.models_pip[i].score(self.l_Xtest[i], self.l_Ytest[i]) for i in range(len(self.models))]
    
    def predict(self, X):
        """
            Fonction retournant une liste de prédiction sur X pour chaque
            modèle.
        """
        return [self.models_pip[i].predict(X[i]) for i in range(len(self.models))]
  
    def getCoef(self):
        """
            Fonction retournant les paramètres appris pour chaque modèle.
        """
        return [self.models[i].coef_ for i in range(len(self.models))]
    
    def calculMse(self):
        ypred = self.predict(self.l_Xtest)
        return [mean_squared_error(self.l_Ytest[i],ypred[i]) for i in range(len(self.models))]
    
    
    # ------------------------- Fonctions d'affichage -------------------------
    
    def afficher_score(self):
        """
            Fonction affichant les scores pour chaque modèle.
        """
        scores = self.score()
        for i in range(len(self.models)):
            print(f"Score obtenu pour le modèle {type(self.models[i]).__name__ : <10} : {scores[i]}")
            
    def afficher_coef(self):
        """
            Fonction affichant les coefficients pour chaque modèle.
        """
        coefs = self.getCoef()
        for i in range(len(self.models)):
            print(f"Coefficients obtenu pour le modèle {i : <10} : {coefs[i]}")
            
    def afficher_mse(self):
        ypred = self.predict(self.l_Xtest)
        print("MSE sur les données de test:\n")
        for i in range(len(self.models)):
            print(f"MSE obtenue pour {type(self.models[i]).__name__ : <10} : {mean_squared_error(self.l_Ytest[i],ypred[i])}")
            #print(f"MSE obtenue pour {type(self.models[i]).__name__ : <10} : {np.mean((self.l_Ytest[i]-ypred[i])**2)}")
        
    def afficher_resultats(self):
        """
            Fonction appelant les autres fonctions d'affichage.
        """
        #self.afficher_score()
        print()
        self.afficher_mse()
        print()
        #self.afficher_coef()
        
    #def afficher_pred(self):
        
        
    # ----------------------------- Fonctions MSE -----------------------------
    
    def tabMSEFreq(self,  liste_freq, freq_train,train_size=0.8):
        tab_mse = []
        models = [deepcopy(m) for m in self.models]
        
        for freq in liste_freq:
            traitement  = Traitement(self.traitement.df, self.traitement.l_attrs, self.traitement.labels,
                                     freq_train, freq, self.traitement.preprocessor)
            traitement.set_data_train_test(train_size)
            
            evaluateur = Evaluation(models,traitement)
            evaluateur.fit()
            
            tab_mse.append(evaluateur.calculMse())
        
        tab_mse = np.array(tab_mse)
        """         
        #Affichage MSE pour le premier modèle
        plt.figure(figsize=(15,5))
        plt.title("Erreur MSE en fonction de la fréquence")
        plt.plot(liste_freq, tab_mse[:,0], label=type(models[0]).__name__)
        plt.xlabel("Temps entre deux points")
        plt.ylabel("MSE")
        plt.legend()
        plt.show()
        """
        #Affichage des erreurs MSE des modèles en fonction de la fréquence    

        
        for i in range(len(models)):
            plt.figure(figsize=(15,5))
            plt.plot(tab_mse[:,i], label=type(models[i]).__name__)

            plt.xticks(np.arange(len(liste_freq)), np.array(liste_freq))
            plt.xlabel("Fréquences")
            plt.xlabel("Temps entre deux points")
            plt.ylabel("MSE")
            plt.legend()
            plt.show()

        plt.figure(figsize=(10,5))
        for i in range(len(models)):
            plt.plot(tab_mse[:,i], label=type(models[i]).__name__)
            plt.xticks(np.arange(len(liste_freq)), np.array(liste_freq))
        plt.xlabel("Fréquences")
        plt.xlabel("Temps entre deux points")
        plt.ylabel("MSE")
        plt.legend()
        plt.show()
        #Tableau des erreurs MSE en DataFrame
        columns = [type(m).__name__ for m in models]
        errMSE = pd.DataFrame(tab_mse, columns=columns, index=liste_freq)
        
        return errMSE
    
    
    def matMSECase(self, freq_train, freq_test, lat_min, long_min, e_x, e_y, min_datapts=20, train_size=0.8, n_interval=10):      
        #Copie des modèles
        models = [deepcopy(m) for m in self.models]
        # liste matrices erreurs des cases
        l_mat_err= [np.zeros((n_interval, n_interval)) for i in range(len(models))]
            
        df = self.traitement.df
        
        #Opérations pour stocker les MSE par effectif et par case
        eff = np.unique(df["Effectif_case"])
        ind_eff = {eff[i]:i for i in range(len(eff))}
        
        vit = np.unique(df["Vitesse_moy_case"])
        ind_vit = {vit[i]:i for i in range(len(vit))}
        
        var = np.unique(df["Vitesse_var_case"])
        ind_var = {var[i]:i for i in range(len(var))}
        
        l_mse_eff = [np.zeros(len(eff)) for _ in range(len(models))]
        l_mse_vit = [np.zeros(len(vit)) for _ in range(len(models))]
        l_mse_var = [np.zeros(len(var)) for _ in range(len(models))]
        
        eff_count = [np.zeros(len(eff)) for _ in range(len(models))]
        vit_count = [np.zeros(len(vit)) for _ in range(len(models))]
        var_count = [np.zeros(len(var)) for _ in range(len(models))]
        
        # parcours de toutes les cases
        for i in range(n_interval):
            for j in range(n_interval):
                # récupération des données de la case
                case_df=ds.trouve_data_case(df, (i, j), lat_min, long_min, e_x, e_y)

                #On prend les Trips qui ont au moins $min_datapoints$ points
                #c'est pas au moins 2 points car tu splits en train et en test, ca aura moins d'un point 
                ctrips, ccounts = np.unique(case_df["Trip"], return_counts=True)
                ctrips = ctrips[ccounts>min_datapts]
                case_df = case_df[case_df['Trip'].isin(ctrips)]

                #Cases qui ont au moins 2 trips
                if len(pd.unique(case_df["Trip"])) > 1 :
                    traitement = Traitement(case_df, self.traitement.l_attrs, self.traitement.labels, 
                                            freq_train, freq_test, self.traitement.preprocessor)
                    traitement.set_data_train_test(train_size)
    
                    l_ypred = self.predict(traitement.l_Xtest)
    
                    for mi in range(len(models)):

                        mse_ij = mean_squared_error(traitement.l_Ytest[mi],l_ypred[mi])
                        l_mat_err[mi][n_interval-1-i, j] = mse_ij
                        
                        ei = ind_eff[pd.unique(df['Effectif_case'].loc[case_df.index])[0]]
                        vi = ind_vit[pd.unique(df['Vitesse_moy_case'].loc[case_df.index])[0]]
                        vi2 = ind_var[pd.unique(df['Vitesse_var_case'].loc[case_df.index])[0]]
                        l_mse_eff[mi][ei] += mse_ij
                        l_mse_vit[mi][vi] += mse_ij
                        l_mse_var[mi][vi2] += mse_ij
                        eff_count[mi][ei] += 1
                        vit_count[mi][vi] += 1
                        var_count[mi][vi2] += 1
                        
                        
        for mi in range(len(models)):
            tmp = np.where(eff_count[mi] != 0)[0]
            l_mse_eff[mi][tmp] /= eff_count[mi][tmp]
            tmp = np.where(vit_count[mi] != 0)[0]
            l_mse_vit[mi][tmp] /= vit_count[mi][tmp]
            tmp = np.where(var_count[mi] != 0)[0]
            l_mse_var[mi][tmp] /= var_count[mi][tmp]
        
        fig, ax = plt.subplots(2,2, figsize=(15,13))     
        for m in range(len(l_mat_err)):
            # fig, ax = plt.subplots(3,2, figsize=(13,13))
            #fig.suptitle(f'{type(models[m]).__name__}', fontsize=16)
            
            ax[m//2][m%2].set_title(f"Erreur MSE par case : {type(models[m]).__name__}")
            # sns.heatmap(l_mat_err[m], linewidths=.5,annot=True, cmap="YlGnBu", yticklabels=np.arange(n_interval-1, -1, -1), ax=ax[0][0])
            sns.heatmap(l_mat_err[m], linewidths=.5,annot=True, cmap="YlGnBu", yticklabels=np.arange(n_interval-1, -1, -1), ax=ax[m//2][m%2])
            # ax[0][1].set_title("Histogramme des valeurs MSE")
            # val = l_mat_err[m].ravel()[l_mat_err[m].ravel() != 0]
            # sns.histplot(val, ax=ax[0][1])
            
            # ax[1][0].set_title("Histplot MSE moy par effectif")
            # h1 = sns.histplot(x=ind_eff.keys(), y=l_mse_eff[m], ax=ax[1][0], cmap="RdPu", cbar=True)
            # h1.set(xlabel='Effectif', ylabel='MSE')
            
            # ax[1][1].set_title("Histplot MSE moy par vitesse moy")
            # h2 = sns.histplot(x=ind_vit.keys(), y=l_mse_vit[m], ax=ax[1][1], cmap="YlOrRd", cbar=True)
            # h2.set(xlabel='Vitesse_moy', ylabel='MSE')
                   
            # ax[2][0].set_title("Histplot MSE moy par variance vitesse")
            # h3 = sns.histplot(x=ind_var.keys(), y=l_mse_var[m], ax=ax[2][0], cmap="YlOrRd", cbar=True)
            # h3.set(xlabel='Variance_vit', ylabel='MSE')
            
            # fig.delaxes(ax[2][1])
            
        plt.show()
        
    def scatterPred(self, begin_point, end_point):
        models = [deepcopy(m) for m in self.models]
        
        txt = [f"Point n°{t}" for t in range(end_point-begin_point)]
        trace_0 = go.Scatter(x=self.l_Xtest[0]['Latitude'].iloc[begin_point:end_point], y=self.l_Xtest[0]['Longitude'].iloc[begin_point:end_point], mode="lines",name="Xtest", text=txt)
        trace_1 = go.Scatter(x=self.l_Ytest[0].iloc[begin_point:end_point,0], y=self.l_Ytest[0].iloc[begin_point:end_point,1], mode="lines+markers", name="Target", text=txt)
        data = [trace_0,trace_1]
        l_mse = []
        
        for mi in range(len(models)):
            ypred = models[mi].predict(self.l_Xtest[mi])[begin_point:end_point]
            y = self.l_Ytest[mi].iloc[begin_point:end_point].to_numpy()
            # mse = (ypred-y)**2
            mse = [mean_squared_error(y[i], ypred[i]) for i in range(len(y))]
            #txt = [f"Point n°{i}<br>MSE_Lat = {mse[i,0]}<br>MSE_Long = {mse[i,1]}" for i in range(len(mse))]
            txt = [f"Point n°{i}<br>MSE = {mse[i]}" for i in range(len(mse))]
            data.append(go.Scatter(x=ypred[:,0], y=ypred[:,1], mode="lines+markers", name=type(models[mi]).__name__, text=txt))
            l_mse.append(np.sum(mse))
            
        layout = go.Layout(
            title='Targets et Predictions',
            xaxis = dict(
                title='Latitude',
                ticklen = 5,
                showgrid = True,
                zeroline = False
            ),
            yaxis = dict(
                title='Logitude',
                ticklen=5,
                showgrid=True,
                zeroline=False,
            )
        )

        fig = go.Figure(data=data, layout=layout)
        iplot(fig, filename="ScatterPred")
        
        return l_mse