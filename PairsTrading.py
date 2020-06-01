import numpy as np
from sklearn.cluster import KMeans
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint as dickeyFuller
from numpy import linalg as la
from collections import OrderedDict
import copy as cp



### Titolo: Pairs Trading
### Autore: Davide Roznowicz
## Note utili: -il metodo OnData è stato approssimativamente diviso in 5 classi (accompagnate da una sintetica descrizione), così come descritto nella tesi
##             -con simbolo si intende equivalentemente il codice identificativo relativo al titolo
##             -dove possibile sono state utilizzate le funzioni interne dei pacchetti statistici di Python, dato che l'algoritmo di backtest è già di per sè sufficientemente lungo


### Classe (obbligatoria) che genera l'istanza relativa al backtest
class PairsTrading(QCAlgorithm):
    
    def Initialize(self):


        BarPeriod = TimeSpan.FromDays(1)  # estensione di ogni periodo della rolling window : 1 giorno (sono ammessi Resolution.Minutes/Hours/Daily)
        RollingWindowSize = 60  # La rolling window ha estensione 60
        self.Data = {}
        

        self.EquitySymbols = ['TDG', 'EQT', 'LMT', 'NEM', 'VLO', 'BP', 'WDAY', 'ABBV', 'PCG', 'LNC', 'URI', 'DRI', 'AZO', 'DFS', 'DAL', 'PG', 'VZ', 'KMX', 'OKE', 'CMG', 'PGR', 'PCLN', 'NOC', 'PNR', 'VMW', 'WHR', 'HES', 'JNPR', 'MHK', 'ALL', 'MDLZ', 'ORLY', 'RSG', 'GM', 'MAS', 'ICE', 'CXO', 'UTX', 'HUN', 'MS', 'R', 'BIIB', 'SWKS', 'HPQ', 'CY', 'ROST', 'GOOGL', 'MYL', 'VIPS', 'NBL', 'CVS', 'PEG', 'PCAR', 'BBT', 'AMBA', 'FTI', 'BXP', 'TSLA', 'LBTYA', 'HCP', 'PXD', 'PPG', 'X', 'UA', 'EQIX', 'ABT', 'PHM', 'ISIS', 'M', 'WDC', 'SNY', 'BEN', 'ABC', 'CMS', 'MCO', 'MOS', 'PSA', 'GE', 'PEP', 'DVN', 'CERN', 'MMM', 'MNK', 'NSC', 'TROW', 'DHI', 'COP', 'TPX', 'MO', 'SRCL', 'ADP', 'FEYE', 'ALGN', 'TXT', 'JNJ', 'DLTR', 'GOOG', 'EIX', 'ES', 'APA', 'FTNT', 'TEVA', 'NKE', 'T', 'SCCO', 'CMI', 'BBBY', 'WM', 'AKAM', 'FIS', 'ADBE', 'QRVO', 'NEE', 'GME', 'MAT', 'PFPT', 'LB', 'NI', 'MDT', 'SINA', 'HIG', 'SPLK', 'CAR', 'PII', 'KSS', 'AEM', 'TSM', 'KSU', 'YUM', 'GWW', 'MUR', 'RHI', 'CNQ', 'CME', 'WCC', 'COST', 'USB', 'ITEK', 'CPN', 'MGM', 'TWTR', 'SCHW', 'AGN', 'NTAP', 'CTL', 'RDS.A', 'ALNY', 'BABA', 'FDX', 'BMY', 'CMCSA', 'XLNX', 'RY', 'SKX', 'RRC', 'IBM', 'HCA', 'BAX', 'TEL', 'REGN', 'ULTA', 'CHKP', 'PRGO', 'FCAU', 'CF', 'GSK', 'HLT', 'MAR', 'CYBR', 'GD', 'DUK', 'CRZO', 'CBS', 'MLM', 'PH', 'DTE', 'FBHS', 'PRU', 'LOGM', 'SU', 'VIAB', 'APD', 'NE', 'MEOH', 'MRK', 'CTSH', 'STLD', 'GILD', 'PSX', 'LBTYK', 'GLW', 'GT', 'CSCO', 'CE', 'ENDP', 'NTRS', 'RTN', 'FAST', 'V', 'CI', 'OAS', 'NFLX', 'TXN', 'EW', 'BX', 'SBAC', 'BHP', 'BPOP', 'FET', 'GRUB', 'HOLX', 'DOV', 'COG', 'ORCL', 'MET', 'KR', 'UAL', 'ETFC', 'MA', 'NTCT', 'CP', 'SIVB', 'INCY', 'JD', 'JWN', 'NXPI', 'SEE', 'TJX', 'JAZZ', 'FNV', 'ESV', 'RL', 'RIO', 'FLEX', 'BDX', 'ILMN', 'ADM', 'EOG', 'OMC', 'CHE', 'DB', 'BLUE', 'EQR', 'SAVE', 'MRO', 'BWA', 'CNX', 'CNI', 'MNST', 'BUD', 'PLD', 'RS', 'PTEN', 'XOM', 'WBC', 'RDUS', 'CLR', 'JCI', 'OI', 'TMK', 'FB', 'CCL', 'VOD', 'SAP', 'CELG', 'SLB', 'PYPL', 'ETN', 'WMB', 'NVS', 'LUV', 'AAPL', 'IR', 'FFIV', 'RE', 'PNC', 'TOT', 'KMB', 'TSCO', 'VRTX', 'DISH', 'YELP', 'ASML', 'POT', 'CTXS', 'LOW', 'Z', 'FITB', 'HZNP', 'SHW', 'LLY', 'RCL', 'HD', 'HST', 'CMA', 'LEN', 'ADI', 'FLR', 'INFY', 'CAH', 'UNH', 'AAP', 'KORS', 'YHOO', 'DE', 'DLR', 'AMP', 'EBAY', 'TRIP', 'HFC', 'AER', 'GPS', 'AR', 'VFC', 'KO', 'MSFT', 'NRG', 'PBCT', 'AXP', 'GPOR', 'LNG', 'AVGO', 'EMN', 'KMI', 'TIF', 'MSI', 'HP', 'COH', 'LVS', 'RF', 'SWK', 'PAYX', 'AMGN', 'PFG', 'VMC', 'BAC', 'BIDU', 'AIG', 'TGT', 'KLAC', 'MGA', 'CHTR', 'FANG', 'NVDA', 'TRV', 'EPD', 'PKG', 'SBUX', 'WYNN', 'ESPR', 'ROK', 'C', 'TTM', 'HON', 'CLB', 'CSX', 'COF', 'BLK', 'DEO', 'SWN', 'CCI', 'JBHT', 'STZ', 'BRK.B', 'TERP', 'EL', 'DHR', 'SLW', 'MAC', 'WFC', 'PANW', 'MCHP', 'ISRG', 'BMRN', 'AMTD', 'GIS', 'ZION', 'TMO', 'HUM', 'ROP', 'EA', 'TSN', 'HAS', 'HBAN', 'BSX', 'FLT', 'AFL', 'ALK', 'SYNA', 'WMT', 'NOW', 'VRX', 'UPS', 'PX', 'AAN', 'INTC', 'STX', 'MCK', 'LULU', 'JBLU', 'ASH', 'XEC', 'FISV', 'LEA', 'AEP', 'MU', 'LRCX', 'ADS', 'CAT', 'MPC', 'ECL', 'KEY', 'TRN', 'MMC', 'DGX', 'GS', 'VRSN', 'EXC', 'LYB', 'PM', 'K', 'NCR', 'EFX', 'RIG', 'SO', 'STI', 'PFE', 'DG', 'PAA', 'AZN', 'AMX', 'MCD', 'CNC', 'HAL', 'AMT', 'FIT', 'CRM', 'NBR', 'CVX', 'FCX', 'VTR', 'UN', 'UNM', 'WBA', 'HCN', 'F', 'BA', 'ZTS', 'EXPE', 'MXIM', 'GPRO', 'TOL', 'SYY', 'NUE', 'HDS', 'ITW', 'DIS', 'JPM', 'TMUS', 'CRUS', 'NOV', 'CL', 'ATVI', 'ANTM', 'AMAT', 'ED', 'SPG', 'ZBH', 'EMR', 'SYK', 'ACN', 'L', 'STT', 'NTES', 'AAL', 'AMZN', 'IP', 'D', 'FSLR', 'OII', 'QCOM', 'PPL', 'KHC', 'CFX', 'BK', 'XRX', 'KKR', 'UNP', 'ACE', 'UTHR', 'OXY', 'HOG', 'TRGP', 'CTRP', 'ALXN', 'WLL']
        self.CashStart = 100000  # denaro di partenza
        self.SetStartDate(2015, 8, 1)
        self.SetEndDate(2018, 3, 1)
        self.SetCash(self.CashStart)
        
        
        self.kclusters = 20  # numero di clusters utilizzati (in accordo con ciò che si è detto nella tesi per 500 titoli)
        self.TradableGroupDictionary = {}  # dizionario : assegnato successivamente a self.SubGroups appartenente a Cluster()
        self.PositionManagement = PositionManagement()
        self.maxPositions = 3  # scegliamo un numero massimo di posizioni distinte da avere in portafoglio in un qualsiasi momento (in questa sede, con posizione si fa riferimento ad un gruppetto distinto di elementi cointegrati, quindi può significare comprare/vendere diverse azioni distinte)
        
        self.Xcompleted = False   # indica se la matrice X è stata aggiornata completamente o meno
        self.counter = 1   # contatore
        self.FirstTime=True   # variabile indicatrice: permette di entrare solo una volta in un ciclo, prima di essere settata a False
        self.RebuildSubGroups=1   # variabile che dice ogni quanto ricondurre completamente l'analisi: altrimenti si usano le analisi precedentemente condotte precedentemente costruiti


        i=-1
        for symbol in self.EquitySymbols:
            i += 1
            equity = self.AddEquity(symbol, Resolution.Daily)
            self.Data[str(equity.Symbol.ID)] = SymbolData(equity.Symbol, BarPeriod,RollingWindowSize)  # assegniamo come value del dizionario un'istanza appena creata di SymbolData
            self.EquitySymbols[i] = str(equity.Symbol.ID)   # sovrascriviamo la lista con l'identificazione univoca dei simboli di ogni titolo (così, anche se i ticker cambiano, l' ID adottato rimane)
            self.Securities[ self.Data[str(equity.Symbol.ID)].Symbol.Value ].FeeModel = ConstantFeeModel(0)
            

            
        for symbol, symbolData in self.Data.items():
            consolidator = TradeBarConsolidator(BarPeriod)
            consolidator.DataConsolidated += self.OnDataConsolidated  # permette di fissare quanto deve essere temporalmente esteso un periodo della rolling window (nel nostro caso 1 giorno) )
            self.SubscriptionManager.AddConsolidator(symbolData.Symbol, consolidator)   # permette l'aggiornamento continuo e automatico dei dati della rolling window
        
        
        self.rowsX = len(self.EquitySymbols)  # numero di azioni dopo la pre-selezione
        self.columnsX = RollingWindowSize  # lunghezza serie storica
        self.X = np.zeros((self.rowsX, self.columnsX), dtype=float)



    def OnDataConsolidated(self, sender, bar):
        self.Data[str(bar.Symbol.ID)].Bars.Add(bar)

    
    ### OnData è la funzione (obbligatoria) in cui si costruisce la strategia effettiva
    ## Tale metodo è richiamato ogni giorno per identificare nuove opportunità di trading
    def OnData(self, data):
        
        
        ########## INIZIO PRIMA FASE: AGGREGAZIONE DATI
        ## -Costruzione matrice di prezzi X sfuttando il dizionario di oggetti SymbolData predisposto in Initialize
        ## -Conversione di X in matrice di rendimenti normalizzata
        

        ## Costruzione matrice di prezzi X
        self.X = np.zeros((self.rowsX, self.columnsX), dtype=float)   # dichiarazione matrice X
        i = -1
        for symbol in self.Data.keys():
            i += 1
            if self.Data[symbol].IsReady() and self.Data[symbol].WasJustUpdated(self.Time):
                for j in range(0, self.columnsX):
                    self.X[i, j] = self.Data[symbol].Bars[self.columnsX - j - 1].Close   # perchè la rolling window di QuantConnect è al contrario
                self.Data[symbol].PriceSeries = np.copy(self.X[i, :])
                if i == self.rowsX - 1:
                    self.Xcompleted = True   # se si arriva a questo punto, la matrice è stata completamente aggiornata 
        

        if self.Xcompleted and la.norm(self.X, 2)>10:   # se la matrice è aggiornata e non zero

            self.Xcompleted = False
            myCluster = Cluster()   # creazione istanza Cluster()
            ClusterDictionary = {}   # dizionario : la gestione degli elementi di ogni cluster
            for k in range(0, self.kclusters):  # costruiamo da subito un dizionario contenente le istanze Cluster()
                ClusterDictionary[k] = cp.copy(myCluster)
            self.myTradableGroup = TradableGroup()
            
            
            ## per non appesantire eccessivamente il backtest si effettua il ricomputo complessivo dei sottogruppi ogni 30 giorni
            self.RebuildSubGroups += 1
            if self.FirstTime or self.RebuildSubGroups %30 == 0:
                self.PositionManagement.ReadySubGroups={}
                self.FirstTime=False
            

                # modifico X in matrice di rendimenti
                for j in range(0, self.columnsX - 1):
                    self.X[:, j] = np.divide(self.X[:, j + 1] - self.X[:, j], self.X[:, j]);
    
                self.X = self.X[:, 0:self.columnsX - 1]  # non ci sono rendimenti giornalieri per l'ultimo giorno
                columnsX = self.columnsX
                columnsX -= 1  # nuova dimensione di X sulle colonne
                self.X = (self.X - self.X.mean(1)[:, None]) / (self.X.std(1)[:, None])  # normalizziamo ogni serie riconducendola ad una distribuzione N(0,1)
    
                
                ii = -1
                for symbol in self.Data.keys():
                    ii += 1
                    self.Data[symbol].ReturnSeries = np.copy(self.X[ii, :])
                
                ########## FINE PRIMA FASE
    
    
    
                ########## INIZIO SECONDA FASE: CLUSTERING E GESTIONE CLUSTERS
                ## -Applicazione kmeans sulla matrice X: si ottengono i clusters
                ## -Per ogni cluster ottenuto si costruisce una matrice X1 similmente a come si è fatto per X (ma solo per i titoli nel cluster)
                ## -Si esegue kmeans su X1 per ottenere sottogruppi molto piccoli (3-4 elementi), su cui nella terza fase si conduranno i test
                
                
                
                ## Applicazione kmeans sulla matrice X: si ottengono i clusters
                kmeans = KMeans(n_clusters=self.kclusters, init='k-means++')
                kmeans = kmeans.fit(self.X)  # applicazione kmeans a X
    
    
                for k in range(0, self.kclusters):
                    labelsymb = np.where(kmeans.labels_ == k)[0]  # raccogliamo insieme i simboli che appartengono al medesimo cluster
                    myCluster.SymbolElements = [self.EquitySymbols[pos] for pos in labelsymb]  # attribuiamo i simboli degli elementi separatamente ad ogni cluster indicizzato k
                    numberOfElements = myCluster.NumberOfElements()
                    
                    
                    ## Per ogni cluster ottenuto si costruisce una matrice X1 similmente a come si è fatto per X (ma solo per i titoli nel cluster)
                    X1 = np.zeros((numberOfElements, columnsX), dtype=float)
                    X1 = np.array([self.Data[symb].ReturnSeries for symb in myCluster.SymbolElements])
    
                    num = int(np.ceil(numberOfElements / 3))  # costruiamo sottogruppi con mediamente 3-4 elementi: per cui, in maniera più immediata, si devono effettuare pochi test per sottogruppo (se vengono soddisfatti si considerano candidati validi, altrimenti si controllano gli elementi del gruppo successivo)
                    kmeans1 = KMeans(n_clusters=num, init='k-means++').fit(X1)
    
                    for kk in range(0, num):  # fase di scelta dei gruppi su cui operare effettuando gli opportuni test
                        posSubGroupElem = np.where(kmeans1.labels_ == kk)[0]  # indice di ogni elemento raccolto rispetto alla disposizione in X1
                        numSubGroup = np.count_nonzero(posSubGroupElem)  # numero elementi per ogni sottogruppo individuato
                        if numSubGroup < 2:  # non abbiamo abbastanza elementi per continuare con l'analisi
                            continue
                        else:  # siamo nel caso in cui il cluster abbia almeno due elementi (sufficienti per tentare di trovare una coppia)
                            symbSubGroup = [myCluster.SymbolElements[pos] for pos in posSubGroupElem]  # simboli del sottogruppo
                            returnSeries = np.array([self.Data[symb].ReturnSeries for symb in symbSubGroup])  # raccogliamo le serie dei rendimenti relativi ai soli simboli del sottogruppo
                            symbCentroid = ClosestToCentroid(kmeans1.cluster_centers_[kk], returnSeries,symbSubGroup)  # determiniamo il 'centroide del sottogruppo' come l'elemento più vicino al centroide: sarà il punto di partenza per i test successivi
                            
                            ########## FINE SECONDA FASE
                            
                            
                            ########## INIZIO TERZA FASE: TEST E RACCOLTA SOTTOGRUPPI
                            ## -Per ogni sottogruppo individuato da kmeans1 (applicato su X1) si sceglie l'elemento più vicino al centroide come riferimento
                            ## -si effettuano i test dickey-fuller tra tale elemento ed un altro del sottogruppo: se esibiscono cointegrazione,
                            ##  vengono raccolti in un'istanza TradeableGroup e successivamente memorizzati in un dizionario; altrimenti si passa 
                            ##  a verificare la cointegrazione tra il primo elemento ed un altro facente parte del sottogruppo (finchè ne rimangono)
                            
                            for symb in symbSubGroup:  # cerchiamo relazioni di cointegrazione dentro un sottogruppo
                                if symb != symbCentroid:
                                    symbPriceSeries = self.Data[symb].PriceSeries
                                    centroidPriceSeries = self.Data[symbCentroid].PriceSeries
                                    df1 = dickeyFuller(centroidPriceSeries, symbPriceSeries, trend='c', method='aeg',maxlag=None, autolag='BIC',return_results=None)  # stimiamo i residui e svolgiamo il test Dickey Fuller: regredisco symbCentroid su symb
                                    df2 = dickeyFuller(symbPriceSeries, centroidPriceSeries, trend='c', method='aeg',maxlag=None, autolag='BIC',return_results=None)  # stimiamo i residui e svolgiamo il test Dickey Fuller: regredisco symb su symbCentroid
                                    if df1[0] < df1[2][1] and df2[0] < df2[2][1]:  # confronto del risultato della statistica test contro il relativo valore critico: imponiamo che entrambe le regressioni effettuate debbano superare il test (alpha=0.05%)
                                        if df1[0] < df2[0]:  # significa che la statistica test di df1 esibisce maggiore cointegrazione di df2
                                            XX = np.column_stack([symbPriceSeries])  # scriviamo in colonna la serie dei prezzi di symb
                                            XX = sm.add_constant(XX)  # inseriamo anche un termine costante
                                            model = sm.OLS(np.column_stack([centroidPriceSeries]), XX)  # regrediamo la serie di symbCentroid su quella di symb
                                            results = model.fit()
                                            beta1 = results.params[0]  # beta1 è il coefficiente del termine costante
                                            beta2 = results.params[1]  # beta2 è il coefficiente relativo a symb
                                            beta = np.array([1,-beta2])  # vettore beta dei coefficienti: contribuisce ad identificare l'attrattore
                                            
                                            
                                            self.myTradableGroup.Res = centroidPriceSeries - beta2 * symbPriceSeries
                                            self.myTradableGroup.Mu = beta1
                                            self.myTradableGroup.Sigma = np.std(self.myTradableGroup.Res)
                                            self.myTradableGroup.SymbolTradableElements = [symbCentroid, symb]
                                            self.myTradableGroup.Coeff = beta
                                            self.TradableGroupDictionary[kk] = self.myTradableGroup  # attribuiamo l'istanza TradableGroup()
    
                                            self.myTradableGroup = TradableGroup()  # costruiamo una nuova istanza pronta ad essere utlizzata al turno successivo
    
                                            break
                                        else:  # significa che la statistica test di df2 esibisce maggiore cointegrazione di df1
                                            XX = np.column_stack([centroidPriceSeries])  # scriviamo in colonna la serie dei prezzi di symbCentroid
                                            XX = sm.add_constant(XX)  # inseriamo anche un termine costante
                                            model = sm.OLS(np.column_stack([symbPriceSeries]), XX)  # regrediamo la serie di symb su quella di symbCentroid
                                            results = model.fit()
                                            beta1 = results.params[0]  # beta1 è il coefficiente del termine costante
                                            beta2 = results.params[1]  # beta2 è il coefficiente relativo a symbCentroid
                                            beta = np.array([1,-beta2])  # vettore beta dei coefficienti: contribuisce ad identificare l'attrattore
    
    
                                            self.myTradableGroup.Res = symbPriceSeries - beta2 * centroidPriceSeries
                                            self.myTradableGroup.Mu = beta1
                                            self.myTradableGroup.Sigma = np.std(self.myTradableGroup.Res)
                                            self.myTradableGroup.SymbolTradableElements = [symb, symbCentroid]
                                            self.myTradableGroup.Coeff = beta
                                            self.TradableGroupDictionary[kk] = cp.copy(self.myTradableGroup)  # attribuiamo l'istanza TradableGroup()
    
                                            self.myTradableGroup = TradableGroup()  # costruiamo una nuova istanza pronta ad essere utliizzata al turno successivo
    
                                            break
                                    else:
                                        continue  # non considereremo questa coppia; andiamo ad analizzare altre possibili relazioni di cointegrazione
    
                    myCluster.SubGroups = cp.copy(self.TradableGroupDictionary)  # attribuiamo il dizionario con i sottogruppi selezionati all'attributo SubGroups dell'istanza myCluster
                    ClusterDictionary[k] = cp.copy(myCluster)  # ogni key (denotata dal label del cluster) punta ad un value costituito da un'istanza Cluster()
                    myCluster = Cluster()  # ridefiniamo il cluster vuoto utile per l'analisi all'interno del prossimo cluster
                    self.TradableGroupDictionary = {}  # ridefiniamo il dizionario vuoto utile per l'analisi all'interno del prossimo cluster
    
                    ########## FINE TERZA FASE
    
    
    
    
                ########## INIZIO QUARTA FASE: SCORE/RANKING DEI SOTTOGRUPPI
                ## -costruiamo un dizionario ordinato in base allo score di oggetti TradableGroup()
    
                
                dictionarySubGroups = {}
                dictionarySorted = {}
                i = -1
                for k in range(0, self.kclusters):
                    subGroups = ClusterDictionary[k].SubGroups
                    len_subGroups = len(ClusterDictionary[k].SubGroups)
                    for j in range(0, len_subGroups):
                        if ClusterDictionary[k].SubGroups.get(j) == None:  # non abbiamo un sottogruppo per tale indice j
                            continue
                        else:
                            i += 1
                            ClusterDictionary[k].SubGroups[j].ZeroCrossing(2 * (subGroups[j].Sigma))  # determiniamo lo score in base a ZeroCrossing
                            dictionarySubGroups[i] = cp.copy(subGroups[j])
                            dictionarySorted[i] = cp.copy(ClusterDictionary[k].SubGroups[j].Score)
    
                dictionarySorted = OrderedDict(sorted(dictionarySorted.items(), key=lambda x: x[1],reverse=True))  # dizionario ordinato in base allo score
                for key in dictionarySorted.keys():  # scorriamo le keys e costruiamo un dizionario il cui value sia un'istanza TradableGroup con attributo score ordinato
                    if dictionarySorted[key]>0:   # selezioniamo in ordine solo i sottogruppi con score > 0 (e non == 0)
                        self.PositionManagement.ReadySubGroups[key] = cp.copy(dictionarySubGroups[key])  # Sottogruppi pronti (ordinati per score): sono in attesa del segnale
                    else:
                        break   # in quanto tutti gli altri sottogruppi hanno score == 0 e quindi il segnale risulterebbe poco attendibile
                    
            ########## FINE QUARTA FASE


            # liquidazione immediata di eventuali titoli che per le più svariate ragioni sono appena stati aggiunti impropriamente al portafoglio
            invested = [str(x.Symbol.ID) for x in self.Portfolio.Values if x.Invested]
            mysymb=[]
            toLiquidate=[]
            for openPos in list(self.PositionManagement.OpenPositions.values()):
                for symb in openPos.SymbolTradableElements:
                    mysymb.append(symb)
            toLiquidate=list(set(invested) - set(mysymb))
            if len(toLiquidate)!=0:
                for symb in toLiquidate:
                    self.Liquidate(self.Data[symb].Symbol.Value)
                self.Log(toLiquidate)


            ########## INIZIO QUINTA FASE: SEGNALI E GESTIONE POSIZIONI
            ## -si verifica la presenza di segnali di exit; se presenti su qualche sottogruppo, si procede alla liquidazione
            ## -si verifica la presenza di segnali di entry; se presenti su qualche sottogruppo, si procede aggiungendoli al portafoglio se non si è superato il numero massimo di posizioni disponibili



            # si liquidano le posizioni su cui si è innescato un segnale di exit
            if len(self.PositionManagement.OpenPositions) >= 1:  # bisogna verificare che ci sia almeno un sottogruppo in portafoglio, altrimenti non si sono sottogruppi da liquidare
                for key in list(self.PositionManagement.OpenPositions.keys()):
                    nowVal=self.PositionManagement.OpenPositions.get(key)
                    if nowVal!=None:
                        if nowVal.ExitSignal(data):  # boolean: True se sussiste segnale di exit, False altrimenti
                            for symb in nowVal.SymbolTradableElements:  # iteriamo sui symb con lo scopo di liquidare l'intero sottogruppo
                                
                                self.Liquidate(self.Data[symb].Symbol.Value)  # liquido la posizione relativa a symb
                            del self.PositionManagement.OpenPositions[key]  # elimino dal dizionario la posizione aperta relativa a key

            # dopo aver verificato che il portafoglio non è pieno, immettiamo i nuovi ordini in seguito ad un eventuale segnale di entry
            if len(self.PositionManagement.OpenPositions) != self.maxPositions:  # deteniamo al massimo maxPositions=3 sottogruppi in portafoglio: quindi se in un determinato momento ne abbiamo 3, allora il portafoglio è pieno e non facciamo nulla
                if len(self.PositionManagement.ReadySubGroups) >= 1:
                    valSet = self.PositionManagement.ReadySubGroups.values()
                    for val in list(valSet):
                        if len(self.PositionManagement.OpenPositions) == self.maxPositions:  # abbiamo già riempito tutte le 3 posizioni prestabilite
                            break
                        else:  # abbiamo ancora delle posizioni da riempire: controlliamo la presenza di segnali di entrata
                            if (val.EntrySignal(data))[0]:  # se è scattato un segnale di Entry
                                length = 0
                                for openValue in self.PositionManagement.OpenPositions.values():   # evitiamo di avere contemporaneamente diverse posizioni aperte sullo stesso titolo
                                    length += len(np.intersect1d(val.SymbolTradableElements, openValue.SymbolTradableElements))
                                if length == 0:  # controlliamo se non è già aperta una posizione con le azioni del sottogruppo che manifesta il segnale di Entry
                                    CashNext = self.CashStart / self.maxPositions  # denaro da investire nel prossimo sottogruppo
                                    self.PositionManagement.OpenPositions[self.counter] = cp.copy(val)  # assegniamo l'istanza TradeableGroup alle posizioni aperte, in modo da poter essere monitorata
                                    self.counter += 1

                                    if (val.EntrySignal(data))[1] == "Long":  # se il segnale è long, allora compriamo la combinazione lineare espressa da self.Coeff
                                        coeff = val.Coeff
                                    else:  # vendiamo la combinazione lineare
                                        coeff = -1 * val.Coeff
                                    sumcoeff = np.sum(np.abs(coeff))
                                    ccc = -1
                                    for symb in val.SymbolTradableElements:  # iteriamo sui symb con lo scopo aggiungere un intero sottogruppo
                                        ccc += 1
                                        symbcoeff = coeff[ccc]  # il coefficiente relativo alla serie di symb rappresenta la posizione
                                        cashToSymb = coeff[ccc] * CashNext / sumcoeff  # entità della posizione in denaro con segno
                                        quantityPerSymb = int(np.round(cashToSymb / self.Securities[symb].Price))  # numero di azioni di symb da comprare/vendere short
                                        self.MarketOrder(self.Data[symb].Symbol, quantityPerSymb)
                                    ccc = -1
                                    
                            else:  # se il segnale di Entry non è scattato
                                continue  # passiamo al prossimo sottogruppo in lista
                             
                             
                            ########## FINE QUINTA FASE  
                                
            
            
            
            
# si stabilisce qual è l'elemento più vicino al centroide (di cui si conosce la serie di rendimenti normalizzata)
# tra simboli (in una lista) appartenenti al medesimo cluster
def ClosestToCentroid(centroid, returnSeries, symbols):
    closestSymbol = symbols[0]
    closestSeries = returnSeries[0]
    distanceFromFirstSymbol = la.norm(closestSeries - centroid, 2)
    for k in range(0, len(symbols)):
        if la.norm(returnSeries[k] - centroid, 2) < distanceFromFirstSymbol:
            closestSymbol = symbols[k]
    return closestSymbol


class SymbolData(object):

    def __init__(self, symbol, barPeriod, windowSize):
        self.Symbol = symbol   # simbolo del titolo
        self.BarPeriod = barPeriod   # Frequenza della rolling window : 1 giorno
        self.Bars = RollingWindow[TradeBar](windowSize)   # oggetto RollingWindow (TradeBar è un tipo di QuantConnect)
        self.PriceSeries = None   # serie dei prezzi
        self.ReturnSeries = None   # serie dei rendimenti normalizzati


    def IsReady(self):   # verifica se l'oggetto Bars (e quindi la rolling window) sia stato aggiornato: restituisce un boolean
        return self.Bars.IsReady   # IsReady è attributo interno di Bars che verifica l'aggiornamento dell'oggetto

    def WasJustUpdated(self, current):   # verifica che l'aggiornamento sia appena avvenuto
        return self.Bars.Count > 0 and self.Bars[0].Time == current - self.BarPeriod


class Cluster(object):

    def __init__(self):
        self.SymbolElements = None   # simboli dei titoli che appartengono ad un medesimo cluster
        self.SubGroups = {}   # dizionario di sottogruppi definiti dalla classe TradableGroup

    def NumberOfElements(self):
        return len(self.SymbolElements)



# gestione sottogruppi di azioni (coppie se Pairs-Trading) su cui si può effettivamente operare analizzando le serie delle opportune combinazioni lineari
class TradableGroup(object):

    def __init__(self):
        self.SymbolTradableElements = None  # titoli su cui si può effettivamente fare trading: basta attendere il segnale giusto
        self.Coeff = None  # coefficienti della combinazione lineare generata
        self.Mu = None  # media dei residui
        self.Sigma = None  # deviazione standard dei residui
        self.Res = None  # serie stimata dei residui (== combinazione lineare generata)
        self.Score = None  # score determinato da ZeroCrossing(self, limit)
        self.Direction = None  # direzione del segnale sulla combinazione lineare ("Long" o "Short" se è presente un segnale)


    def NumberOfElements(self):
        return len(self.SymbolTradableElements)

    # confronta prezzo attuale per determinare un eventuale segnale di entry : restituisce un boolean
    def EntrySignal(self, data):  
        signal = False
        Res = self.Res - self.Mu  # viene tolta la media: i residui hanno ora media 0
        totsum = 0
        self.Direction = "Short"  # direzione della posizione da assumere sulla combinazione di titoli
        for k in range(0, len(self.Coeff)):
            totsum += self.Coeff[k] * data[self.SymbolTradableElements[k]].Price
        currentPrice = totsum - self.Mu  # valore attuale della combinazione lineare
        if (np.abs(currentPrice) > 2 * self.Sigma) and (np.abs(currentPrice) < 2.25 * self.Sigma):  # segnale sulla combinazione
            signal = True
            if currentPrice < 0:  # segnale long sulla combinazione
                self.Direction = "Long"
        direction = self.Direction

        return signal, direction


    # confronta prezzo attuale per determinare un eventuale segnale di exit : restituisce un boolean
    def ExitSignal(self, data):
        takeProfit = 0.5  # livello rispetto alla media 0 per cui si esce dalla posizione aperta (è espresso come coefficiente di Sigma)
        signal = False
        Res = self.Res - self.Mu  # viene tolta la media: i residui hanno ora media 0
        totsum = 0
        for k in range(0, len(self.Coeff)):
            totsum += self.Coeff[k] * data[self.SymbolTradableElements[k]].Price
        currentPrice = totsum - self.Mu  # valore attuale della combinazione lineare
        
        if np.abs(currentPrice) > 2.5 * self.Sigma:  # segnale di exit sulla combinazione
            signal = True
        
        if self.Direction == "Long":  # la posizione (ancora aperta) è long sulla combinazione di titoli
            if currentPrice > (-1) * takeProfit * self.Sigma:
                signal = True
                
        if self.Direction == "Short":  # la posizione (ancora aperta) è short sulla combinazione di titoli
            if currentPrice < takeProfit * self.Sigma:
                signal = True
        
        return signal

    def ZeroCrossing(self, limit):
        # limit e' il livello, rispetto a 0, dopo il quale si innesca il segnale di trading nella combinazione lineare (2*sigma)
        beta1 = self.Mu
        x = self.Res - beta1 * np.ones(len(self.Res))  # centriamo i residui attorno alla media 0
        nzeros = 0  # nzeros==1 significa che la combinazione ritorna a 0 una sola volta
        posSignals = np.where(np.abs(x) > limit)  # identifico le posizioni dei segnali
        posSignals = posSignals[0]  # è una tupla di un elemento
        k = 0
        len_x = len(x)
        len_posSignals = len(posSignals)
        timesteps = 0
        if len_posSignals > 1:  # altrimenti c'è un numero insufficiente di segnali
            while k < len_posSignals - 1:

                if (posSignals[k + 1] - posSignals[k]) == 1:
                    if (x[posSignals[k]] > 0 and x[posSignals[k + 1]] < 0) or (x[posSignals[k]] < 0 and x[posSignals[k + 1]] > 0):
                        nzeros += 1
                        timesteps += 1

                else:
                    for j in range(posSignals[k] + 1, posSignals[k + 1]):

                        if (x[posSignals[k]] > 0 and x[j] < 0) or (x[posSignals[k]] < 0 and x[j] > 0):
                            nzeros += 1
                            timesteps += j - posSignals[k]
                            break
                        if (j == posSignals[k + 1] - 1):
                            timesteps += posSignals[k + 1] - posSignals[k]
                            if (x[posSignals[k]] > 0 and x[posSignals[k + 1]] < 0) or (x[posSignals[k]] < 0 and x[posSignals[k + 1]] > 0):
                                nzeros += 1
                k += 1
            if posSignals[len_posSignals - 1] != (len_x - 1):
                for j in range(posSignals[len_posSignals - 1] + 1, len_x):
                    if (x[posSignals[len_posSignals - 1]] > 0 and x[j] < 0) or (x[posSignals[len_posSignals - 1]] < 0 and x[j] > 0):
                        nzeros += 1
                        timesteps += j - posSignals[k]
                        break
            if nzeros > 0:
                AvgTime = timesteps / nzeros  # stima tempo medio per mean-reversion
                self.Score = 1 / AvgTime
            else:
                self.Score = 0
        else:
            self.Score = 0  # se i segnali sono pochi, mettiamo il sottogruppo in fondo alla classifica
        return


class PositionManagement(object):

    def __init__(self):
        self.ReadySubGroups = {}  # dizionario di sottogruppi ammissibili ma su cui si è in attesa del segnale
        self.OpenPositions = {}  # dizionario di sottogruppi su cui si sono aperte delle posizioni che attendono di essere chiuse al momento opportuno
