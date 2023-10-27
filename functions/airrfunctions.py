import scanpy as sc
from glob import glob
import pandas as pd
import numpy as np
import seaborn as sns
import anndata
import scipy
import re
import os
import matplotlib
import math
import random
import itertools

import populationfunctions as pf
import generalfunctions as gf

from statannot import add_stat_annotation
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.legend import Legend
import matplotlib.gridspec as gridspec
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 200)
pd.options.display.max_seq_items = 2000

sc.set_figure_params(scanpy=True, dpi=300, dpi_save=300, frameon=True, vector_friendly=True, fontsize=12, 
                         color_map='Dark2', format='pdf', transparent=True, ipython_format='png2x')

rcParams.update({'font.size': 8})
rcParams.update({'font.family': 'Helvetica'})
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['svg.fonttype'] = 'none'
rcParams['figure.facecolor'] = (1,1,1,1)

import warnings
warnings.filterwarnings("ignore")

## percent expansion
def exp_pct(adata,new=False,normalization='cells',receptor_column=None):
    
    exp_dict = {'Base':'E1',
               'PD1':'NE1_E2',
               'RTPD1':'NE1_NE2_E3'}
    
    if new:
        tx = adata.obs.treatment.unique().tolist()[0]
        exp = adata[(adata.obs[exp_dict[tx]]=='Y')]    
    else:
        exp = adata[(adata.obs.medianExpansion=='expanded')]
    
    if (normalization=='cells'):
        totalsize = len(adata)
        expsize = len(exp)
    elif (normalization=='clonotype'):
        totalsize = len(adata.obs[receptor_column].unique())
        expsize = len(exp.obs[receptor_column].unique())
        
    return (expsize/totalsize)*100
    
## calculate Gini coefficient
## from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
def gini(adata,receptor_column=None):

    tcrs = adata.obs[receptor_column].value_counts()
    tcrs = tcrs[tcrs>0].to_numpy()

    tcrs = tcrs.flatten() #all values are treated equally, tcrs must be 1d 
    tcrs = np.sort(tcrs) #values must be sorted
    index = np.arange(1,tcrs.shape[0]+1) #index per tcrs element
    n = tcrs.shape[0] #number of tcrs elements
    
    return ((np.sum((2 * index - n  - 1) * tcrs)) / (n * np.sum(tcrs))) #Gini coefficient

## calculate normalized shannon entropy clonality
## clonality = 1 - (SE/ln(# of tcrs))
def shannon(adata,receptor_column=None):

    totalsize = len(adata)

    tcrs = adata.obs[receptor_column].value_counts()
    tcrs = tcrs/totalsize
    totaltcrs = len(tcrs)
    
    runningSum = 0
    
    for i in tcrs:
        runningSum += -1*(i * np.log(i))

    return (1 - (runningSum/np.log(totaltcrs)))

## Jensen-Shannon diversity
def JSD(adata,first,second,category,topn=None,receptor_column=None):
    
    if (topn is not None):
        firstCount = adata[(adata.obs[category]==first)].obs[receptor_column].value_counts().sort_values(ascending=False)
        secondCount = adata[(adata.obs[category]==second)].obs[receptor_column].value_counts().sort_values(ascending=False)
        
        num = int(len(firstCount)*(topn/100))
        tcrs = firstCount.index[:num].tolist()
        
        num = int(len(secondCount)*(topn/100))
        tcrs = tcrs + secondCount.index[:num].tolist()
        
        tcrs = np.unique(tcrs)
        
    else:
        tcrs = adata[(adata.obs[category]==first)|(adata.obs[category]==second)].obs[receptor_column].unique().tolist()
    
    firstCount = adata[(adata.obs[category]==first)].obs[receptor_column].value_counts()
    secondCount = adata[(adata.obs[category]==second)].obs[receptor_column].value_counts()
    
    ## convert index from CategoricalIndex to list, so that we can add to it
    firstCount.index = firstCount.index.tolist()
    secondCount.index = secondCount.index.tolist()
    
    if (len(firstCount)==0)|(len(secondCount)==0):
        return np.NaN
    
    ms = [x for x in tcrs if x not in firstCount.index]
    for m in ms:
        firstCount.loc[m] = 0
        
    ms = [x for x in tcrs if x not in secondCount.index]
    for m in ms:
        secondCount.loc[m] = 0
        
    firstCountFiltered = firstCount.loc[tcrs]
    secondCountFiltered = secondCount.loc[tcrs]
    
#     firstCountFiltered = firstCountFiltered/firstCount.sum()
#     secondCountFiltered = secondCountFiltered/secondCount.sum()
    
    firstCountFiltered = firstCountFiltered/firstCountFiltered.sum()
    secondCountFiltered = secondCountFiltered/secondCountFiltered.sum()
    
    KL_P = 0
    KL_Q = 0
    
    for t in tcrs:
        p = firstCountFiltered.loc[t]
        q = secondCountFiltered.loc[t]
        if (p>0):
            KL_P = KL_P + p*np.log2(2*(p/(p+q)))
        if (q>0):
            KL_Q = KL_Q + q*np.log2(2*(q/(p+q)))
    
    return 0.5*(KL_P + KL_Q)


def clonality_df(adata,
                 groupby= None,
                 rep= None,
                 xcat= None,
                 hcat= None,
                 metrics= None,
                 drop_na= True,
                 receptor_column= None):

    if receptor_column is None:
        print("Must specify receptor_column")
        return None
    
    if metrics is None:
        metrics = ['percent_cells','percent_clonotype','shannon','gini']

    pt = adata.obs[rep].unique().tolist()
    tx = adata.obs[xcat].unique().tolist()

    if groupby is None:
        idx = pd.MultiIndex.from_product([pt,tx],names=[rep,xcat])
    else:
        grps = adata.obs[groupby].unique().tolist()
        idx = pd.MultiIndex.from_product([pt,tx,grps],names=[rep,xcat,groupby])
        
    df = pd.DataFrame(index=idx,columns=metrics)

    for i in df.index:
        
        if groupby is None:
            total = adata[(adata.obs[idx.names[0]]==i[0])&(adata.obs[idx.names[1]]==i[1])]
        else:
            total = adata[(adata.obs[idx.names[0]]==i[0])&(adata.obs[idx.names[1]]==i[1])&(adata.obs[idx.names[2]]==i[2])]
            
        totalsize = len(total)
        
        drop_cols = []
        if (totalsize>0):
            for m in metrics:
                if (m=='percent_cells'):
                    df.loc[i,m] = exp_pct(total,new=False,normalization='cells',receptor_column=receptor_column)
                elif (m=='percent_clonotype'):
                    df.loc[i,m] = exp_pct(total,new=False,normalization='clonotype',receptor_column=receptor_column)
                elif (m=='shannon'):
                    df.loc[i,m] = shannon(total,receptor_column=receptor_column)
                elif (m=='gini'):
                    df.loc[i,m] = gini(total,receptor_column=receptor_column)
                else:
                    print(f"{m} not recognized, will be skipped.")
                    drop_cols.append(m)

        df.drop(columns= drop_cols, inplace= True)

    df.reset_index(inplace=True)

    metadata = adata.obs.groupby(by=[rep]).first()[hcat]
    df = df.merge(metadata, how='left', left_on= rep, right_on= rep)

    if drop_na:
        df.dropna(how='any',inplace=True)

    return df


def clonality_boxplot(adata,
                      groupby= None,
                      rep= None,
                      xcat= None,
                      hcat= None,
                      xorder= None,
                      horder= None,
                      metrics= None,
                      receptor_column= None,
                      show_stats= True,
                      df= None,
                      drop_na= True,
                      dot_size= 1,
                      fontsize= 4,
                      ylim= None,
                      figsize= None):
    
    if receptor_column is None:
        print("Must specify receptor_column")
        return None

    if metrics is None:
        metrics = ['percent_cells','percent_clonotype','shannon','gini']

    if df is None:
        df = clonality_df(adata,
                          groupby= groupby,
                          rep= rep,
                          xcat= xcat,
                          hcat= hcat,
                          metrics= metrics,
                          drop_na= drop_na,
                          receptor_column= receptor_column)
    
    sns.set_style("white", rc={"font.family":"Helvetica","axes.grid":False})
    sns.set_context("paper", rc={"font.size":fontsize,"axes.titlesize":fontsize,"axes.labelsize":fontsize,"font.family":"Helvetica","xtick.labelsize":fontsize,"ytick.labelsize":fontsize})
    
    if groupby is None:
    
        nr = 1
        nc = len(metrics)
        
        if figsize is None:
            fig,axs = plt.subplots(nrows=nr,ncols=nc,sharex=False,sharey=False,figsize=(nc,nr))
        else:
            fig,axs = plt.subplots(nrows=nr,ncols=nc,sharex=False,sharey=False,figsize=figsize)
        
        combs = list(itertools.combinations(horder,2))

        for m,ax in zip(metrics,np.ravel(axs)):

            _= sns.boxplot(x= xcat,
                           y= m,
                           hue= hcat,
                           data= df,
                           color= 'w',
                           linewidth= 0.5,
                           fliersize= 0,
                           palette= 'colorblind',
                           order= xorder,
                           hue_order= horder,
                           boxprops= dict(alpha=0.2),
                           ax= ax)
            
            if show_stats:
                try:
                    test_results = add_stat_annotation(ax,x=xcat,y=m,hue=hcat,data=df,order=xorder,hue_order=horder,
                                                        box_pairs = [tuple((x,y) for y in c) for c in combs for x in xorder],
                                                        test='Mann-Whitney',comparisons_correction=None,text_format='simple',loc='inside',verbose=0,
                                                        fontsize=4,linewidth=0.5,line_height=0.01,text_offset=0.01)
                except:
                    pass

            _= sns.stripplot(x= xcat,
                             y= m,
                             hue= hcat,
                             dodge= True,
                             jitter= 0.1,
                             data= df,
                             size= dot_size,
                             order= xorder,
                             hue_order= horder,
                             ax= ax)
            
            _= ax.set(xlabel='', title=m)
            _= ax.legend(fontsize=2, title_fontsize=2, markerscale=0.05)
            
            if (ylim is not None):
                _= ax.set_ylim(ylim)
            elif ('percent' in m):
                _= ax.set_ylim(-2,102)
            else:
                _= ax.set_ylim(-0.1,1.1)
                
    else:
            
        st = df[groupby].unique().tolist()
        
        nr = len(st)
        nc = len(metrics)
        
        forder = [(x,y) for x in st for y in metrics]
        
        if figsize is None:
            fig,axs = plt.subplots(nrows=nr,ncols=nc,sharex=False,sharey=False,figsize=(nc,nr))
        else:
            fig,axs = plt.subplots(nrows=nr,ncols=nc,sharex=False,sharey=False,figsize=figsize)

        combs = list(itertools.combinations(horder,2))

        for f,ax in zip(forder,np.ravel(axs)):
            
            s = f[0]
            m = f[1]
            
            tmp = df[(df[groupby]==s)]
            
            _= sns.boxplot(x= xcat,
                           y= m,
                           hue= hcat,
                           data= tmp,
                           color= 'w',
                           linewidth= 0.5,
                           fliersize= 0,
                           palette= 'colorblind',
                           order= xorder,
                           hue_order= horder,
                           boxprops= dict(alpha=0.2),
                           ax= ax)
            
            if show_stats:
                try:
                    test_results = add_stat_annotation(ax,x=xcat,y=m,hue=hcat,data=tmp,order=xorder,hue_order=horder,
                                                        box_pairs = [tuple((x,y) for y in c) for c in combs for x in xorder],
                                                        test='Mann-Whitney',comparisons_correction=None,text_format='simple',loc='inside',verbose=2,
                                                        fontsize=4,linewidth=0.5,line_height=0.01,text_offset=0.01)
                except:
                    print("stat annotations failed.")

            _= sns.stripplot(x= xcat,
                             y= m,
                             hue= hcat,
                             dodge= True,
                             jitter= 0.1,
                             data= tmp,
                             size= dot_size,
                             order= xorder,
                             hue_order= horder,
                             ax= ax)
            
            _= ax.set(xlabel='', title=m+' '+s)
            _= ax.legend(fontsize=2, title_fontsize=2, markerscale=0.05)
    
            if (ylim is not None):
                _= ax.set_ylim(ylim)
            elif ('percent' in m):
                _= ax.set_ylim(-2,102)
            else:
                _= ax.set_ylim(-0.1,1.1)
            
    plt.tight_layout()

    return fig,axs


def clonality_lineplot(adata,
                       rep= None,
                       xcat= None,
                       hcat= None,
                       xorder= None,
                       horder= None,
                       metrics= None,
                       receptor_column= None,
                       df= None,
                       drop_na= True,
                       fontsize= 4,
                       ylim= None,
                       figsize= None):

    if receptor_column is None:
        print("Must specify receptor_column")
        return None

    if metrics is None:
        metrics = ['percent_cells','percent_clonotype','shannon','gini']

    if df is None:
        df = clonality_df(adata,
                          rep= rep,
                          xcat= xcat,
                          hcat= hcat,
                          metrics= metrics,
                          drop_na= drop_na,
                          receptor_column= receptor_column)
    
    ##lineplot
    forder = [(x,y) for x in horder for y in metrics]

    nr = len(horder)
    nc = len(metrics)
    
    sns.set_style("white", rc={"font.family":"Helvetica","axes.grid":False})
    sns.set_context("paper", rc={"font.size":fontsize,"axes.titlesize":fontsize,"axes.labelsize":fontsize,"font.family":"Helvetica","xtick.labelsize":fontsize,"ytick.labelsize":fontsize})
    
    if figsize is None:
        fig,axs = plt.subplots(nrows=nr,ncols=nc,sharex=False,sharey=False,figsize=(nc,nr))
    else:
        fig,axs = plt.subplots(nrows=nr,ncols=nc,sharex=False,sharey=False,figsize=figsize)
    
    for f,ax in zip(forder,np.ravel(axs)):
        
        h = f[0]
        m = f[1]
        
        tmp = df[(df[hcat]==h)]

        pt = tmp[rep].unique().tolist()

        for p in pt:
            if len(tmp[(tmp[rep]==p)][xcat].unique().tolist()) < 3:
                tmp = tmp[(tmp[rep]!=p)]

        ys1 = tmp[(tmp[xcat]==xorder[0])][m].to_numpy(dtype='float64')
        ys2 = tmp[(tmp[xcat]==xorder[1])][m].to_numpy(dtype='float64')
        ys3 = tmp[(tmp[xcat]==xorder[2])][m].to_numpy(dtype='float64')

        xs1 = np.full(shape=(len(ys1),1),fill_value=1)
        xs2 = np.full(shape=(len(ys2),1),fill_value=2)
        xs3 = np.full(shape=(len(ys3),1),fill_value=3)

        counter = 0
        for x1,y1,x2,y2 in zip(xs1,ys1,xs2,ys2):
    #         _= ax.plot((x1,x2),(y1,y2),marker='o',color=cs[counter],linewidth=0.5,markersize=1)

            if (y2-y1>0):
                _= ax.plot((x1,x2),(y1,y2),marker='o',color='k',linewidth=0.5,markersize=1)
            else:
                _= ax.plot((x1,x2),(y1,y2),marker='o',color='0.8',linewidth=0.5,markersize=1)
                
            counter += 1
            
        counter = 0
        for x1,y1,x2,y2 in zip(xs2,ys2,xs3,ys3):
    #         _= ax.plot((x1,x2),(y1,y2),marker='o',color=cs[counter],linewidth=0.5,markersize=1)
            
            if (y2-y1>0):
                _= ax.plot((x1,x2),(y1,y2),marker='o',color='k',linewidth=0.5,markersize=1)
            else:
                _= ax.plot((x1,x2),(y1,y2),marker='o',color='0.8',linewidth=0.5,markersize=1)
            
            counter += 1

        if (ylim is not None):
            _= ax.set_ylim(ylim)
        elif ('percent' in m):
            _= ax.set_ylim(-2,102)
        else:
            _= ax.set_ylim(-0.1,tmp[m].max()+0.1)

        _= ax.set(xticks=(1,2,3),xticklabels=xorder,ylabel=m,title=h+' '+m)
        
    plt.tight_layout()

    return fig,axs


## TCR/BCR sharing
def receptor_sharing(adata,
                     receptor_column= None,
                     rep= None,
                     xcat= None,
                     min_clonotypes= 5,
                     df= None,
                     fontsize= 2,
                     figsize= None,
                     vmin= None,
                     vmax= None,
                     cmap= None,
                     cbar= True,
                     cbar_kws= None):

    if df is None:

        pt = sorted(adata.obs[rep].unique().tolist())
        tx = sorted(adata.obs[xcat].unique().tolist())

        dfList = []
        for t in tx:
            for p1 in pt:
                
                total = adata[(adata.obs[rep]==p1)&(adata.obs[xcat]==t)]
                r = total.obs[receptor_column].unique().tolist()
                num_r = len(r)
                
                if (num_r < min_clonotypes):
                    continue
                    
                for p2 in pt:
                    
                    total2 = adata[(adata.obs[rep]==p2)&(adata.obs[xcat]==t)]
                    r2 = total2.obs[receptor_column].unique().tolist()
                    
                    if (len(r2) < min_clonotypes):
                        continue
                
                    match = [x for x in r if x in r2]
                    
                    match = (len(match)/num_r)*100
                    match = pd.DataFrame(match,index=['test'],columns=['percent'])
                    match['from'] = p1
                    match['to'] = p2
                    match['treatment'] = t
                    
                    dfList.append(match)
                    
            print(t)
        df = pd.concat(dfList)

    pt = sorted(adata.obs[rep].unique().tolist())
    tx = sorted(adata.obs[xcat].unique().tolist())

    if figsize is None:
        figsize = (4*len(tx), 4)
    if cmap is None:
        cmap = 'inferno'
    if cbar_kws is None:
        cbar_kws = {
                    'ticks': [vmin, vmax / 2, vmax],
                    'orientation': 'horizontal',
                    'shrink': 0.5,
                    'label': '% shared'
                   }
        
    fig,axs = plt.subplots(nrows= 1,
                           ncols= len(tx),
                           figsize= figsize)

    for t, ax in zip(tx, axs.flat):
        
        tmp = df[(df[xcat]==t)]
        
        tmp = tmp.pivot(index='from', columns='to', values='percent')
        tmp = tmp.loc[[x for x in pt if x in tmp.index], [x for x in pt if x in tmp.columns]]

        _= sns.heatmap(tmp, 
                       ax= ax, 
                       vmin= vmin, 
                       vmax= vmax,
                       cmap= cmap,
                       cbar= cbar,
                       cbar_kws= cbar_kws,
                       square= True)

        _= ax.set_title(t, fontsize= 8)
        _= ax.set_ylabel("from", fontsize= 8)
        _= ax.set_xlabel("to", fontsize= 8)
        _= ax.set_xticks(np.arange(tmp.shape[1]) + 0.5)
        _= ax.set_xticklabels(tmp.columns.tolist(), fontsize= fontsize)
        _= ax.set_yticks(np.arange(tmp.shape[0]) + 0.5)
        _= ax.set_yticklabels(tmp.index.tolist(), fontsize= fontsize)
        _= ax.tick_params(axis= 'both', length= 1, width= 0.5)
        _= ax.grid(visible= False, which= 'both')

    return fig, axs, df


## TCR/BCR transitions
def receptor_transition(adata=None,df=None,calc=True,groupby=None,tx=None,tx2=None,receptor_column=None,fontsize=2,figsize=None,vmax=None):

    """
    tx and tx2 should be lists of tuples containing the desired transitions, 
    where tx indicates the detection options and tx2 indicates the timepoints
    for example:
    tx = [('D1','D2'),('D2','D3'),('D1','D3')]
    tx2 = [('Base','PD1'),('PD1','RTPD1'),('Base','RTPD1')]

    """

    if calc:

        pt = adata.obs.cohort.unique().tolist()

        st = adata.obs[groupby].unique().tolist()

        trans = [(x,y) for x in st for y in st]

        idx = pd.MultiIndex.from_product([pt,trans,tx],names=['cohort','trans','tx'])
        df = pd.DataFrame(0.0,index=idx,columns=['frequency'])
        df.reset_index(inplace=True)

        for i,j in zip(tx,tx2):
            
            tcrs = adata[(adata.obs[i[0]]=='Y')&(adata.obs[i[1]]=='Y')].obs[receptor_column].unique().tolist()
            
            for t in tcrs:
                
                p = t.split('_')[0]
                
                cellFirst = adata[(adata.obs.treatment==j[0])&(adata.obs[receptor_column]==t)]
                cellSecond = adata[(adata.obs.treatment==j[1])&(adata.obs[receptor_column]==t)]
                
                numFirst = len(cellFirst)
                numSecond = len(cellSecond)
                
                grpFirst = cellFirst.obs[groupby].unique().tolist()
                grpSecond = cellSecond.obs[groupby].unique().tolist()
                
                grps = [x for x in trans if x[0] in grpFirst and x[1] in grpSecond]
                
                for g in grps:
                    
                    f1 = len(cellFirst[(cellFirst.obs[groupby]==g[0])])/numFirst
                    f2 = len(cellSecond[(cellSecond.obs[groupby]==g[1])])/numSecond
                    f = f1*f2
                    
                    current = df.loc[(df.cohort==p)&(df.trans==g)&(df.tx==i),'frequency']
                    
                    df.loc[(df.cohort==p)&(df.trans==g)&(df.tx==i),'frequency'] = current + f

            print(i)


    tx = df.tx.unique().tolist()

    norm = [df[(df.cohort==df.loc[x,'cohort'])].frequency.sum() for x in df.index]

    df['norm'] = norm

    df['freq_norm'] = df['frequency']/df['norm']

    df.dropna(how='any',inplace=True)

    metadata = pd.read_csv('../PEMBRORT_CLINICAL_METADATA_FORSCSEQ_KHG20210624.csv',index_col=None,header=0)
    cols = [x for x in metadata.columns if x not in df.columns]
    metadata = metadata[cols+['Patient_Number']]
    df = df.merge(metadata,left_on='cohort',right_on='Patient_Number')

    if figsize is None:
        figsize = (6,4)

    fig,axs = plt.subplots(nrows=2,ncols=3,figsize=figsize)

    R = ['R','NR']
    forder = [(x,y) for x in R for y in tx]

    for f,ax in zip(forder,axs.flat):
        
        tmp = df[(df.tx==f[1])&(df.pCR==f[0])]

        transmat = pd.DataFrame(index=st,columns=st)

        for t in trans:
            transmat.loc[t[0],t[1]] = np.median(tmp[(tmp.trans==t)]['freq_norm'])

        _= sns.heatmap(transmat.to_numpy(dtype=np.float64),ax=ax,vmax=vmax)
        _= ax.set_title(f,fontsize=fontsize)
        _= ax.set_xticks(np.arange(len(st))+0.5)
        _= ax.set_yticks(np.arange(len(st))+0.5)
        _= ax.set_xticklabels(st,fontsize=fontsize)
        _= ax.set_yticklabels(st,fontsize=fontsize)
        _= ax.set_aspect('equal')

    return fig,axs,df



def vdj_usage_heatmap(adata,df=None,calc_pct=True,comparator=None,reference=None,logic_comp='and',logic_ref='and',
                        groupby=None,rep='cohort',normalization='cells',receptor_column=None,drop_na=True,thresh=None,
                        paired_test=False,vmin=None,vmax=None,size_max=None,fontsize=4,figsize=(2,2)):
    
    tmp = adata.copy()
    
    tmp.obs['marker'] = pd.DataFrame(index=tmp.obs.index,columns=['marker'])
    xcat = 'marker'
    
    if (logic_comp=='and'):
        tmp.obs.loc[[all(tmp.obs.loc[x,y]==comparator[y] for y in comparator) for x in tmp.obs.index],'marker'] = 'comparator'
    elif (logic_comp=='or'):
        tmp.obs.loc[[any(tmp.obs.loc[x,y]==comparator[y] for y in comparator) for x in tmp.obs.index],'marker'] = 'comparator'

    if reference is None:
        tmp.obs.loc[(tmp.obs.marker!='comparator'),'marker'] = 'reference'
    elif (logic_ref=='and'):
        tmp.obs.loc[[all(tmp.obs.loc[x,y]==reference[y] for y in reference) for x in tmp.obs.index],'marker'] = 'reference'
    elif (logic_ref=='or'):
        tmp.obs.loc[[any(tmp.obs.loc[x,y]==reference[y] for y in reference) for x in tmp.obs.index],'marker'] = 'reference'
 
    if calc_pct:
        df = pf.pct_df_faster(tmp,groupby=groupby,rep=rep,xcat=xcat,drop_na=drop_na,thresh=thresh,normalization=normalization,receptor_column=receptor_column)
        print('calculated percent')
    else:
        metadata = pd.read_csv('../PEMBRORT_CLINICAL_METADATA_FORSCSEQ_KHG20210624.csv',index_col=None,header=0)
        cols = [x for x in metadata.columns if x not in df.columns] + ['Patient_Number']
        cols = np.unique(cols)
        df = df.merge(metadata[cols],left_on='cohort',right_on='Patient_Number')  
        if drop_na:
            df.dropna(how='any',inplace=True)
    
    st = df[groupby].unique().tolist()
    cols = df[xcat].unique().tolist()

  
    effect_size = pd.DataFrame(index=st,columns=['effect_size'],dtype=np.float64)
    pval = pd.DataFrame(index=st,columns=['pval'],dtype=np.float64)

    if paired_test:

        for s in st:

            y0 = df[(df[groupby]==s)&(df[xcat]=='comparator')].set_index(rep)
            y1 = df[(df[groupby]==s)&(df[xcat]=='reference')].set_index(rep)
                
            test = [y0.loc[x,'percent'] - y1.loc[x,'percent'] for x in y0.index if x in y1.index]
            
            if (len(test)>0)&(~all(x==0 for x in test)):

                S,P = scipy.stats.wilcoxon(test)

                effect_size.loc[s,'effect_size'] = np.median(test)
                pval.loc[s,'pval'] = P

    else:

        for s in st:

            y0 = df[(df[groupby]==s)&(df[xcat]=='comparator')]['percent'].to_numpy(dtype=np.float64)
            y1 = df[(df[groupby]==s)&(df[xcat]=='reference')]['percent'].to_numpy(dtype=np.float64)
            S,P = scipy.stats.ranksums(y0,y1)

            effect_size.loc[s,'effect_size'] = S
            pval.loc[s,'pval'] = P

    pval.dropna(how='any',inplace=True)
    effect_size.dropna(how='any',inplace=True)

    newst = pval.index

    pval = -1*np.log10(pval.to_numpy(dtype=np.float64))
    pval[np.isinf(pval)] = np.max(pval[~np.isinf(pval)])
    pval = pd.DataFrame(pval,index=newst,columns=['pval'])


    d = scipy.spatial.distance.pdist(effect_size.to_numpy(dtype='float64'),metric='euclidean')
    l = scipy.cluster.hierarchy.linkage(d,metric='euclidean',method='complete',optimal_ordering=True)
    dn = scipy.cluster.hierarchy.dendrogram(l,no_plot=True)
    order = dn['leaves']
    
    effect_size = effect_size.iloc[order,:]
    pval = pval.iloc[order,:]
    
    if size_max is None:
        size_max = pval.max().item()
    
    if vmin is None:
        vmin = effect_size.min().item()
    if vmax is None:
        vmax = effect_size.max().item()

    ref_s = [1.30,size_max]
    ref_l = ['0.05','maxsize: '+'{:.1e}'.format(10**(-1*size_max))]

    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=figsize)
    
    gf.heatmap2(effect_size,cmap='RdBu_r',vmin=vmin,vmax=vmax,cellsize=pval,square=True,cellsize_vmax=size_max,ref_sizes=ref_s,ref_labels=ref_l,fontsize=fontsize,figsize=figsize,ax=ax[0])

    icoord = np.array(dn['icoord'] )
    dcoord = np.array(dn['dcoord'] )

    for xs, ys in zip(dcoord, icoord):
        _= ax[1].plot(xs, ys, color='k', linewidth=0.5)

    ax[1].grid(b=False)
    ax[1].set(xticks=[],yticks=[])
    
    return fig, ax, df, effect_size, pval