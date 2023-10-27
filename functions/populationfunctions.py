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
from statannot import add_stat_annotation
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.legend import Legend
import matplotlib.gridspec as gridspec

import generalfunctions as gf

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


def get_clinical_metadata():

    metadata = pd.read_csv('../PEMBRORT_CLINICAL_METADATA_FORSCSEQ_KHG20210624.csv',index_col=None,header=0)

    return metadata

def get_total_counts():

    cellcounts = pd.read_csv('../global_cellnumbers.csv',index_col=None,header=0)
    return cellcounts

def pct_df(adata,groupby='leiden',rep='cohort',xorder='treatment',drop_na=True,thresh=None,use_global=False,version=None):
    
    ## calculate cluster percentages for each sample
    pt = adata.obs[rep].unique().tolist()
    tx = adata.obs[xorder].unique().tolist()
    grps = adata.obs[groupby].unique().tolist()

    idx = pd.MultiIndex.from_product([pt,tx,grps],names=[rep,xorder,groupby])
    df = pd.DataFrame(index=idx,columns=['percent'])

    if thresh is None:
        cellthresh = len(grps)
    else:
        cellthresh = thresh

    if use_global:
        cellcounts = get_total_counts()

    for i in df.index:
        
        total = adata[(adata.obs[idx.names[0]]==i[0])&(adata.obs[idx.names[1]]==i[1])]

        if use_global:
            batch = i[0]+'_'+i[1]
            if batch in cellcounts.index:
                totalsize = cellcounts.loc[batch,'numcells']
            else:
                continue
        else:
            totalsize = len(total)
        
        if (totalsize>=cellthresh):
            df.loc[i,'percent'] = (len(total[(total.obs[idx.names[2]]==i[2])])/totalsize)*100

    df.reset_index(inplace=True)

    metadata = get_clinical_metadata()

    df = df.merge(metadata,left_on='cohort',right_on='Patient_Number')
    
    if drop_na:
        df.dropna(how='any',inplace=True)
    
    return df

def pct_df_faster(adata,groupby='leiden',rep='cohort',xcat='treatment',drop_na=True,thresh=None,use_global=False,normalization=None,receptor_column=None,version=None):
    
    pt = adata.obs[rep].unique().tolist()
    tx = adata.obs[xcat].unique().tolist()
    grps = adata.obs[groupby].unique().tolist()

    df = pd.DataFrame(columns=[rep,xcat,groupby,'percent'])

    idx = pd.MultiIndex.from_product([pt,tx],names=[rep,xcat])

    if thresh is None:
        cellthresh = len(grps)
    else:
        cellthresh = thresh

    if use_global:
        cellcounts = get_total_counts()

    for i in idx:

        total = adata[(adata.obs[idx.names[0]]==i[0])&(adata.obs[idx.names[1]]==i[1])].obs

        if (normalization=='clonotype'):
            total.drop_duplicates(subset=receptor_column,inplace=True)

        if use_global:
            batch = i[0]+'_'+i[1]
            if batch in cellcounts.index:
                totalsize = cellcounts.loc[batch,'numcells']
            else:
                continue
        else:
            totalsize = len(total)

        if (totalsize>=cellthresh):

            counts = total[groupby].value_counts()

            counts.index = counts.index.tolist()
            ms = [x for x in grps if x not in counts.index]
            for m in ms:
                counts.loc[m] = 0
            counts = counts.loc[grps]

            counts = (counts/totalsize)*100
  
            counts = pd.DataFrame(counts.to_numpy(),index=counts.index,columns=['percent'])
            counts[rep] = i[0]
            counts[xcat] = i[1]
            counts[groupby] = counts.index.tolist()
            counts.reset_index(inplace=True)
            counts.drop(columns='index',inplace=True)

            df = pd.concat([df, counts])
    
    return df

## calculate normalized shannon entropy clonality
## clonality = 1 - (SE/ln(# of tcrs))
def shannon(adata,groupby='leiden',rep='cohort',xcat='treatment',drop_na=True,thresh=None,normalization='cells',receptor_column=None,version=None):

    pt = adata.obs[rep].unique().tolist()
    tx = adata.obs[xcat].unique().tolist()

    idx = pd.MultiIndex.from_product([pt,tx],names=[rep,xcat])
    df = pd.DataFrame(index=idx,columns=['shannon'])

    st = adata.obs[groupby].unique().tolist()
    totalgrps = len(st)
    
    if thresh is None:
        thresh = totalgrps

    for i in df.index:
        
        total = adata[(adata.obs[idx.names[0]]==i[0])&(adata.obs[idx.names[1]]==i[1])].obs

        if (normalization=='clonotype'):
            total.drop_duplicates(subset=receptor_column,inplace=True)

        totalsize = len(total)
        
        if (totalsize>=thresh):

            grps = total[groupby].value_counts()
            
            grps = grps/totalsize
            
            runningSum = 0
            
            for g in grps:
                runningSum += -1*(g * np.log(g))

            df.loc[i,'shannon'] = (1 - (runningSum/np.log(totalgrps)))

    df.reset_index(inplace=True)

    metadata = get_clinical_metadata()

    df = df.merge(metadata,left_on='cohort',right_on='Patient_Number')

    if drop_na:
        df.dropna(how='any',inplace=True)

    return df
    

def pct_comparison(adata,df=None,groupby='leiden',rep='cohort',xcat='treatment',hcat='pCR',ycat=None,normalization='cells',receptor_column=None,logic=None,
                    xorder=None,horder=None,show_stats=True,calc_pct=False,calc_pct_faster=False,calc_shannon=False,version=None,drop_na=True,thresh=None,use_global=False,
                    dotsize=2,fontsize=4,ylim=None,figsize=None,return_df=False,size_max=None,vmin=-1,vmax=1,plot_type='boxplot',tight=False):
                
    tmp = adata.copy()

    if isinstance(xcat,dict):

        tmp.obs['marker'] = pd.DataFrame(index=tmp.obs.index,columns=['marker'])
        
        for element in xcat:

            comparator = xcat[element]

            if (logic=='and'):
                tmp.obs.loc[[all(tmp.obs.loc[x,y]==comparator[y] for y in comparator) for x in tmp.obs.index],'marker'] = element

            elif (logic=='or'):
                tmp.obs.loc[[any(tmp.obs.loc[x,y]==comparator[y] for y in comparator) for x in tmp.obs.index],'marker'] = element

        xcat = 'marker'

    if calc_pct:
        df = pct_df(tmp,groupby=groupby,rep=rep,xorder=xcat,drop_na=drop_na,thresh=thresh,use_global=use_global)
        ycat = 'percent'
        print('calculated percent')
    elif calc_pct_faster:
        df = pct_df_faster(tmp,groupby=groupby,rep=rep,xcat=xcat,drop_na=drop_na,thresh=thresh,use_global=use_global,normalization=normalization,receptor_column=receptor_column)
        ycat = 'percent'
        print('calculated percent')
    elif calc_shannon:
        df = shannon(tmp,groupby=groupby,rep=rep,xcat=xcat,drop_na=drop_na,thresh=thresh,normalization=normalization,receptor_column=receptor_column)
        ycat = 'shannon'
    else:
        metadata = get_clinical_metadata()
        cols = [x for x in metadata.columns if x not in df.columns] + ['Patient_Number']
        cols = np.unique(cols)
        df = df.merge(metadata[cols],left_on='cohort',right_on='Patient_Number')  
        if drop_na:
            df.dropna(how='any',inplace=True)

    if xorder is None:
        xorder = df[xcat].unique().tolist()
        cols = xorder
    if horder is None and hcat is not None:
        horder = df[hcat].unique().tolist()

    if (plot_type=='boxplot'):

        if (groupby is None) | (calc_shannon):
        
            combs = list(itertools.combinations(horder,2))

            if figsize is None:
                figsize = (2,2)

            fig,axs = plt.subplots(figsize=figsize)

            _= sns.boxplot(x=xcat,y=ycat,hue=hcat,data=df,color='w',linewidth=0.5,fliersize=0,palette='colorblind',order=xorder,hue_order=horder,boxprops=dict(alpha=0.2),ax=axs)
            
            if show_stats:
                try:
                    test_results = add_stat_annotation(axs,x=xcat,y=ycat,hue=hcat,data=df,order=xorder,hue_order=horder,
                                                        box_pairs = [tuple((x,y) for y in c) for c in combs for x in xorder],
                                                        test='Mann-Whitney',comparisons_correction=None,text_format='simple',
                                                        loc='outside',verbose=0,fontsize=4,linewidth=0.5,line_height=0.01,text_offset=0.01)
                except:
                    pass

            _= sns.stripplot(x=xcat,y=ycat,hue=hcat,dodge=True,jitter=0.1,data=df,size=dotsize,order=xorder,hue_order=horder,ax=axs)

            if ylim is None:
                ymin = df[ycat].min() - 0.1*df[ycat].min()
                ymax = df[ycat].max() + 0.1*df[ycat].max()
                ylim = (ymin,ymax)

            _= axs.set(ylim=ylim,xlabel='')
            _= axs.set_ylabel(ycat,fontsize=fontsize)
            _= axs.legend(fontsize=2,title_fontsize=2, markerscale=0.05)
            _= axs.grid(visible=False)
            _= axs.set_xticks(np.arange(len(xorder)))
            _= axs.set_xticklabels(xorder,rotation=90)
            _= axs.tick_params(axis='both',labelsize=fontsize)
            
            if tight:
                plt.tight_layout()

        else:
        
            st = df[groupby].unique().tolist()
            
            combs = list(itertools.combinations(horder,2))
            
            if figsize is None:
                figsize = (len(st)*2,1)
   
            fig,axs = plt.subplots(nrows=1,ncols=len(st),sharex=False,sharey=False,figsize=figsize)

            for s,ax in zip(st,np.ravel(axs)):

                tmp = df.loc[[x for x in df.index if df.loc[x,groupby]==s],:]

                if (tmp[ycat].sum()==0):
                    continue

                _= sns.boxplot(x=xcat,y=ycat,hue=hcat,data=tmp,color='w',linewidth=0.5,fliersize=0,palette='colorblind',order=xorder,hue_order=horder,boxprops=dict(alpha=0.2),ax=ax)
                
                if show_stats:
                    try:
                        test_results = add_stat_annotation(ax,x=xcat,y=ycat,hue=hcat,data=tmp,order=xorder,hue_order=horder,
                                                            box_pairs = [tuple((x,y) for y in c) for c in combs for x in xorder],
                                                            test='Mann-Whitney',comparisons_correction=None,text_format='simple',
                                                            loc='outside',verbose=0,fontsize=4,linewidth=0.5,line_height=0.01,text_offset=0.01)
                    except:
                        pass

                _= sns.stripplot(x=xcat,y=ycat,hue=hcat,dodge=True,jitter=0.1,data=tmp,size=dotsize,order=xorder,hue_order=horder,ax=ax)
                
                if ylim is None:
                    ymin = df[ycat].min() - 0.1*df[ycat].min()
                    ymax = df[ycat].max() + 0.1*df[ycat].max()
                    ylim = (ymin,ymax)

                _= ax.set(ylim=ylim,xlabel='')
                _= ax.set_title(s,fontsize=fontsize)
                _= ax.set_ylabel(ycat,fontsize=fontsize)
                _= ax.legend(fontsize=2,title_fontsize=2, markerscale=0.05)
                _= ax.grid(visible=False)
                _= ax.set_xticks(np.arange(len(xorder)))
                _= ax.set_xticklabels(xorder,rotation=90)
                _= ax.tick_params(axis='both',labelsize=fontsize)
            
            if tight:
                plt.tight_layout()
            
    elif (plot_type=='heatmap'):

        st = df[groupby].unique().tolist()

        effect_size = pd.DataFrame(index=st,columns=cols,dtype=np.float64)
        pval = pd.DataFrame(index=st,columns=cols,dtype=np.float64)
        
        for s in st:
            for c in cols:
                
                y0 = df[(df[groupby]==s)&(df[xcat]==c)&(df[hcat]==horder[0])][ycat].to_numpy(dtype=np.float64)
                y1 = df[(df[groupby]==s)&(df[xcat]==c)&(df[hcat]==horder[1])][ycat].to_numpy(dtype=np.float64)
                S,P = scipy.stats.ranksums(y0,y1)
                
                effect_size.loc[s,c] = S
                pval.loc[s,c] = P
        
        pval = -1*np.log10(pval.to_numpy(dtype=np.float64))
        pval[np.isinf(pval)] = np.max(pval[~np.isinf(pval)])
        pval = pd.DataFrame(pval,index=st,columns=cols)
        
        d = scipy.spatial.distance.pdist(effect_size.to_numpy(dtype='float64'),metric='euclidean')
        l = scipy.cluster.hierarchy.linkage(d,metric='euclidean',method='complete',optimal_ordering=True)
        dn = scipy.cluster.hierarchy.dendrogram(l,no_plot=True)
        order = dn['leaves']
        
        effect_size = effect_size.iloc[order,:]
        pval = pval.iloc[order,:]
        
        if size_max is None:
            size_max = pval.max().item()

        ref_s = [1.30,size_max]
        ref_l = ['0.05','maxsize: '+'{:.1e}'.format(10**(-1*size_max))]

        if figsize is None:
            figsize = (2,len(st))
        
        fig,axs = plt.subplots(nrows=1,ncols=2,figsize=figsize)
        
        gf.heatmap2(effect_size,cmap='RdBu_r',vmin=vmin,vmax=vmax,cellsize=pval,square=True,cellsize_vmax=size_max,ref_sizes=ref_s,ref_labels=ref_l,fontsize=fontsize,figsize=figsize,ax=axs[0])

        icoord = np.array(dn['icoord'] )
        dcoord = np.array(dn['dcoord'] )

        for xs, ys in zip(dcoord, icoord):
            _= axs[1].plot(xs, ys, color='k', linewidth=0.5)

        axs[1].grid(visible=False)
        axs[1].set(xticks=[],yticks=[])
        

    if return_df:
        return fig,axs,df
    else:
        return fig,axs


def fisher_response(adata,df=None,groupby='leiden',metric='percent',calc_pct=True,clip=4,drop_na=True,thresh=None,use_global=False,remove_ns=False,plot_umap=False,version=None):
    
    if calc_pct:
        df = pct_df(adata,groupby=groupby,drop_na=True,thresh=thresh,use_global=use_global)
    else:
        metadata = get_clinical_metadata()
        cols = [x for x in metadata.columns if x not in df.columns]
        df = df.merge(metadata[cols],left_on='cohort',right_on='Patient_Number')  
        if drop_na:
            df.dropna(how='any',inplace=True)

    tx = df['treatment'].unique().tolist()
    grps = df[groupby].unique().tolist()

    mediandf = pd.DataFrame(index=pd.MultiIndex.from_product([tx,grps],names=['treatment',groupby]),columns=['med'])
    for i in mediandf.index:
        mediandf.loc[i,'med'] = df[(df['treatment']==i[0])&(df[groupby]==i[1])][metric].median()

    mediandf.reset_index(inplace=True)

    ## generate contingency tables
    ngrps = len(grps)
    ntx = len(tx)

    fishtable = np.zeros(shape=(2,2,ngrps,ntx),dtype=int)

    for i,g in enumerate(grps):
        for j,t in enumerate(tx):

            tmp = df[(df[groupby]==g)&(df['treatment']==t)]

            med = mediandf[(mediandf[groupby]==g)&(mediandf['treatment']==t)].med.item()

            fishtable[0,0,i,j] = len(tmp[(tmp.pCR=='R')&(tmp[metric]>med)])
            fishtable[1,0,i,j] = len(tmp[(tmp.pCR=='R')&(tmp[metric]<=med)])
            fishtable[0,1,i,j] = len(tmp[(tmp.pCR=='NR')&(tmp[metric]>med)])
            fishtable[1,1,i,j] = len(tmp[(tmp.pCR=='NR')&(tmp[metric]<=med)])


    ## collect fisher's exact test results and plot in heatmap form
    results = pd.DataFrame(index=pd.MultiIndex.from_product([tx,grps],names=['treatment',groupby]),columns=['pval','ratio'])

    for i,g in enumerate(grps):
        for j,t in enumerate(tx):

            ratio,pval = scipy.stats.fisher_exact(fishtable[:,:,i,j])

            results.loc[(t,g),'pval'] = pval
            results.loc[(t,g),'ratio'] = ratio

    results.reset_index(inplace=True)

    pvals = results[['treatment',groupby,'pval']].pivot(index='treatment',columns=groupby)
    ratios = results[['treatment',groupby,'ratio']].pivot(index='treatment',columns=groupby)
    ratios.columns = ratios.columns.droplevel()
    pvals.columns = pvals.columns.droplevel()

    ratios.clip(upper=clip,inplace=True)

    from matplotlib import cm

    minima = ratios.min().min()
    maxima = ratios.max().max()

    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=False)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.RdBu_r)

    test = mapper.to_rgba(ratios.to_numpy(dtype=np.float64))

    for i in np.arange(test.shape[0]):
        for j in np.arange(test.shape[1]):
            if remove_ns:
                if (pvals.iloc[i,j]>0.05):
                    test[i,j,3] = 0
            else:
                test[i,j,3] = 1 - pvals.iloc[i,j]
    
    if plot_umap:

        fig,axs = plt.subplots(nrows=4,ncols=1,figsize=(2,9))
        
        _= axs[0].imshow(X=test)
        _= axs[0].grid(visible=False)
        _= axs[0].set(xticks=np.arange(test.shape[1]),xticklabels=ratios.columns.tolist(),
                yticks=np.arange(test.shape[0]),yticklabels=ratios.index.tolist())
        _= axs[0].tick_params(axis='both',labelsize=4)

        for i in np.arange(test.shape[0]):
            for j in np.arange(test.shape[1]):
                _= axs[0].text(j, i, np.round(pvals.iloc[i,j],decimals=2), ha='center', va='center', color='k', fontsize=2)

        dotsize = (120000/len(adata))*2

        for i,t in enumerate(tx):
            clustcolors = {g:test[np.argwhere(ratios.index==t).item(),np.argwhere(ratios.columns==g).item(),:] for g in grps}
            
            for g in grps:
                xs = adata[(adata.obs.leiden==g)].obsm['X_umap'][:,0]
                ys = adata[(adata.obs.leiden==g)].obsm['X_umap'][:,1]
                _= axs[i+1].scatter(x=xs,y=ys,color=clustcolors[g],s=0.2,linewidths=0.001,edgecolors='k')
            _= axs[i+1].set_xticks([])
            _= axs[i+1].set_yticks([])
            _= axs[i+1].grid(visible=False)
            _= axs[i+1].set_title(t,fontdict={'fontsize':4},pad=0.1)

    else:

        fig,axs = plt.subplots(figsize=(2,1))
        
        _= axs.imshow(X=test)
        _= axs.grid(visible=False)
        _= axs.set(xticks=np.arange(test.shape[1]),xticklabels=ratios.columns.tolist(),
                yticks=np.arange(test.shape[0]),yticklabels=ratios.index.tolist())
        _= axs.tick_params(axis='both',labelsize=4)

        for i in np.arange(test.shape[0]):
            for j in np.arange(test.shape[1]):
                _= axs.text(j, i, np.round(pvals.iloc[i,j],decimals=2), ha='center', va='center', color='k', fontsize=2)

    return fig,axs,ratios,pvals


def pct_lineplot(adata,df=None,groupby='leiden',rep='cohort',xcat='treatment',hcat='pCR',
                xorder=['Base','PD1','RTPD1'],horder=['R','NR'],show_stats=True,thresh=None,
                 direction='two-sided',calc_pct=True,drop_na=True,version=None):
    
    if calc_pct:
        df = pct_df(adata,groupby=groupby,rep=rep,xorder=xcat,drop_na=drop_na,thresh=thresh)
    
    st = df[groupby].unique().tolist()
          
    ##lineplot

    forder = [(x,y) for x in horder for y in st]

    fig,axs = plt.subplots(nrows=len(horder),ncols=len(st),sharex=True,sharey=False,figsize=(len(st),len(horder)))
    sns.set_style("white", rc={"font.family":"Helvetica","axes.grid":False})                                                  
    sns.set_context("paper", rc={"font.size":4,"axes.titlesize":4,"axes.labelsize":4,"font.family":"Helvetica","xtick.labelsize":4,"ytick.labelsize":4})

    for f,ax in zip(forder,np.ravel(axs)):

        h = f[0]
        s = f[1]

        tmp = df[(df[hcat]==h)&(df[groupby]==s)]

        pt = tmp.cohort.unique().tolist()

        for p in pt:
            if len(tmp[(tmp.cohort==p)].treatment.unique().tolist()) < 3:
                tmp = tmp[(tmp.cohort!=p)]

        ys1 = tmp[(tmp.treatment=='Base')]['percent'].to_numpy(dtype='float64')
        ys2 = tmp[(tmp.treatment=='PD1')]['percent'].to_numpy(dtype='float64')
        ys3 = tmp[(tmp.treatment=='RTPD1')]['percent'].to_numpy(dtype='float64')

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

        _= ax.set(xticks=(1,2,3),xticklabels=['Base','PD1','RTPD1'],
                  ylim=(-2,tmp.percent.max()+5),ylabel='percent',
                  title=s+' '+h)
        
        if show_stats:
            if (np.count_nonzero(np.append(ys2,ys1))>0):
                S,P = scipy.stats.wilcoxon(ys2,ys1,alternative=direction)
                if (P<=0.05):
                    _= ax.text(x=1, y=1, s='B2P: '+str(np.round(S,decimals=1))+', p='+str(np.round(P,decimals=3)),color='k')
            if (np.count_nonzero(np.append(ys3,ys2))>0):
                S,P = scipy.stats.wilcoxon(ys3,ys2,alternative=direction)
                if (P<=0.05):
                    _= ax.text(x=1, y=0.9, s='P2R: '+str(np.round(S,decimals=1))+', p='+str(np.round(P,decimals=3)),color='k')
            if (np.count_nonzero(np.append(ys3,ys1))>0):
                S,P = scipy.stats.wilcoxon(ys3,ys1,alternative=direction)
                if (P<=0.05):
                    _= ax.text(x=1, y=0.8, s='B2R: '+str(np.round(S,decimals=1))+', p='+str(np.round(P,decimals=3)),color='k')
            
    plt.tight_layout()
    return fig,axs

def med_lineplot(adata,df=None,groupby='leiden',rep='cohort',xcat='treatment',hcat='pCR',thresh=None,
                xorder=['Base','PD1','RTPD1'],horder=['R','NR'],calc_pct=True,drop_na=True,version=None):
    
    if calc_pct:
        df = pct_df(adata,groupby=groupby,rep=rep,xorder=xcat,drop_na=drop_na,thresh=thresh)
    
    st = df[groupby].unique().tolist()
    
    ##lineplot

    fig,axs = plt.subplots(nrows=1,ncols=len(st),sharex=True,sharey=False,figsize=(len(st)*2,1))
    sns.set_style("white", rc={"font.family":"Helvetica","axes.grid":False})                                                  
    sns.set_context("paper", rc={"font.size":4,"axes.titlesize":4,"axes.labelsize":4,"font.family":"Helvetica","xtick.labelsize":4,"ytick.labelsize":4})
    
    ##lineplot colors
    cols = {'R':'r',
           'NR':'b'}
    
    for s,ax in zip(st,np.ravel(axs)):
        for h in horder:

            
            tmp = df[(df[hcat]==h)&(df[groupby]==s)]
            
            maxV = df[(df[groupby]==s)].percent.max()
            
            pt = tmp.cohort.unique().tolist()

            for p in pt:
                if len(tmp[(tmp.cohort==p)].treatment.unique().tolist()) < 3:
                    tmp = tmp[(tmp.cohort!=p)]

            ys1 = tmp[(tmp.treatment=='Base')]['percent'].to_numpy(dtype='float64')
            ys2 = tmp[(tmp.treatment=='PD1')]['percent'].to_numpy(dtype='float64')
            ys3 = tmp[(tmp.treatment=='RTPD1')]['percent'].to_numpy(dtype='float64')

            xs1 = np.full(shape=(len(ys1),1),fill_value=1)
            xs2 = np.full(shape=(len(ys2),1),fill_value=2)
            xs3 = np.full(shape=(len(ys3),1),fill_value=3)

            counter = 0
            for x1,y1,x2,y2 in zip(xs1,ys1,xs2,ys2):
                _= ax.plot((x1,x2),(y1,y2),marker='.',color=cols[h],alpha=0.2,markersize=0.01,linewidth=0.1)
                counter += 1
            counter = 0
            for x1,y1,x2,y2 in zip(xs2,ys2,xs3,ys3):
                _= ax.plot((x1,x2),(y1,y2),marker='.',color=cols[h],alpha=0.2,markersize=0.01,linewidth=0.1)
                counter += 1
            
            P1=P2=P3=P4=P5=P6=1
            if (np.count_nonzero(np.append(ys2,ys1))>0):
                S1,P1 = scipy.stats.wilcoxon(ys2,ys1,alternative='greater')
                S2,P2 = scipy.stats.wilcoxon(ys2,ys1,alternative='less')
            if (np.count_nonzero(np.append(ys3,ys2))>0):
                S3,P3 = scipy.stats.wilcoxon(ys3,ys2,alternative='greater')
                S4,P4 = scipy.stats.wilcoxon(ys3,ys2,alternative='less')
            if (np.count_nonzero(np.append(ys3,ys1))>0):
                S5,P5 = scipy.stats.wilcoxon(ys3,ys1,alternative='greater')
                S6,P6 = scipy.stats.wilcoxon(ys3,ys1,alternative='less')

            avgys1 = np.median(ys1)
            avgys2 = np.median(ys2)
            avgys3 = np.median(ys3)

            if (P1<0.05)|(P2<0.05):
                _ = ax.plot((1,2),(avgys1,avgys2),marker='.',color=cols[h],markersize=0.01,linewidth=0.5,alpha=1,linestyle='--')
            else:
                _ = ax.plot((1,2),(avgys1,avgys2),marker='.',color=cols[h],markersize=0.01,linewidth=0.5,alpha=1,linestyle='-')

            if (P3<0.05)|(P4<0.05):
                _ = ax.plot((2,3),(avgys2,avgys3),marker='.',color=cols[h],markersize=0.01,linewidth=0.5,alpha=1,linestyle='--')
            else:
                _ = ax.plot((2,3),(avgys2,avgys3),marker='.',color=cols[h],markersize=0.01,linewidth=0.5,alpha=1,linestyle='-')

            if (P5<0.05)|(P6<0.05):
                _ = ax.plot((1,3),(avgys1,avgys3),marker='.',color=cols[h],markersize=0.01,linewidth=0.5,alpha=1,linestyle='--')
            else:
                _ = ax.plot((1,3),(avgys1,avgys3),marker='.',color=cols[h],markersize=0.01,linewidth=0.5,alpha=1,linestyle='-')

            _= ax.set(xticks=(1,2,3),xticklabels=['Base','PD1','RTPD1'],
                      ylim=(-2,maxV+5),ylabel='percent',
                      title=s)


    plt.tight_layout()
    return fig,axs

def metadata_query(target_cell=None):
    obs = pd.read_csv('../'+target_cell+'_obs.csv',index_col=0,header=0)
    obs.leiden = [str(x) for x in obs.leiden]
    obs.leiden = obs.leiden.astype('category')
    return obs

def metadata_summarize(obs=None,groupby=None,metric=None,treatment=None,thresh=0,prefix=''):
    
    pt = obs.cohort.unique().tolist()

    if treatment is None:
        tx = obs.treatment.unique().tolist()
    elif isinstance(treatment,list):
        tx = treatment
    else:
        tx = [treatment]

    if groupby is None:
        idx = pd.MultiIndex.from_product([pt,tx],names=['cohort','treatment'])
    else:
        st = list(groupby.keys())
        idx = pd.MultiIndex.from_product([pt,tx,st],names=['cohort','treatment','group'])

    
    if metric == 'percent':
        cols = [prefix+'_'+metric]
    else:
        cols = [prefix+'_'+x for x in metric]

    df = pd.DataFrame(index=idx,columns=cols)
    
    
    for i in df.index:
        
        total = obs[(obs[idx.names[0]]==i[0])&(obs[idx.names[1]]==i[1])]
        totalsize = len(total)
        
        if (totalsize<thresh):
            continue
            
        if groupby is None:
            
            if metric == 'percent':
                print('no')
            else:
                for m,c in zip(metric,df.columns):
                    df.loc[i,c] = total[m].median(axis=0)
                
        else:
            
            total['marker'] = 'N'
            total.loc[[all(total.loc[x,y]==groupby[i[2]][y] for y in groupby[i[2]]) for x in total.index],'marker'] = 'Y'
            
            if metric == 'percent': 
                target = total[(total.marker=='Y')]
                df.loc[i,cols] = (len(target)/totalsize)*100
            else:
                target = total[(total.marker=='Y')]
                df.loc[i,:] = target[metric].median(axis=0)

    if groupby is not None:
        df.reset_index(inplace=True)
        df = df.pivot(index=['cohort','treatment'],columns=['group'],values=cols)
    
    return df


def prediction_query(adata=None,metadata=None,treatment=None,
                     pred_group=None,pred_metric='percent',
                     target_cell=None,target_group=None,target_metric='percent',
                     thresh=10,drop_na=False,
                     return_df=True,plot=False,custom_heatmap=False,vmin=-1,vmax=1,fontsize=4,figsize=(2,2)):
    
    """
    predict a metric (target) in one celltype (target_cell) with a metric (predictor) in another celltype
    
    need to provide either an anndata with obs containing the predictor, or a pandas dataframe
    (metadata) with cell barcodes on the indices and predictor, cohort, and treatment contained within the columns
    
    if group is None, it is assumed you want the metric taken across the whole population
    if group == 'leiden', then the metric will be calculated for each leiden cluster
    if group is a dictionary, the metric will be calculated for each group within the dictionary
    
    if metric is not provided, it is assumed that you want a percent of the total cells
    if metric is provided, then the median within the group will be taken
    
    """
    
    ## deal with target population
    obs = metadata_query(target_cell=target_cell)
    
    if target_group == 'leiden':
        grps = obs['leiden'].unique().tolist()
        grps = {'c'+str(x):{'leiden':x} for x in grps}
    else:
        grps = target_group
  
    target_df = metadata_summarize(obs=obs,groupby=grps,metric=target_metric,treatment=treatment,thresh=thresh,prefix='target')

    
    
    ## deal with predictor population
    if adata is None:
        obs = metadata
    else:
        obs = adata.obs
        
    if pred_group == 'leiden':
        grps = obs['leiden'].unique().tolist()
        grps = {'c'+str(x):{'leiden':x} for x in grps}
    else:
        grps = pred_group
        
    pred_df = metadata_summarize(obs=obs,groupby=grps,metric=pred_metric,treatment=treatment,thresh=thresh,prefix='pred')
 
    
    
    ## combine dataframes
    df = target_df.merge(pred_df,left_index=True,right_index=True)
    
    df.reset_index(inplace=True)
    
    if drop_na:
        df.dropna(how='any',inplace=True)
    
    
    ## plot spearman correlation matrix, if desired
    if plot:
        
        df.dropna(how='any',inplace=True)
        
        cols = [x for x in df.columns if 'target' in x[0] or 'target' in x or 'pred' in x[0] or 'pred' in x]
        
        mat = df[cols].to_numpy()
        
        corr,pval = scipy.stats.spearmanr(mat,axis=0,nan_policy='raise')
        

        corr = pd.DataFrame(corr,index=cols,columns=cols)

        pval = -1*np.log10(pval)
        pval[np.isinf(pval)] = np.max(pval[~np.isinf(pval)])
        pval = pd.DataFrame(pval,index=cols,columns=cols)

        d = scipy.spatial.distance.pdist(corr.to_numpy(dtype='float64'),metric='euclidean')
        l = scipy.cluster.hierarchy.linkage(d,metric='euclidean',method='complete',optimal_ordering=True)
        dn = scipy.cluster.hierarchy.dendrogram(l,no_plot=True)
        order = dn['leaves']

        corr = corr.iloc[order,order]
        pval = pval.iloc[order,order]

        CL = []
        for c in cols:
            if 'target' in c[0] or 'target' in c:
                CL = CL + ['r']
            else:
                CL = CL + ['b']

        if custom_heatmap:
            
            size_max = 4
            ref_s = [1.30,2.00,3.00,size_max]
            ref_l = ['0.05','0.01','1e-3','maxsize: '+'{:.1e}'.format(10**(-1*size_max))]

            fig,ax = plt.subplots(nrows=2,ncols=1,figsize=figsize)
            gf.heatmap2(corr,cmap='RdBu_r',vmin=vmin,vmax=vmax,cellsize=pval,square=True,cellsize_vmax=size_max,ref_sizes=ref_s,ref_labels=ref_l,rowcolors=CL,colcolors=CL,fontsize=fontsize,figsize=figsize,ax=ax[0])

            icoord = np.array(dn['icoord'] )
            dcoord = np.array(dn['dcoord'] )

            for xs, ys in zip(icoord, dcoord):
                _= ax[1].plot(xs, ys, color='k', linewidth=0.5)
            
            ax[1].grid(visible=False)

        else:

            labels = [str(x) for x in cols]
            
            sns.set_style("white", rc={"font.family":"Helvetica","axes.grid":False})                                                  
            sns.set_context("paper", rc={"font.size":fontsize,"axes.titlesize":fontsize,"axes.labelsize":fontsize,"font.family":"Helvetica","xtick.labelsize":fontsize,"ytick.labelsize":fontsize})

            ax = sns.clustermap(corr,xticklabels=labels,yticklabels=labels,row_cluster=False,col_cluster=False,
                                cmap='RdBu_r',vmin=vmin,vmax=vmax,dendrogram_ratio=(0.05,0.05),
                                row_colors=CL,col_colors=CL,method='complete',figsize=figsize)
        
    
    if return_df & plot:
        return df, ax
    elif return_df:
        return df
    elif plot:
        return ax


def summarize_metric(adata,catnames=None,calc='median',metric=None,gene=False,use_raw=False,thresh=None,drop_na=False,version=None):

    cats = [adata.obs[c].unique().tolist() for c in catnames]

    idx = pd.MultiIndex.from_product(cats,names=catnames)

    df = pd.DataFrame(index=idx,columns=metric)

    if gene:
        
        if use_raw:
            tmp = adata[:,metric].raw.X.toarray()
        else:
            tmp = adata[:,metric].X.toarray()

        tmp = pd.DataFrame(tmp,index=adata.obs.index,columns=metric)
        tmp = tmp.merge(adata.obs,how='left',left_index=True,right_index=True)
    
    else:

        tmp = adata.obs


    if (calc=='median'):
        df = tmp.groupby(catnames)[metric].agg(['count','median']).reset_index()
    elif (calc=='mean'):
        df = tmp.groupby(catnames)[metric].agg(['count','mean']).reset_index()

    if thresh is not None:
        cols = [x for x in df.columns if 'count' in x[1]]
        df = df[(df[cols[0]]>=thresh)]

    metadata = get_clinical_metadata()

    df = df.merge(metadata,left_on='cohort',right_on='Patient_Number')

    ## deal with the multiindex column
    cols = df.columns.tolist()
    cols2 = []
    for c in cols:
        if (type(c) is tuple):
            if (calc in c) | ('count' in c):
                cols2 = cols2 + [c[0]+'_'+c[1]]
            else:
                cols2 = cols2 + [c[0]]
        else:
            cols2 = cols2 + [c]
    df.columns = cols2

    if drop_na:
        df.dropna(how='any',inplace=True)

    return df


def plot_summary_metric(adata,df=None,calc_metric=True,catnames=None,calc='median',metric=None,gene=False,use_raw=False,thresh=None,drop_na=False,
                        plot_type='boxplot',xcat=None,hcat=None,xorder=None,horder=None,show_stats=False,ylim=None,
                        nrows=None,ncols=None,figsize=None):

    if calc_metric:
        df = summarize_metric(adata,catnames=catnames,calc=calc,metric=metric,gene=gene,use_raw=use_raw,thresh=thresh,drop_na=drop_na)

    if (plot_type=='boxplot'):

        ## so we can use for-loop down there even if there is only 1 metric
        if nrows==ncols==1:
            nrows = 2
            metric = metric + ['blank']

        if figsize is None:
            fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(ncols,nrows))
        else:
            fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=figsize)

        if xorder is None:
            xorder = sorted(df[xcat].unique().tolist())
        if (horder is None) & (hcat is not None):
            horder = df[hcat].unique().tolist()

        if horder is not None:
            combs = list(itertools.combinations(horder,2))
            bp = [tuple((x,y) for y in c) for c in combs for x in xorder]
        else:
            bp = list(itertools.combinations(xorder,2))

        for m,ax in zip(metric,axs.flat):
            
            if (m=='blank'):
                _= ax.set(ylabel='',xlabel='',xticks=[],yticks=[])
                _= ax.grid(visible=False)
                continue
            
            ycat = m+'_'+calc

            _= sns.boxplot(x=xcat,y=ycat,hue=hcat,data=df,color='w',linewidth=0.5,fliersize=0,palette='colorblind',order=xorder,hue_order=horder,boxprops=dict(alpha=0.2),ax=ax)
            
            if show_stats:
                try:
                    test_results = add_stat_annotation(ax,x=xcat,y=ycat,hue=hcat,data=df,order=xorder,hue_order=horder,
                                                        box_pairs = bp,
                                                        test='Mann-Whitney',comparisons_correction=None,text_format='simple',
                                                        loc='outside',verbose=0,fontsize=2,linewidth=0.5,line_height=0.01,text_offset=0.01)
                except:
                    pass

            _= sns.stripplot(x=xcat,y=ycat,hue=hcat,dodge=True,jitter=0.1,data=df,size=1,order=xorder,hue_order=horder,ax=ax)
            
            if horder is not None:
                _= ax.legend(fontsize=2,title_fontsize=2,markerscale=0.05)
            
            minV = df[ycat].min() - 0.1*df[ycat].min()
            maxV = df[ycat].max() + 0.1*df[ycat].max()

            if ylim is None:
                ylim = (minV,maxV)
        
            _= ax.set(ylim=ylim,xlabel='')
            _= ax.set_ylabel(m,fontsize=4)
            _= ax.tick_params(axis='both',labelsize=4)
            _= ax.grid(visible=False)


    elif (plot_type=='lineplot'):
        
        st = sorted(adata.obs.leiden.unique().tolist())

        if figsize is None:
            figsize = (len(st),len(metric))

        fig,axs = plt.subplots(nrows=len(metric),ncols=len(st),sharex=True,sharey=False,figsize=figsize)
        
        ##lineplot colors
        cols = {'R':'r',
                'NR':'b'}
        
        forder = [(x,y) for x in metric for y in st]

        for f,ax in zip(forder,axs.flat):
            for h in horder:
                
                m = f[0]
                s = f[1]

                ycat = m+'_'+calc

                tmp = df[(df[hcat]==h)&(df['leiden']==s)]
                
                maxV = df[(df['leiden']==s)][ycat].max()
                
                pt = tmp.cohort.unique().tolist()

                for p in pt:
                    if len(tmp[(tmp.cohort==p)].treatment.unique().tolist()) < 3:
                        tmp = tmp[(tmp.cohort!=p)]

                ys1 = tmp[(tmp.treatment=='Base')][ycat].to_numpy(dtype='float64')
                ys2 = tmp[(tmp.treatment=='PD1')][ycat].to_numpy(dtype='float64')
                ys3 = tmp[(tmp.treatment=='RTPD1')][ycat].to_numpy(dtype='float64')

                xs1 = np.full(shape=(len(ys1),1),fill_value=1)
                xs2 = np.full(shape=(len(ys2),1),fill_value=2)
                xs3 = np.full(shape=(len(ys3),1),fill_value=3)

                counter = 0
                for x1,y1,x2,y2 in zip(xs1,ys1,xs2,ys2):
                    _= ax.plot((x1,x2),(y1,y2),marker='.',color=cols[h],alpha=0.2,markersize=0.01,linewidth=0.1)
                    counter += 1
                counter = 0
                for x1,y1,x2,y2 in zip(xs2,ys2,xs3,ys3):
                    _= ax.plot((x1,x2),(y1,y2),marker='.',color=cols[h],alpha=0.2,markersize=0.01,linewidth=0.1)
                    counter += 1
                
                P1=P2=P3=P4=P5=P6=1
                if (np.count_nonzero(np.append(ys2,ys1))>0):
                    S1,P1 = scipy.stats.wilcoxon(ys2,ys1,alternative='greater')
                    S2,P2 = scipy.stats.wilcoxon(ys2,ys1,alternative='less')
                if (np.count_nonzero(np.append(ys3,ys2))>0):
                    S3,P3 = scipy.stats.wilcoxon(ys3,ys2,alternative='greater')
                    S4,P4 = scipy.stats.wilcoxon(ys3,ys2,alternative='less')
                if (np.count_nonzero(np.append(ys3,ys1))>0):
                    S5,P5 = scipy.stats.wilcoxon(ys3,ys1,alternative='greater')
                    S6,P6 = scipy.stats.wilcoxon(ys3,ys1,alternative='less')

                avgys1 = np.median(ys1)
                avgys2 = np.median(ys2)
                avgys3 = np.median(ys3)

                if (P1<0.05)|(P2<0.05):
                    _ = ax.plot((1,2),(avgys1,avgys2),marker='.',color=cols[h],markersize=0.01,linewidth=0.5,alpha=1,linestyle='--')
                else:
                    _ = ax.plot((1,2),(avgys1,avgys2),marker='.',color=cols[h],markersize=0.01,linewidth=0.5,alpha=1,linestyle='-')

                if (P3<0.05)|(P4<0.05):
                    _ = ax.plot((2,3),(avgys2,avgys3),marker='.',color=cols[h],markersize=0.01,linewidth=0.5,alpha=1,linestyle='--')
                else:
                    _ = ax.plot((2,3),(avgys2,avgys3),marker='.',color=cols[h],markersize=0.01,linewidth=0.5,alpha=1,linestyle='-')

                if (P5<0.05)|(P6<0.05):
                    _ = ax.plot((1,3),(avgys1,avgys3),marker='.',color=cols[h],markersize=0.01,linewidth=0.5,alpha=1,linestyle='--')
                else:
                    _ = ax.plot((1,3),(avgys1,avgys3),marker='.',color=cols[h],markersize=0.01,linewidth=0.5,alpha=1,linestyle='-')

                minV = df[ycat].min() - 0.1*df[ycat].min()
                maxV = df[ycat].max() + 0.1*df[ycat].max()

                if ylim is None:
                    ylim = (minV,maxV)

                _= ax.set(xticks=(1,2,3),xticklabels=['Base','PD1','RTPD1'],ylim=ylim,xlabel='')
                _= ax.set_title(s,fontsize=4)
                _= ax.set_ylabel(m,fontsize=4)
                _= ax.tick_params(axis='both',labelsize=4)
                _= ax.grid(visible=False)

    plt.tight_layout()

    return df,fig,axs


## pie chart of cluster distribution for each category
def custom_pie(adata,comparator=None,reference=None,groupby='leiden',labels=None,fontsize=None,figsize=None):

    samp1 = adata[[all(adata.obs.loc[x,y]==comparator[y] for y in comparator) for x in adata.obs.index]]
    samp2 = adata[[all(adata.obs.loc[x,y]==reference[y] for y in reference) for x in adata.obs.index]] 

    samples = [samp1,samp2]

    if labels is None:
        labels = ['comparator','reference']
    
    grps = sorted(adata.obs[groupby].unique().tolist())
    
    if fontsize is None:
        fontsize = 4
    if figsize is None:
        figsize = (4,2)

    fig,axs = plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,figsize=figsize)

    for label,tmp,ax in zip(labels,samples,np.ravel(axs)):

        counts = tmp.obs[groupby].value_counts()
        counts.index = counts.index.tolist()

        ms = [x for x in grps if x not in counts.index] 
        for m in ms:
            counts.loc[m] = 0

        counts = counts.loc[grps]

        ax.pie(counts,labels=grps,normalize=True,rotatelabels=True,textprops={'fontsize':fontsize})
        ax.set_title(label)
        
    return fig,axs