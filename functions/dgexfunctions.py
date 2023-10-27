import scanpy as sc
from glob import glob
import pandas as pd
import numpy as np
import seaborn as sns
import gseapy as gp
import anndata
import scipy
import re
import os
import matplotlib
import math
import random
import itertools
import sklearn
from statannot import add_stat_annotation
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.legend import Legend
import matplotlib.gridspec as gridspec

import generalfunctions as gf

rcParams.update({'font.size': 8})
rcParams.update({'font.family': 'Helvetica'})
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['svg.fonttype'] = 'none'
rcParams['figure.facecolor'] = (1,1,1,1)

def leiden_dgex(adata,groupby='leiden',use_raw=False,method='wilcoxon',pts=True,tie_correct=True):

    sc.tl.rank_genes_groups(adata,groupby=groupby,use_raw=use_raw,method=method,pts=pts,tie_correct=tie_correct)
    
    grps = adata.obs[groupby].unique().tolist()

    dfList = []
    for i,g in enumerate(grps):
        tmp = sc.get.rank_genes_groups_df(adata,group=str(g))
        tmp['group'] = g
        dfList.append(tmp)

    dgex = pd.concat(dfList)
    
    return dgex


def custom_dgex(adata,comparator,reference,grouping_type='custom',groupby=None,logic='and',bycluster=True,filter_pval=True):

    """
    if grouping_type is 'normal', then creates annotations based on groups from one obs column (groupby).


    if grouping_type is 'custom', then creates custom annotations based on arbritrary groupings of obs columns.
    the comparator and reference should be dictionaries with obs columns as keys and desired groups as values.
    the logic can be set to "and" which means all conditions in the dictionary must be true,
    or it can be set to "or" which means that only one of the conditions in the dictionary must be true.
    
    if grouping_type is 'barcode' then comparator and reference should be lists of cell barcodes.

    if reference is set to None, then the reference will be all cells not contained in the comparator.

    """

    tmp = adata.copy()

    tmp.obs['dgex_marker'] = pd.DataFrame(index=tmp.obs.index,columns=['dgex_marker'])

    if reference is None:
        tmp.obs['dgex_marker'] = 'reference'

    if (grouping_type=='normal'):
        
        tmp.obs.loc[[x for x in tmp.obs.index if tmp.obs.loc[x,groupby] in comparator],'dgex_marker'] = 'comparator'
        
        if reference is not None:
            tmp.obs.loc[[x for x in tmp.obs.index if tmp.obs.loc[x,groupby] in reference],'dgex_marker'] = 'reference'
    
    elif (grouping_type=='custom'):
        
        if (logic=='and'):
            tmp.obs.loc[[all(tmp.obs.loc[x,y]==comparator[y] for y in comparator) for x in tmp.obs.index],'dgex_marker'] = 'comparator'
            if reference is not None:
                tmp.obs.loc[[all(tmp.obs.loc[x,y]==reference[y] for y in reference) for x in tmp.obs.index],'dgex_marker'] = 'reference'
        elif (logic=='or'):
            tmp.obs.loc[[any(tmp.obs.loc[x,y]==comparator[y] for y in comparator) for x in tmp.obs.index],'dgex_marker'] = 'comparator'
            if reference is not None:
                tmp.obs.loc[[any(tmp.obs.loc[x,y]==reference[y] for y in reference) for x in tmp.obs.index],'dgex_marker'] = 'reference'
    
    elif (grouping_type=='barcode'):

        tmp.obs.loc[comparator,'dgex_marker'] = 'comparator'

        if reference is not None:
            tmp.obs.loc[reference,'dgex_marker'] = 'reference'

    print(tmp.obs.dgex_marker.value_counts())

    sc.tl.rank_genes_groups(tmp,groupby='dgex_marker',use_raw=False,groups=['comparator'],
                            reference='reference',method='wilcoxon',pts=True,tie_correct=True)

    dgex = sc.get.rank_genes_groups_df(tmp,group='comparator')
    dgex['group'] = 'all'

    if bycluster:

        grps = tmp.obs.leiden.unique().tolist()

        for g in grps:
            
            tmp2 = tmp[(tmp.obs.leiden==g)].copy()
            compSize = len(tmp2[(tmp2.obs.dgex_marker=='comparator')])
            refSize = len(tmp2[(tmp2.obs.dgex_marker=='reference')])
            
            if (compSize>1)&(refSize>1):
                
                sc.tl.rank_genes_groups(tmp2,groupby='dgex_marker',use_raw=False,groups=['comparator'],
                                        reference='reference',method='wilcoxon',pts=True,tie_correct=True)
                
                df = sc.get.rank_genes_groups_df(tmp2,group='comparator')
                df['group'] = g
                
                dgex = dgex.append(df)

    if filter_pval:
        dgex = dgex[(dgex.pvals<=0.05)]

    return dgex

def dgex_plot(adata,dgex=None,groupby='leiden',topn=20,pvalCutoff=0.05,fcCutoff=1,pctCutoff=0.3,use_FDR=True,
              dendro=False,order=None,plot_type='dotplot',cmap='Reds',figsize=None,vmin=0,vmax=1,fontsize=4,size_max=None):
    
    """
    plot type options are: dotplot, matrixplot, scanpy_heatmap, custom_heatmap

    """

    grp = sorted(adata.obs[groupby].unique().tolist())
    
    GOI = []
    
    for g in grp:
        
        if use_FDR:
            if groupby=='leiden':
                GOI += dgex[(dgex.pvals_adj<=pvalCutoff)&
                            (dgex.logfoldchanges>=fcCutoff)&
                            (dgex.pct_nz_group>=pctCutoff)&
                            (dgex.group==int(g))].sort_values(by='scores',ascending=False).names[:topn].tolist()
            else:
                GOI += dgex[(dgex.pvals_adj<=pvalCutoff)&
                            (dgex.logfoldchanges>=fcCutoff)&
                            (dgex.pct_nz_group>=pctCutoff)&
                            (dgex.group==g)].sort_values(by='scores',ascending=False).names[:topn].tolist()
        else:
            if groupby=='leiden':
                GOI += dgex[(dgex.pvals<=pvalCutoff)&
                            (dgex.logfoldchanges>=fcCutoff)&
                            (dgex.pct_nz_group>=pctCutoff)&
                            (dgex.group==int(g))].sort_values(by='scores',ascending=False).names[:topn].tolist()
            else:
                GOI += dgex[(dgex.pvals<=pvalCutoff)&
                            (dgex.logfoldchanges>=fcCutoff)&
                            (dgex.pct_nz_group>=pctCutoff)&
                            (dgex.group==g)].sort_values(by='scores',ascending=False).names[:topn].tolist()

    _,idx = np.unique(GOI,return_index=True)
    GOI = [GOI[index] for index in sorted(idx)]
    
    tmp = adata[:,GOI].copy()
    
    if dendro:
        sc.tl.dendrogram(tmp,groupby=groupby,var_names=GOI,use_raw=False,optimal_ordering=True)    
    
    sns.set_style("white", rc={"font.family":"Helvetica","axes.grid":False})                                                  
    sns.set_context("paper", rc={"font.size":fontsize,"axes.titlesize":fontsize,"axes.labelsize":fontsize,"font.family":"Helvetica","xtick.labelsize":fontsize,"ytick.labelsize":fontsize})
    
    if figsize is None:
        figsize = (2,2)

    if (plot_type=='dotplot'):
        
        test = sc.pl.dotplot(tmp,var_names=GOI,groupby=groupby,use_raw=False,standard_scale='var',vmin=vmin,vmax=vmax,
                              dendrogram=dendro,categories_order=order,swap_axes=True,figsize=figsize,show=False,return_fig=True)
        
        test.style(color_on='square',cmap=cmap,dot_edge_color='white',
                   dot_edge_lw=0.5,grid=True,size_exponent=3,largest_dot=1,
                  dot_min=0,dot_max=1)
        
        return test
    
    elif (plot_type=='matrixplot'):
        
        test = sc.pl.matrixplot(tmp,var_names=GOI,groupby=groupby,use_raw=False,standard_scale='var',vmin=vmin,vmax=vmax,
                                  dendrogram=dendro,categories_order=order,swap_axes=True,cmap=cmap,figsize=figsize,show=False,return_fig=True)
        
        test.style(edge_lw=0)
        
        return test
    
    elif (plot_type=='scanpy_heatmap'):
        
        ax = sc.pl.heatmap(tmp,var_names=GOI,groupby=groupby,use_raw=False,standard_scale='var',vmin=vmin,vmax=vmax,
                              dendrogram=dendro,categories_order=order,cmap=cmap,swap_axes=True,show_gene_labels=True,figsize=figsize,show=False)
    
        return ax

    elif (plot_type=='custom_heatmap'):
        
        effect_size = pd.DataFrame(index=GOI,columns=grp,dtype=np.float64)
        pval = pd.DataFrame(index=GOI,columns=grp,dtype=np.float64)

        for idx in effect_size.index:
            for col in effect_size.columns:

                effect_size.loc[idx,col] = dgex.loc[(dgex['names']==idx)&(dgex['group']==int(col)),'logfoldchanges'].item()

                if use_FDR:
                    pval.loc[idx,col] = dgex.loc[(dgex['names']==idx)&(dgex['group']==int(col)),'pvals_adj'].item()
                else:
                    pval.loc[idx,col] = dgex.loc[(dgex['names']==idx)&(dgex['group']==int(col)),'pvals'].item()

        pval = -1*np.log10(pval.to_numpy(dtype=np.float64))
        pval[np.isinf(pval)] = np.max(pval[~np.isinf(pval)])
        pval = pd.DataFrame(pval,index=GOI,columns=grp)
        
        d = scipy.spatial.distance.pdist(effect_size.to_numpy(dtype='float64'),metric='euclidean')
        l = scipy.cluster.hierarchy.linkage(d,metric='euclidean',method='complete',optimal_ordering=True)
        dn = scipy.cluster.hierarchy.dendrogram(l,no_plot=True)
        order = dn['leaves']
        
        effect_size = effect_size.iloc[order,:]
        pval = pval.iloc[order,:]
        
        if size_max is None:
            size_max = np.amax(pval.to_numpy())

        ref_s = [1.30,size_max]
        ref_l = ['0.05','maxsize: '+'{:.1e}'.format(10**(-1*size_max))]


        fig,axs = plt.subplots(nrows=1,ncols=2,figsize=figsize)
        
        gf.heatmap2(effect_size,cmap='RdBu_r',vmin=vmin,vmax=vmax,cellsize=pval,square=True,cellsize_vmax=size_max,ref_sizes=ref_s,ref_labels=ref_l,fontsize=fontsize,figsize=figsize,ax=axs[0])

        icoord = np.array(dn['icoord'] )
        dcoord = np.array(dn['dcoord'] )

        for xs, ys in zip(dcoord, icoord):
            _= axs[1].plot(xs, ys, color='k', linewidth=0.5)

        axs[1].grid(b=False)
        axs[1].set(xticks=[],yticks=[])

        return effect_size,pval,fig,axs

def tx_dgex_per_patient(adata,groupby='leiden',rep='cohort',thresh=10,use_raw=False,method='wilcoxon'):

    st = adata.obs[groupby].unique().tolist()
    pt = adata.obs[rep].unique().tolist()
    tx = ['Base','PD1','RTPD1']
    combs = list(itertools.combinations(tx,2))
    cellthresh = thresh

    for i,s in enumerate(st):
        for j,c in enumerate(combs):
            pt2 = [x for x in pt if (len(adata[(adata.obs[rep]==x)&(adata.obs.treatment==c[0])&(adata.obs[groupby]==s)])>=cellthresh)
                & (len(adata[(adata.obs[rep]==x)&(adata.obs.treatment==c[1])&(adata.obs[groupby]==s)])>=cellthresh)]
            len(pt2)
            
            for k,p in enumerate(pt2):
                tmp = adata[(adata.obs[rep]==p)&(adata.obs[groupby]==s)]
                sc.tl.rank_genes_groups(tmp,groupby='treatment',groups=[c[1]],reference=c[0],use_raw=use_raw,method=method)
                tmp2 = sc.get.rank_genes_groups_df(tmp,group=c[1])
                tmp2[groupby] = s
                tmp2[rep] = p
                tmp2['group'] = c[1]+'v'+c[0]
                if ((i==0)&(j==0)&(k==0)):
                    df = tmp2
                else:
                    df = df.append(tmp2)
        print(s)
    
    return df


def tx_dgex_per_patient_heatmap(dgex,clust=None,consistency_cutoff=0.5,pval_cutoff=0.25,use_fdr=False,fc_cutoff=0,response='R',hormone='TNBC',ref_s=None,ref_l=None,fontsize=4,figsize=None):
    
    tx = ['Base','PD1','RTPD1']
    combs = list(itertools.combinations(tx,2))

    metadata = pd.read_csv('../PEMBRORT_CLINICAL_METADATA_FORSCSEQ_KHG20201119.csv',index_col=None,header=0)
    dgex = dgex.merge(metadata,how='left',left_on='cohort',right_on='Patient_Number')
    dgex = dgex[(dgex.pCR==response)&(dgex.hormone_receptor==hormone)]

    if figsize is None:
        figsize = (4,4)

    fig,axs = plt.subplots(nrows=1,ncols=2,figsize=figsize)

    direction = ['up','down']

    for d,ax in zip(direction,axs.flat):

        # filter genes
        finalGOI = []

        for i,c in enumerate(combs):
            tmp = dgex[(dgex.leiden==clust)&(dgex.group==(c[1]+'v'+c[0]))]
            numpt = len(tmp.cohort.unique().tolist())
            print(numpt)
            if (d=='up'):
                if use_fdr:
                    tmp = tmp[(tmp.pvals_adj<=pval_cutoff)&(tmp.logfoldchanges>fc_cutoff)]
                else:
                    tmp = tmp[(tmp.pvals<=pval_cutoff)&(tmp.logfoldchanges>fc_cutoff)]
            else:
                if use_fdr:
                    tmp = tmp[(tmp.pvals_adj<=pval_cutoff)&(tmp.logfoldchanges<fc_cutoff)]
                else:
                    tmp = tmp[(tmp.pvals<=pval_cutoff)&(tmp.logfoldchanges<fc_cutoff)]
                    
            GOI = tmp.names.value_counts()
            GOI = GOI[GOI>int(np.ceil(consistency_cutoff*numpt))].index.tolist()
            finalGOI = finalGOI + GOI
            
        finalGOI = np.unique(finalGOI)
        print(len(finalGOI))

        # make tables
        fc_df = pd.DataFrame(index=finalGOI,columns=pd.MultiIndex.from_tuples(combs),dtype=np.float64)
        pc_df = pd.DataFrame(index=finalGOI,columns=pd.MultiIndex.from_tuples(combs),dtype=np.float64)        

        for i,g in enumerate(fc_df.index):
            for j,c in enumerate(fc_df.columns):
                tmp = dgex[(dgex.leiden==clust)&(dgex.group==(c[1]+'v'+c[0]))]
                numpt = len(tmp.cohort.unique().tolist())

                if (numpt==0):
                    continue

                if (d=='up'):
                    if use_fdr:
                        tmp = tmp[(tmp.pvals_adj<=pval_cutoff)&(tmp.logfoldchanges>fc_cutoff)&(tmp.names==g)]
                    else:
                        tmp = tmp[(tmp.pvals<=pval_cutoff)&(tmp.logfoldchanges>fc_cutoff)&(tmp.names==g)]
                else:
                    if use_fdr:
                        tmp = tmp[(tmp.pvals_adj<=pval_cutoff)&(tmp.logfoldchanges<fc_cutoff)&(tmp.names==g)]
                    else:
                        tmp = tmp[(tmp.pvals<=pval_cutoff)&(tmp.logfoldchanges<fc_cutoff)&(tmp.names==g)]
                        
                fc_df.loc[g,c] = np.nanmean(tmp.logfoldchanges)
                pc_df.loc[g,c] = len(tmp.cohort.unique().tolist())/numpt

        fc_df.columns = [x[1]+'vs'+x[0] for x in fc_df.columns]
        pc_df.columns = [x[1]+'vs'+x[0] for x in pc_df.columns]
        
        fc_df.fillna(0,inplace=True)
        pc_df.fillna(0,inplace=True)

        if (d=='up'):
            tmp = fc_df
            cmap = 'Reds'
        else:
            tmp = -1*fc_df
            cmap = 'Blues'

        if (len(tmp)>0):
            vmax = np.quantile(tmp.to_numpy(dtype=np.float64),q=0.95)

            gf.heatmap2(data=tmp,vmin=0,vmax=vmax,cmap=cmap,
                        cellsize=pc_df,cellsize_vmax=1,
                        ref_sizes=ref_s,ref_labels=ref_l,
                        ax=ax,fontsize=fontsize,figsize=(figsize[0],int(figsize[1]/2)))
        
    return fig,axs,fc_df,pc_df

def pathway_enrich(rnk,genesets=None,test='enrich',org='Human',background=None,no_plot=True,outdir=None,ascending=False):

    """
    test can be either 'enrich' or 'gsea'.
    'enrich' expects just a list of genes, and provide the appropriate background gene list.
    'gsea' expects a dataframe with genes on the index and one column containing a ranking metric (specify ascending order).
    if you want to run a mouse dataset, you need to convert the gene names to human.

    """

    if genesets is None:
        
        genesets = ['enrichr.GO_Biological_Process_2018.gmt',
                    'enrichr.KEGG_2016.gmt',
                    'enrichr.Reactome_2016.gmt']

    if (test=='enrich'):

        enr_results = gp.enrichr(gene_list=rnk,
                                gene_sets=genesets,
                                organism=org, 
                                description='test_name',
                                background=background,
                                outdir=outdir,
                                no_plot=no_plot,
                                cutoff=1)

        enr_results = enr_results.results

    elif (test=='gsea'):

        enr_results = pd.DataFrame(columns=['es','nes','pval','fdr','geneset_size','matched_size','genes','ledge_genes','group'])

        for g in genesets:
            pre_res = gp.prerank(rnk=rnk,
                                gene_sets=g, 
                                processes=4,
                                permutation_num=100,
                                ascending=ascending,
                                outdir=outdir, 
                                format='png', 
                                seed=6,
                                no_plot=no_plot,
                                min_size=0,
                                max_size=500,
                                verbose=True)

            pre_res = pre_res.res2d
            pre_res['group'] = g
            enr_results = enr_results.append(pre_res)

    return enr_results

def filter_pathways(enr_results,significance='fdr',pval_cutoff=0.05,direction=None,test='enrich',add_blacklist=None,term_overlap=0.2,gene_overlap=0.5,fontsize=2,figsize=None,ax=None):

    blacklist = ['Homo','sapiens','Immune','immune','Cell','Cellular','cell','cellular',
                 'Pathway','pathway','Response','response','to','of','in']
    
    if add_blacklist is not None:
        blacklist = blacklist + add_blacklist

    if (test=='enrich'):

        enr_results['overlapsize'] = [len(x.split(';')) for x in enr_results.Genes]
        enr_results['setsize'] = [int(x.split('/')[1]) for x in enr_results.Overlap]
        enr_results['fraction'] = enr_results['overlapsize']/enr_results['setsize']
        enr_results.index = enr_results.Term

        gene_col = 'Genes'

        if significance=='fdr':
            significance = 'Adjusted P-value'
        else:
            significance = 'P-value'

        enr_results = enr_results[(enr_results[significance]<=pval_cutoff)]

    elif (test=='GSEA'):

        enr_results['fraction'] = enr_results['matched_size']/enr_results['geneset_size']

        gene_col = 'genes'

        if (direction=='up'):
            enr_results = enr_results[(enr_results['nes']>0)&(enr_results[significance]<=pval_cutoff)]
        elif (direction=='down'):
            enr_results = enr_results[(enr_results['nes']<0)&(enr_results[significance]<=pval_cutoff)]

    tmp = enr_results

    tmp.sort_values(by=significance,inplace=True)


    terms = []

    for i,t in enumerate(tmp.index):

        if (i==0):
            terms = terms + [t]

        else:

            test = t.split(' ')
            test = [x for x in test if x not in blacklist]
            test = [x for x in test if 'R-HSA' not in x and 'GO:' not in x and 'hsa' not in x]

            counter = 0

            for j in terms:
                reference = j.split(' ')
                match = [x for x in reference if x in test]
                if (len(match) > term_overlap*len(test)):
                    counter = 1

            if (counter==0):
                terms = terms + [t]

    tmp = tmp.loc[terms,:]

    tmp.sort_values(by='fraction',ascending=False,inplace=True)
    genes = []
    terms = []
    for i,t in enumerate(tmp.index):

        if (i==0):
            genes = genes + [tmp.loc[t,gene_col]]
            terms = terms + [t]
        else:

            test = tmp.loc[t,gene_col].split(';')

            counter = 0

            for j in genes:
                reference = j.split(';')
                match = [x for x in reference if x in test]
                if (len(match) > gene_overlap*len(test)):
                    counter = 1

            if (counter==0):
                genes = genes + [tmp.loc[t,gene_col]]
                terms = terms + [t]

    tmp = tmp.loc[terms,:]
    tmp.sort_values(by=significance,ascending=False,inplace=True)


    df = pd.DataFrame(-1*np.log10(tmp[significance].to_numpy(dtype=np.float64)),index=tmp.index.tolist(),columns=[significance])
    df.loc[np.isinf(df[significance]),significance] = np.max(df.loc[~np.isinf(df[significance]),significance])

    if ax is None:
        if figsize is None:
            figsize=(2,2)
        fig,ax = plt.subplots(figsize=figsize)

    ys = np.arange(len(df))
    ws = df.to_numpy(dtype=np.float64).flatten()
    _= ax.barh(y=ys,width=ws,height=0.5,tick_label=df.index.tolist())
    _= ax.grid(b=False)
    _= ax.set_yticklabels(labels=df.index.tolist(),fontdict={'fontsize':fontsize})
    _= ax.set_xlabel('-log10(pval)')

    return ax,tmp