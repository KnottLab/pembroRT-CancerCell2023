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
import sklearn
from statannot import add_stat_annotation
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.legend import Legend
import matplotlib.gridspec as gridspec

# from __future__ import division

from matplotlib.collections import LineCollection
import matplotlib.patheffects as patheffects

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

from seaborn import cm
from seaborn.axisgrid import Grid
from seaborn.utils import (despine, axis_ticklabels_overlap, relative_luminance, to_utf8)
from seaborn.axisgrid import Grid
from seaborn.utils import (despine, axis_ticklabels_overlap, relative_luminance, to_utf8)

sc.set_figure_params(scanpy=True, dpi=300, dpi_save=300, frameon=True, vector_friendly=True, fontsize=12, 
                         color_map='Dark2', format='pdf', transparent=True, ipython_format='png2x')

rcParams.update({'font.size': 8})
rcParams.update({'font.family': 'Helvetica'})
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['svg.fonttype'] = 'none'
rcParams['figure.facecolor'] = (1,1,1,1)


def radar_factory(num_vars, frame='circle', fontsize=4):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels, fontsize=fontsize)
            
        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def diff_radarplot(adata, embedding='X_diffmap', use_obs=False, comps=None, groupby='leiden', foldchange=False, clip=None,ref=None, q=None, filled=True, figsize=None, fontsize=4, ylim=None):
       
    ## collect data
    ncomp = len(comps)
    st = sorted(adata.obs[groupby].unique().tolist())
    
    if use_obs:
        cols = comps
        tmp = adata.obs.loc[:,cols].to_numpy()
    else:
        cols = ['X_'+str(x) for x in comps]
        tmp = adata.obsm[embedding][:,comps]
    
    
    
    if foldchange:

        tmp = pd.DataFrame(tmp,index=adata.obs.index,columns=cols)
        tmp = tmp.merge(adata.obs,left_index=True,right_index=True)

        tmp2 = pd.DataFrame(index=st,columns=cols)
        
        for idx in tmp2.index:
            grp_median = tmp[(tmp[groupby]==idx)][cols].median()
            
            if ref is None:
                ref_median = tmp[(tmp[groupby]!=idx)][cols].median()
            else:
                ref_median = tmp[(tmp[groupby]==ref)][cols].median()
            
            # tmp2.loc[idx,:] = (grp_median-ref_median)/np.abs(ref_median)
            # tmp2.loc[idx,:] = (grp_median-ref_median)/np.ptp(tmp[cols])
            tmp2.loc[idx,:] = (grp_median-ref_median)/np.abs(ref_median)

        if clip is not None:
            tmp2.clip(lower=-1*clip,upper=clip,inplace=True)

    else:
        
        tmp = sklearn.preprocessing.minmax_scale(tmp, feature_range=(0,1), axis=0, copy=True)
        tmp = pd.DataFrame(tmp,index=adata.obs.index,columns=cols)
        tmp = tmp.merge(adata.obs,left_index=True,right_index=True)

        if q is None:
            tmp2 = tmp.groupby([groupby],axis=0).mean()[cols]
        else:
            tmp2 = tmp.groupby([groupby],axis=0)[cols].quantile(q=q)

    
    ## plot
    theta = radar_factory(ncomp, frame='polygon', fontsize=fontsize)
    
    r = np.linspace(np.min(tmp2.to_numpy().flatten()),np.max(tmp2.to_numpy().flatten()),num=5)
    r = np.sort(np.append(r,0))
    
    colors = {s:adata.uns['leiden_colors'][i] for i,s in enumerate(st)}

    if figsize is None:
        figsize = (2,len(st))

    fig,axs = plt.subplots(figsize=figsize, nrows=len(st), ncols=1, subplot_kw=dict(projection='radar'))
    
    for c,ax in zip(colors, axs.flat):
        
        d = tmp2.loc[c,:].to_numpy()

        ax.set_rgrids(r+[0], fontsize=fontsize)
        ax.set_title(c, fontsize=fontsize, pad=0.1)

        _= ax.plot(theta, d, color=colors[c], linewidth=0.5)

        if filled:
            ax.fill(theta, d, facecolor=colors[c], alpha=0.25)

        ax.set_varlabels(cols)
        ax.tick_params(pad=0.0001)

        if ylim is None:
            ax.set_ylim(r[0]+0.2*r[0],r[-1]+0.2*r[-1])
        else:
            ax.set_ylim(ylim)
                
    return fig,axs

def diff_radarplot_response(adata, embedding='X_diffmap', use_obs=False, comps=None, byCluster=True, q=None, filled=True, figsize=None, fontsize=4, ylim=None):
       
    ## collect data
    ncomp = len(comps)
    
    if use_obs:
        cols = comps
        tmp = adata.obs.loc[:,cols].to_numpy()

    else:
        cols = ['X_'+str(x) for x in comps]
        tmp = adata.obsm[embedding][:,comps]
        
    tmp = sklearn.preprocessing.minmax_scale(tmp, feature_range=(0,1), axis=0, copy=True)
    tmp = pd.DataFrame(tmp,index=adata.obs.index,columns=cols)
    tmp = tmp.merge(adata.obs,left_index=True,right_index=True)
    
    if byCluster:
        if q is None:
            tmp = tmp.groupby(['pCR','treatment','leiden'],axis=0).mean()[cols]
        else:
            tmp = tmp.groupby(['pCR','treatment','leiden'],axis=0)[cols].quantile(q=q)
        grps = sorted(adata.obs.leiden.unique().tolist())
    else:
        if q is None:
            tmp = tmp.groupby(['pCR','treatment'],axis=0).median()[cols]
        else:
            tmp = tmp.groupby(['pCR','treatment'],axis=0)[cols].quantile(q=q)
    
    
    ## plot
    theta = radar_factory(ncomp, frame='polygon', fontsize=fontsize)

    r = np.linspace(np.min(tmp.to_numpy().flatten()),np.max(tmp.to_numpy().flatten()),num=5)
    r = np.sort(np.append(r,0))

    if figsize is None:
        figsize = (4,8)
    if byCluster:
        fig, axs = plt.subplots(figsize=figsize, nrows=len(grps), ncols=3, subplot_kw=dict(projection='radar'))
    else:
        fig, axs = plt.subplots(figsize=figsize, nrows=1, ncols=3, subplot_kw=dict(projection='radar'))
    
    colors = {'R':'r',
              'NR':'b'}
    
    tx = ['Base','PD1','RTPD1']
    
    if byCluster:
        
        forder = [(x,y) for x in grps for y in tx]
        for f,ax in zip(forder, axs.flat):

            ax.set_rgrids(r, fontsize=fontsize)
            ax.set_title(str(f), fontsize=fontsize, pad=0.0001)

            for c in colors:

                d = tmp.loc[(c,f[1],f[0]),:].to_numpy()
                _= ax.plot(theta, d, color=colors[c], linewidth=0.5)

                if filled:
                    ax.fill(theta, d, facecolor=colors[c], alpha=0.25)

            ax.set_varlabels(cols)
            ax.tick_params(pad=0.0001)

            if ylim is None:
                ax.set_ylim(r[0]-0.2,r[-1]+0.2)
            else:
                ax.set_ylim(ylim)
            
    else:
        
        forder = tx
        for f,ax in zip(forder, axs.flat):

            ax.set_rgrids([0.2, 0.4, 0.6, 0.8], fontsize=fontsize)
            ax.set_title(f, fontsize=fontsize, pad=0.1)

            for c in colors:

                d = tmp.loc[(c,f),:].to_numpy()
                _= ax.plot(theta, d, color=colors[c], linewidth=0.5)

                if filled:
                    ax.fill(theta, d, facecolor=colors[c], alpha=0.25)

            ax.set_varlabels(cols)
            ax.tick_params(pad=0.0001)
            
            if ylim is None:
                ax.set_ylim(r[0]+0.2*r[0],r[-1]+0.2*r[-1])
            else:
                ax.set_ylim(ylim)
                
    return fig,axs



def _index_to_label(index):
    """Convert a pandas index or multiindex to an axis label."""
    if isinstance(index, pd.MultiIndex):
        return "-".join(map(to_utf8, index.names))
    else:
        return index.name

def _index_to_ticklabels(index):
    """Convert a pandas index or multiindex into ticklabels."""
    if isinstance(index, pd.MultiIndex):
        return ["-".join(map(to_utf8, i)) for i in index.values]
    else:
        return index.values

def _matrix_mask(data, mask):
    """Ensure that data and mask are compatabile and add missing values.

    Values will be plotted for cells where ``mask`` is ``False``.

    ``data`` is expected to be a DataFrame; ``mask`` can be an array or
    a DataFrame.

    """
    if mask is None:
        mask = np.zeros(data.shape, bool)

    if isinstance(mask, np.ndarray):
        # For array masks, ensure that shape matches data then convert
        if mask.shape != data.shape:
            raise ValueError("Mask must have the same shape as data.")

        mask = pd.DataFrame(mask,
                            index=data.index,
                            columns=data.columns,
                            dtype=bool)

    elif isinstance(mask, pd.DataFrame):
        # For DataFrame masks, ensure that semantic labels match data
        if not mask.index.equals(data.index) \
           and mask.columns.equals(data.columns):
            err = "Mask must have the same index and columns as data."
            raise ValueError(err)

    # Add any cells with missing data to the mask
    # This works around an issue where `plt.pcolormesh` doesn't represent
    # missing data properly
    mask = mask | pd.isnull(data)

    return mask


class _HeatMapper2(object):
    """Draw a heatmap plot of a matrix with nice labels and colormaps."""

    def __init__(self, data, vmin, vmax, cmap, center, robust, annot, fmt,
                 annot_kws, cellsize, cellsize_vmax,
                 cbar, cbar_kws,
                 xticklabels=True, yticklabels=True, mask=None, ax_kws=None, rect_kws=None, fontsize=4):
        """Initialize the plotting object."""
        # We always want to have a DataFrame with semantic information
        # and an ndarray to pass to matplotlib
        if isinstance(data, pd.DataFrame):
            plot_data = data.values
        else:
            plot_data = np.asarray(data)
            data = pd.DataFrame(plot_data)

        # Validate the mask and convet to DataFrame
        mask = _matrix_mask(data, mask)

        plot_data = np.ma.masked_where(np.asarray(mask), plot_data)

        # Get good names for the rows and columns
        xtickevery = 1
        if isinstance(xticklabels, int):
            xtickevery = xticklabels
            xticklabels = _index_to_ticklabels(data.columns)
        elif xticklabels is True:
            xticklabels = _index_to_ticklabels(data.columns)
        elif xticklabels is False:
            xticklabels = []

        ytickevery = 1
        if isinstance(yticklabels, int):
            ytickevery = yticklabels
            yticklabels = _index_to_ticklabels(data.index)
        elif yticklabels is True:
            yticklabels = _index_to_ticklabels(data.index)
        elif yticklabels is False:
            yticklabels = []

        # Get the positions and used label for the ticks
        nx, ny = data.T.shape

        if not len(xticklabels):
            self.xticks = []
            self.xticklabels = []
        elif isinstance(xticklabels, str) and xticklabels == "auto":
            self.xticks = "auto"
            self.xticklabels = _index_to_ticklabels(data.columns)
        else:
            self.xticks, self.xticklabels = self._skip_ticks(xticklabels,
                                                             xtickevery)

        if not len(yticklabels):
            self.yticks = []
            self.yticklabels = []
        elif isinstance(yticklabels, str) and yticklabels == "auto":
            self.yticks = "auto"
            self.yticklabels = _index_to_ticklabels(data.index)
        else:
            self.yticks, self.yticklabels = self._skip_ticks(yticklabels,
                                                             ytickevery)

        # Get good names for the axis labels
        xlabel = _index_to_label(data.columns)
        ylabel = _index_to_label(data.index)
        self.xlabel = xlabel if xlabel is not None else ""
        self.ylabel = ylabel if ylabel is not None else ""

        # Determine good default values for the colormapping
        self._determine_cmap_params(plot_data, vmin, vmax,
                                    cmap, center, robust)

        # Determine good default values for cell size
        self._determine_cellsize_params(plot_data, cellsize, cellsize_vmax)

        # Sort out the annotations
        if annot is None:
            annot = False
            annot_data = None
        elif isinstance(annot, bool):
            if annot:
                annot_data = plot_data
            else:
                annot_data = None
        else:
            try:
                annot_data = annot.values
            except AttributeError:
                annot_data = annot
            if annot.shape != plot_data.shape:
                raise ValueError('Data supplied to "annot" must be the same '
                                 'shape as the data to plot.')
            annot = True

        # Save other attributes to the object
        self.data = data
        self.plot_data = plot_data

        self.annot = annot
        self.annot_data = annot_data

        self.fmt = fmt
        self.annot_kws = {} if annot_kws is None else annot_kws
        self.annot_kws.setdefault('color', "black")
        self.annot_kws.setdefault('ha', "center")
        self.annot_kws.setdefault('va', "center")
        self.annot_kws.setdefault('fontsize', fontsize)
        self.cbar = cbar
        self.cbar_kws = {} if cbar_kws is None else cbar_kws
        self.cbar_kws.setdefault('ticks', matplotlib.ticker.MaxNLocator(6))
        self.ax_kws = {} if ax_kws is None else ax_kws
        self.rect_kws = {} if rect_kws is None else rect_kws
        # self.rect_kws.setdefault('edgecolor', "black")

    def _determine_cmap_params(self, plot_data, vmin, vmax,
                               cmap, center, robust):
        """Use some heuristics to set good defaults for colorbar and range."""
        calc_data = plot_data.data[~np.isnan(plot_data.data)]
        if vmin is None:
            vmin = np.percentile(calc_data, 2) if robust else calc_data.min()
        if vmax is None:
            vmax = np.percentile(calc_data, 98) if robust else calc_data.max()
        self.vmin, self.vmax = vmin, vmax

        # Choose default colormaps if not provided
        if cmap is None:
            if center is None:
                self.cmap = cm.rocket
            else:
                self.cmap = cm.icefire
        elif isinstance(cmap, str):
            self.cmap = matplotlib.cm.get_cmap(cmap)
        elif isinstance(cmap, list):
            self.cmap = matplotlib.colors.ListedColormap(cmap)
        else:
            self.cmap = cmap

        # Recenter a divergent colormap
        if center is not None:
            vrange = max(vmax - center, center - vmin)
            normlize = matplotlib.colors.Normalize(center - vrange, center + vrange)
            cmin, cmax = normlize([vmin, vmax])
            cc = np.linspace(cmin, cmax, 256)
            self.cmap = matplotlib.colors.ListedColormap(self.cmap(cc))

    def _determine_cellsize_params(self, plot_data, cellsize, cellsize_vmax):

        if cellsize is None:
            self.cellsize = np.ones(plot_data.shape)
            self.cellsize_vmax = 1.0
        else:
            if isinstance(cellsize, pd.DataFrame):
                cellsize = cellsize.values
            self.cellsize = cellsize
            if cellsize_vmax is None:
                cellsize_vmax = cellsize.max()
            self.cellsize_vmax = cellsize_vmax

    def _skip_ticks(self, labels, tickevery):
        """Return ticks and labels at evenly spaced intervals."""
        n = len(labels)
        if tickevery == 0:
            ticks, labels = [], []
        elif tickevery == 1:
            ticks, labels = np.arange(n) + .5, labels
        else:
            start, end, step = 0, n, tickevery
            ticks = np.arange(start, end, step) + .5
            labels = labels[start:end:step]
        return ticks, labels

    def _auto_ticks(self, ax, labels, axis, fontsize):
        """Determine ticks and ticklabels that minimize overlap."""
        transform = ax.figure.dpi_scale_trans.inverted()
        bbox = ax.get_window_extent().transformed(transform)
        size = [bbox.width, bbox.height][axis]
        axis = [ax.xaxis, ax.yaxis][axis]
        tick, = axis.set_ticks([0])
        max_ticks = int(size // (fontsize / 72))
        if max_ticks < 1:
            return [], []
        tick_every = len(labels) // max_ticks + 1
        tick_every = 1 if tick_every == 0 else tick_every
        ticks, labels = self._skip_ticks(labels, tick_every)
        return ticks, labels

    def plot(self, ax, cax, fontsize, rowcolors=None, colcolors=None, ref_sizes=None, ref_labels=None):
        """Draw the heatmap on the provided Axes."""

        # Remove all the Axes spines
        despine(ax=ax, left=True, bottom=True)

        # Draw the heatmap and annotate
        height, width = self.plot_data.shape
        xpos, ypos = np.meshgrid(np.arange(width) + .5, np.arange(height) + .5)

        data = self.plot_data.data
        cellsize = self.cellsize

        mask = self.plot_data.mask
        if not isinstance(mask, np.ndarray) and not mask:
            mask = np.zeros(self.plot_data.shape, bool)

        annot_data = self.annot_data
        if not self.annot:
            annot_data = np.zeros(self.plot_data.shape)

        # Draw rectangles instead of using pcolormesh
        # Might be slower than original heatmap
        for x, y, m, val, s, an_val in zip(xpos.flat, ypos.flat, mask.flat, data.flat, cellsize.flat, annot_data.flat):
            if not m:
                vv = (val - self.vmin) / (self.vmax - self.vmin)
                size = np.clip(s / self.cellsize_vmax, 0.1, 1.0)
                color = self.cmap(vv)
                rect = plt.Rectangle([x - size / 2, y - size / 2], size, size, facecolor=color, label=None, **self.rect_kws)
                ax.add_patch(rect)

                if self.annot:
                    annotation = ("{:" + self.fmt + "}").format(an_val)
                    text = ax.text(x, y, annotation, **self.annot_kws)
                    # add edge to text
                    text_luminance = relative_luminance(text.get_color())
                    text_edge_color = ".15" if text_luminance > .408 else "w"
                    text.set_path_effects([matplotlib.patheffects.withStroke(linewidth=1, foreground=text_edge_color)])
        
        ## Draw rectangles for size scale using specific reference sizes

        if ref_sizes is not None:

            # ref_s = [1.30,2.00,3.00,5.00,10.00,self.cellsize_vmax]
            # ref_l = ['0.05','0.01','1e-3','1e-5','1e-10','maxsize: '+'{:.1e}'.format(10**(-1*self.cellsize_vmax))]

            ref_s = ref_sizes + [self.cellsize_vmax]
            ref_l = ref_labels + ['maxsize']
            ref_x = -10*np.ones(len(ref_s))
            ref_y = np.arange(len(ref_s))
            
            for x,y,s,l in zip(ref_x,ref_y,ref_s,ref_l):
                size = np.clip(s / self.cellsize_vmax, 0.1, 1.0)
                print(f"{x}-{y}-{size}-{l}")
                rect = plt.Rectangle([x - size / 2, y - size / 2], size, size, facecolor='k', label=l, **self.rect_kws)
                ax.add_patch(rect)
                ax.text(x, y, l, **self.annot_kws)
        
        ## Draw rectangles to provide a row color annotation 
        if rowcolors is not None:
            for i,r in enumerate(rowcolors):
                for x,y,c in zip(xpos[:,0]-(15+i),ypos[:,0],r):
                    size = 1
                    rect = plt.Rectangle([x - size / 2, y - size / 2], size, size, facecolor=c, label=None, linewidth=0, edgecolor=None, **self.rect_kws)
                    ax.add_patch(rect)
            
        ## Draw rectangles to provide a column color annotation  
        if colcolors is not None:
            for i,c in enumerate(colcolors):
                for x,y,c in zip(xpos[0,:],ypos[0,:]-(10+i),c):
                    size = 1
                    rect = plt.Rectangle([x - size / 2, y - size / 2], size, size, facecolor=c, label=None, linewidth=0, edgecolor=None, **self.rect_kws)
                    ax.add_patch(rect)

        # Set the axis limits
        ax.set(xlim=(0, self.data.shape[1]), ylim=(0, self.data.shape[0]))

        # Set other attributes
        ax.set(**self.ax_kws)

        if self.cbar:
            norm = matplotlib.colors.Normalize(vmin=self.vmin, vmax=self.vmax)
            scalar_mappable = matplotlib.cm.ScalarMappable(cmap=self.cmap, norm=norm)
            scalar_mappable.set_array(self.plot_data.data)
            cb = ax.figure.colorbar(scalar_mappable, cax, ax, **self.cbar_kws)
            cb.outline.set_linewidth(0)
            cb.ax.tick_params(labelsize=fontsize) 

        # Add row and column labels
        if isinstance(self.xticks, str) and self.xticks == "auto":
            xticks, xticklabels = self._auto_ticks(ax, self.xticklabels, axis=0, fontsize=fontsize)
        else:
            xticks, xticklabels = self.xticks, self.xticklabels

        if isinstance(self.yticks, str) and self.yticks == "auto":
            yticks, yticklabels = self._auto_ticks(ax, self.yticklabels, axis=1, fontsize=fontsize)
        else:
            yticks, yticklabels = self.yticks, self.yticklabels

        ax.set(xticks=xticks, yticks=yticks)
        xtl = ax.set_xticklabels(xticklabels, fontsize=fontsize)
        ytl = ax.set_yticklabels(yticklabels, rotation="vertical", fontsize=fontsize)

        # Possibly rotate them if they overlap
        ax.figure.draw(ax.figure.canvas.get_renderer())
        if axis_ticklabels_overlap(xtl):
            plt.setp(xtl, rotation="vertical")
        if axis_ticklabels_overlap(ytl):
            plt.setp(ytl, rotation="horizontal")

        # Add the axis labels
        ax.set(xlabel=self.xlabel, ylabel=self.ylabel)

        # Invert the y axis to show the plot in matrix form
        ax.invert_yaxis()


def heatmap2(data, vmin=None, vmax=None, cmap=None, center=None, robust=False,
            annot=None, fmt=".2g", annot_kws=None,
            cellsize=None, cellsize_vmax=None,
            ref_sizes=None, ref_labels=None,
            cbar=True, cbar_kws=None, cbar_ax=None,
            square=True, xticklabels="auto", yticklabels="auto",rowcolors=None,colcolors=None,
            mask=None, ax=None, ax_kws=None, rect_kws=None, fontsize=4, figsize=(2,2)):

    # Initialize the plotter object
    plotter = _HeatMapper2(data, vmin, vmax, cmap, center, robust,
                          annot, fmt, annot_kws,
                          cellsize, cellsize_vmax,
                          cbar, cbar_kws, xticklabels,
                          yticklabels, mask, ax_kws, rect_kws, fontsize)

    # Draw the plot and return the Axes
    if ax is None:
        fig,ax = plt.subplots(figsize=figsize, facecolor=(0,0,0,0), alpha=0)
    if square:
        ax.set_aspect("equal")

    # delete grid
    ax.grid(False)
    
    plotter.plot(ax, cbar_ax, fontsize=fontsize, rowcolors=rowcolors, colcolors=colcolors, ref_sizes=ref_sizes, ref_labels=ref_labels)
      
    return ax



def umap_density(adata=None,df=None,embedding='X_umap',t=0.2,lv=5,gsize=200,alpha=0.4,colors=None,groupby=None,hue=None,fill=True,include_scatter=True,dotsize=0.005,figsize=None):

    if adata is not None:
        df = pd.DataFrame(adata.obsm[embedding],columns=['emb1','emb2'],index=adata.obs.index)
        df = df.merge(adata.obs,left_index=True,right_index=True)

    st = df[groupby].unique().tolist()

    if colors is None:
        
        if (len(st)<=10):
            test = plt.get_cmap('tab10')
        elif (len(st)<=20):
            test = plt.get_cmap('tab20')
        else:
            print('please provide colors because more than 20 categories')
            return

        colors = {g:test(i) for i,g in enumerate(st)}

    sns.set_style("white", rc={"font.family":"Helvetica","axes.grid":False})                                                  
    sns.set_context("paper", rc={"font.size":4,"axes.titlesize":4,"axes.labelsize":4,"font.family":"Helvetica","xtick.labelsize":4,"ytick.labelsize":4})

    if figsize is None:
        figsize = (len(st)*2,2)
        
    fig,axs = plt.subplots(nrows=1,ncols=len(st),sharex=True,sharey=True,figsize=figsize)

    for s,ax in zip(st,axs.flat):
        
        tmp = df.loc[(df[groupby]==s),:]
        
        if hue is None:
            _= sns.kdeplot(x='emb1',y='emb2',data=tmp,fill=fill,color=colors[s],thresh=t,alpha=alpha,levels=lv,gridsize=gsize,ax=ax,legend=False)
        else:
            _= sns.kdeplot(x='emb1',y='emb2',data=tmp,hue=hue,fill=fill,palette=colors,thresh=t,alpha=alpha,levels=lv,gridsize=gsize,ax=ax,legend=True)

        if include_scatter:
            _= ax.scatter(x=df['emb1'],y=df['emb2'],c='tab:gray',s=dotsize,linewidths=0)
        
        _= ax.set_title(s,pad=1,fontsize=4)
        _= ax.grid(b=False)

    return fig,axs


