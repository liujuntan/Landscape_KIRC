#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python wrapper functions for MuTrans

@author: cliffzhou
"""

import matlab.engine
import os, sys
from pathlib import Path

eng = matlab.engine.start_matlab()
sys.path.append(os.path.join(Path(os.getcwd()).parent, "MuTrans"))
eng.eval("addpath(genpath('C:/Users/27226/Desktop/科研论文/类脑/MuTrans-release-main/MuTrans/DynamicalAnalysis/'))")
#C:\Users\27226\Desktop\科研论文\类脑\MuTrans-release-main\MuTrans\DynamicalAnalysis
import numpy as np
import pandas as pd
import pyemma.msm as msm
import pyemma.plots as mpl
import matplotlib.pyplot as plt
import scipy
import networks as nw
import seaborn as sns
from tqdm import tqdm
import scanpy as sc
from matplotlib.colors import ListedColormap

def dynamical_analysis(sc_object,par):
    """ MuTrans dynamical system analysis by multi-scale reduction.
    
    Arguments: 
        
        sc_object: the anndata object storing single-cell datasets, including the expression matrix and the low-dimensional coordinates (umap or tsne) 
        
        par: parameters in MuTrans, stored in dictionary, with important keys:
        
        ### must provide:
        "K_cluster": the number of clusters/attractors. The eigen-peak can serve as the reference, and it's suggested to do Louvain clustering or other biological analysis for the prior knowledge.
        
        ### optional with defaults
        
        [parameters in construcing cell-cell random walk]
        "perplex": the parameter controlling the "diffusiveness" of cell-cell random walk, similar to the parameter in tSNE. Control the local bandwith of Gaussian kernels. Larger perplex represents larger bandwith and more connectiveness between cells, and smaller perplex may reduce "shortcuts" in lineage analysis. Default is N/2.5. Suggest to change from N/5 to N/2.
        "choice_distance": the choice of distance when constructing the Gaussian kernel. Can be "euclid"(default) or "cosine". This is similar to the "metric" parameter in UMAP.
        "weight_scale": When construcing the cell-cell weights by Gaussian kernels, whether adopting the strategy in tSNE to scale the Gaussian kernels by row sums. Default is "True" and suggested for large-scale 10X data to stabilize. 
        "alpha_scale": When construcing the cell-cell weights by Gaussian kernels, the power row sums to scale the Gaussian kernels by adopting the strategy in diffusion map. Default is 0.
        
        [parameters in multiscale reduction]
        "reduce_large_scale": whether to boost the analysis by the DECLARE dynamical-preserving pre-processing step. Default is False, recommended True for large-scale datasets with more than 5000 cells or more than 10 attractors, whenever the default analysis is slow.
        "reduce_num_meta_cell": number of meta-cells or microscopic states in the DECLARE reduction. Only valid when "reduce_large_scale" is True.
        "trials": Because MuTrans uses the EM algorithm for attractor assignment, the results are local minimums and subject to initial assignments. The number of trials allow run the assignments for multiple times, and select the assignment with minimum objective functions. Increasing the trails will increase the running time, while increase the robustness of results.
        "initial": The initial cluster assignment in each trial. Can be "random" (random assignment -- fastest),"pca" (PCA+kmeans) or tsne (TSNE+kmeans). The default is "tsne".
        
        [parameters in dynamical manifold]
        "reduction_coord": The coordinates provided as the basis to construct dynamical manifold. Can be "tsne" or "umap", and the coordinates should be stored in the "obsm" attribute of anndata object. 
        
    Output: return the anndata object, with MuTrans outputs stored in the "da_out" and "land" key in the "uns" attribute of anndata object. Can be loaded back into matlab for gene analysis.
        
    """
    data = sc_object.X
    data_convert = matlab.double(data.tolist())
    
    if par.__contains__('perplex'):
        par['perplex'] = float(par['perplex'])
    
    if par.__contains__('K_cluster'):
        par['K_cluster'] = float(par['K_cluster'])
    
    if par.__contains__('cluster_id1'):
        par['cluster_id1'] = int(par['cluster_id1'])
    
    if par.__contains__('cluster_id2'):
        par['cluster_id2'] = int(par['cluster_id2'])
    
    if par.__contains__('end_set'):
        par['end_set'] = matlab.int64(par['end_set'])
    
    if par.__contains__('start_set'):
        par['start_set'] = matlab.int64(par['start_set'])
    
    if (par.__contains__('run_gene_analysis') == False) or par['run_gene_analysis']== False :
        out = eng.DynamicalAnalysis(data_convert,par)#o
    else:
        out,out_2 = eng.DynamicalAnalysis(data_convert,par)#o
        sc_object.uns['gene_out']=out_2##
        
    #out,out2 = eng.DynamicalAnalysis(data_convert,par)#out = eng.DynamicalAnalysis(adata,par)
    perm = np.asarray(out['perm_class']).ravel()
    perm = perm.astype(int)-1
    
    if (par.__contains__('reduction_coord') == False) or par['reduction_coord']== 'tsne':
        X_embedding = sc_object.obsm['X_tsne']
    elif par['reduction_coord']== 'umap':
        X_embedding = sc_object.obsm['X_umap']
    
    X_embedding = X_embedding- np.mean(X_embedding, axis =0)##0均值，关于原点对称
    
    if (par.__contains__('reduce_large_scale') == False) or par['reduce_large_scale']== False :
        embedding_r = X_embedding[perm]
        out["embedding_2d"] = matlab.double(embedding_r.tolist())
    else:
        out["embedding_2d"] = matlab.double(X_embedding.tolist())
    
    land = eng.ConstructLandscape(out,par)
    sc_object.uns['da_out']=out
    #sc_object.uns['gene_out']=out_2##
    sc_object.uns['land']=land
    
    if par.__contains__('write_anndata') and par['write_anndata']:
        ind = np.argsort(perm)
        sc_object.obs['land'] = np.asarray(land['land_cell'])[ind]
        sc_object.obs['entropy'] = np.asarray(out['H']).ravel()[ind]
        label = np.asarray(out['class_order']).astype(int)-1
        sc_object.obs['attractor'] = label.astype(str).ravel()[ind]
        sc_object.obsm['trans_coord'] = np.asarray(land['trans_coord'])[ind]
        sc_object.obsm['membership'] = np.asarray(out['rho_class'])[ind]
        sc_object.obsm['labs_perm'] = np.asarray(out['labs_perm'])[ind]
    
    
    return sc_object

def plot_landscape(sc_object,alpha_land = 0.5, show_colorbar = False):#True False
    land = sc_object.uns['land']
    land_value = np.asarray(land['land']).T
    x_grid = np.asarray(land['grid_x']).ravel()
    y_grid = np.asarray(land['grid_y']).ravel()
    # 创建自定义颜色映射
    #cmap = ListedColormap(['navy', 'darkblue', 'mediumblue', 'blue', 'dodgerblue', 'deepskyblue', 'lightblue', 'lightyellow', 'yellow'])

    cmap = ListedColormap([ '#3C72FF', '#27C2B1', '#6ECE6A', '#D6BE27', '#FFC23C','#FFFF14'])
    #cmap = ListedColormap([ 'blue', 'dodgerblue','#3C72FF', '#18ADDE', '#27C2B1', '#6ECE6A', '#D6BE27', '#FFC23C','#FFFF14'])
    #cmap = ListedColormap(['navy', 'blue', 'deepskyblue', 'lightblue', 'lightyellow', 'yellow'])
    plt.contourf(x_grid, y_grid, land_value, levels=14, cmap=cmap,zorder=-100, alpha = alpha_land)
    if show_colorbar:#cmap="Greys_r" "BrBG", interpolation='bilinear'
        plt.colorbar()
    
###3D
def plot3_landscape(sc_object,alpha_land = 0.5, show_colorbar = True):#True False
    land = sc_object.uns['land']
    land_value = np.asarray(land['land']).T
    land_cell=np.asarray(land['land_cell'])
    label=np.asarray(land['label']).T
    coord=np.asarray(land['trans_coord'])
    x_grid = np.asarray(land['grid_x']).ravel()
    y_grid = np.asarray(land['grid_y']).ravel()
    X, Y = np.meshgrid(x_grid, y_grid)
    land_value[land_value>max(land_cell)]=max(land_cell)
    ax = plt.axes(projection="3d")
    ax.plot_surface(X,Y,land_value,alpha=0.9, cstride=2, rstride = 2, cmap='rainbow')
    ax.grid()
    a=ax.scatter(coord[:,0],coord[:,1],land_cell,c=label)#,marker='^')
    ax.set_xlabel('coord_1') # 画出坐标轴
    ax.set_ylabel('coord_2')
    ax.set_zlabel('U')
    ax.legend(a.legend_elements())
   # ax.add_artist(legend1)
    #plt.contourf(x_grid, y_grid, land_value, levels=14, cmap="Greys_r",zorder=-100, alpha = alpha_land)
    
    
    if show_colorbar:
        ax.colorbar()
    ax.show()    
'''
def gene_analysis(cluster_id1, cluster_id2, sc_object, par):
    Output=matlab.double(sc_object.uns['da_out'])
    out = eng.GeneAnalysis(cluster_id1, cluster_id2, Output,par)
    out = eng.GeneAnalysis(1, 3, matlab.double(adata.uns['da_out']), par2)
    return out
'''
def gene_analysis(sc_object,sc_object2,genes,par,flux_fraction = 0.9):
    #sc_object=adata;sc_object2=adata2;genes=list(adata.var_names);flux_fraction = 0.5;
    P_hat = np.asarray(sc_object.uns['da_out']['P_hat'])
    M = msm.markov_model(P_hat)
    P_before=[]
    #P_set=[]
    transition_name=[str(a)+"_"+str(b)  for a,b in zip(par["start_set"][0],par["end_set"][0])]
    transition_name2=[str(b)+"_"+str(a)  for a,b in zip(par["start_set"][0],par["end_set"][0])]
    for i in range(len(par["start_set"][0])):
        tpt = msm.tpt(M, [par["start_set"][0][i]], [par["end_set"][0][i]])
        Fsub = tpt.major_flux(fraction=flux_fraction)
        Fsubpercent = 100.0 * Fsub / tpt.total_flux
        P_before=P_before+[Fsubpercent[par["start_set"][0][i],par["end_set"][0][i]]]
    #P_before.append
    land = sc_object.uns['land']
    barrier_height_before = np.asarray(land['barrier_height']).T
    
    P_gene=[]
    barrier_height_gene1=[]
    barrier_height_gene2=[]
    attractor=pd.DataFrame(sc_object.obs['attractor'])
    attractor.columns=["attractor1"]#,columns=["attractor1"])
    for i in tqdm(range(len(genes))):
        sc_temp=sc_object2[:,sc_object2.var_names!=genes[i]]#sc_temp=adata2
        sc.pp.log1p(sc_temp) 
        sc.tl.pca(sc_temp, svd_solver='arpack')
        sc.pp.neighbors(sc_temp,metric = par["choice_distance"],n_neighbors=50,use_rep='X')#n_pcs=10,use_rep='X'
        sc.tl.umap(sc_temp)
        sc.tl.leiden(sc_temp,resolution = 1.0)#need "neighbors" in .uns';resolution = 1.0,Higher values lead to more clusters.
        sc.tl.tsne(sc_temp)#, n_pcs=10
        data = sc_temp.X
        data_convert = matlab.double(data.tolist())
        if (par.__contains__('run_gene_analysis') == False) or par['run_gene_analysis']== False :
            out = eng.DynamicalAnalysis(data_convert,par)#o
        else:
            out,out_2 = eng.DynamicalAnalysis(data_convert,par)#o
            sc_object2.uns['gene_out']=out_2##
        perm = np.asarray(out['perm_class']).ravel()
        perm = perm.astype(int)-1
        ind = np.argsort(perm)
        attractor["attractor"]=(np.asarray(out['class_order']).astype(int)-1).ravel()[ind]
        a=np.array(pd.crosstab(attractor['attractor1'],attractor['attractor']))
        a1=np.argmax(a,axis=1)
        #a1=pd.crosstab(attractor['attractor1'],attractor.index)
        P_hat2=np.asarray(out['P_hat'])
        M2 = msm.markov_model(P_hat2)
        P_after=[]
        for j in range(len(par["start_set"][0])):
           tpt2 = msm.tpt(M2, [int(a1[par["start_set"][0][j]])], [int(a1[par["end_set"][0][j]])])
           Fsub2 = tpt2.major_flux(fraction=flux_fraction)
           Fsubpercent2 = 100.0 * Fsub2 / tpt2.total_flux
           P_after=P_after+[Fsubpercent2[a1[par["start_set"][0][j]],a1[par["end_set"][0][j]]]]
        #P_after=Fsubpercent2[a1[pi],a1[pf]]
        P_gene.append([genes[i]]+P_after)
        
        ##### barrier height
        if (par.__contains__('reduction_coord') == False) or par['reduction_coord']== 'tsne':
             X_embedding = sc_object.obsm['X_tsne']
        elif par['reduction_coord']== 'umap':
             X_embedding = sc_object.obsm['X_umap']
    
        X_embedding = X_embedding- np.mean(X_embedding, axis =0)##0均值，关于原点对称
    
        if (par.__contains__('reduce_large_scale') == False) or par['reduce_large_scale']== False :
             embedding_r = X_embedding[perm]
             out["embedding_2d"] = matlab.double(embedding_r.tolist())
        else:
             out["embedding_2d"] = matlab.double(X_embedding.tolist())
    
        land = eng.ConstructLandscape(out,par)
        barrier=np.asarray(land['barrier_height']).T
        barrier_height_gene1.append([genes[i]]+barrier[0,:].tolist())
        barrier_height_gene2.append([genes[i]]+barrier[1,:].tolist())
    
    return pd.DataFrame(P_gene,columns=["gene"]+transition_name),P_before,pd.DataFrame(barrier_height_gene1,columns=["gene"]+transition_name),pd.DataFrame(barrier_height_gene2,columns=["gene"]+transition_name2),barrier_height_before

def plot_cluster_num (sc_object, par = None, k_plot = 20, order =2.0,save=False):
    data = sc_object.X
    data_convert = matlab.double(data.tolist())
    if par == None:
        par = {}
        par ['choice_distance'] = 'euclid'
    
    par['order'] = order
    cluster_num_est = eng.EstClusterNum(data_convert,par)
    plt.figure()
    plt.plot(np.arange(k_plot)+1,np.asarray(cluster_num_est['ratio'])[:k_plot])
    plt.xlabel("k",fontsize=10)
    plt.ylabel("EPI",fontsize=10)
    if save!=False:
        plt.savefig(save,dpi=300)
    plt.show()
    cluster_num={}
    cluster_num["lambda_k"]=np.asarray(cluster_num_est["lambda_k"]).ravel()
    cluster_num["prop"]=np.asarray(cluster_num_est["prop"]).ravel()
    cluster_num["ratio"]=np.asarray(cluster_num_est["ratio"]).ravel()
    cluster_num["attractor"]=[]
    cluster_num["attractor"].append(np.argmax(cluster_num["ratio"][0:20])+1)
    '''
    for i in range(1,20):
        if cluster_num["ratio"][i]>cluster_num["ratio"][i-1] and cluster_num["ratio"][i]>cluster_num["ratio"][i+1]:
            cluster_num["attractor"].append(i+1)
    '''
    return cluster_num

    
    
def infer_lineage(sc_object,si=0,sf=1,method = 'MPFT',flux_fraction = 0.9, 
                  size_state = 0.1, size_point = 50, alpha_land = 0.5, 
                  alpha_point = 0.5, size_text=20, state_labels=None,
                  show_colorbar = False, color_palette = None,arrow_scale=1.0,
                  save=False):
    """ Infer the lineage by MPFT or MPPT approach.
    sc_object=adata,si=0,sf=1,method = 'MPFT',flux_fraction = 0.9, size_state = 0.1, size_point = 50, alpha_land = 0.5, alpha_point = 0.5, size_text=20
    Arguments: 
        
        sc_object: the anndata object storing single-cell datasets and MuTrans output (after running the dynamical_analysis function).
        
        method: method to infer the lineage, can be "MPFT" (Maximum Probability Flow Tree, global structure) or "MPPT" (Most Probable Path Tree, local transitions).
        
        si and sf: starting and targeting attractors in MPPT method for transition paths analysis.
        
        flux_fraction: the cumulative percentage of top transition paths probability flux shown on the figure.
        
    """
    projection = np.asarray(sc_object.uns['land']['trans_coord'])
    K = sc_object.uns['da_out']['k']
    K = int(K)
    P_hat = np.asarray(sc_object.uns['da_out']['P_hat'])
    mu_hat = np.asarray(sc_object.uns['da_out']['mu_hat'])
    
    centers = []
    labels = np.asarray(sc_object.uns['da_out']['class_order']).ravel()
    labels = labels.astype(int)-1
    
    if color_palette is None:
        color_palette = sns.color_palette('Set1', K)
    cluster_colors = [color_palette[x] for x in labels]
    
    
    for i in range(K):
        index = labels==i
        p = np.mean(projection[index], axis=0)
        centers.append(p)
    centers = np.array(centers)

    if method == 'MPFT':
        Flux_cg = np.diag(mu_hat.reshape(-1)).dot(P_hat)
        max_flux_tree = scipy.sparse.csgraph.minimum_spanning_tree(-Flux_cg)##最小生成树while minimizing the total sum of weights on the edges. 
        #This is computed using the Kruskal algorithm.
        max_flux_tree = -max_flux_tree.toarray()
        for i in range(K):
            for j in range(i+1,K):   
                max_flux_tree[i,j]= max(max_flux_tree[i,j],max_flux_tree[j,i])
                max_flux_tree[j,i] = max_flux_tree[i,j]
        fig = plt.figure(figsize=(4, 4),dpi=300) 
        plot_landscape(sc_object, alpha_land=alpha_land,show_colorbar = show_colorbar)
       # plt.scatter(*projection.T, s=size_point, linewidth=0, c=cluster_colors, alpha=alpha_point)
        
        nw.plot_network(max_flux_tree, pos=centers, state_scale=size_state, state_sizes=mu_hat, 
                        arrow_scale=arrow_scale,arrow_labels= None, arrow_curvature = 0.2, ax=plt.gca(),
                        state_labels=state_labels,max_width=1000, max_height=1000)
        if save!=False:
            fig.savefig(save, dpi=300)
        #plot_landscape(sc_object, alpha_land=alpha_land,show_colorbar = show_colorbar)
        #plt.scatter(*projection.T, s=size_point, linewidth=0, c=cluster_colors, alpha=alpha_point)
        
    if method == 'MPPT':
        M = msm.markov_model(P_hat)###Markov model with a given transition matrix
        #Returns a MSM that contains the transition matrix and 
        #allows to compute a large number of quantities related to Markov models.
        
        #state_reorder = np.array(range(K))
        #state_reorder[0] = si
        #state_reorder[-1] = sf
        #state_reorder[sf+1:-1]=state_reorder[sf+1:-1]+1
        #state_reorder[1:si]=state_reorder[1:si]-1
        tpt = msm.tpt(M, [si], [sf])
        Fsub = tpt.major_flux(fraction=flux_fraction)
        Fsubpercent = 100.0 * Fsub / tpt.total_flux
        
        fig = plt.figure(figsize=(4, 4),dpi=300) 
        plot_landscape(sc_object,alpha_land=alpha_land, show_colorbar = show_colorbar)
        #plt.scatter(*projection.T, s=size_point, linewidth=0, c=cluster_colors, alpha=alpha_point)
        plt.axis('off')
        nw.plot_network(Fsubpercent, state_scale=size_state,pos=centers, arrow_label_format="%3.1f",
                        state_labels=state_labels,arrow_label_size = size_text,ax=plt.gca(), max_width=1000, max_height=1000)
        if save!=False:
            fig.savefig(save, dpi=300)
        #plt.scatter(*projection.T, s=size_point, linewidth=0, c=cluster_colors, alpha=alpha_point)
        #plt.axis('off')        
                
                
                
                