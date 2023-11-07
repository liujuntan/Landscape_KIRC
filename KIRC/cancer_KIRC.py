# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:38:50 2023

@author: 27226
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib import rcParams
sc.settings.set_figure_params(dpi=100, frameon=False, figsize=(3, 3), facecolor='white')

import matplotlib.pyplot as plt
import seaborn as sns
color_palette = sns.color_palette('Set1', 5)

import sys
# 将__init__.py文件所在目录加入Python搜索目录中，否则会提示找不到myModule1
sys.path.append(r'C:\Users\27226\Desktop\Landscape_KIRC\KIRC')
import pyMuTrans as pm
import leidenalg
from tqdm import tqdm
import matlab.engine

##读取数据
datadir = "C:/Users/27226/Desktop/Landscape_KIRC/KIRC/"
adata = sc.read_csv(datadir+'KIRC_data.csv').T
savedir="C:/Users/27226/Desktop/Desktop/Landscape_KIRC/result/"

#gene_nameKIRC:
A=[0,73,197,41,112,68]#COAD:A=[0,41,121,307,439,505];
A=np.cumsum(A)
#A=[0,41,121,307,439,505];
true_labs=[]
true_labs=['A' for i in range(A[0],A[1])]+['I' for i in range(A[1],A[2])]+\
['II' for i in range(A[2],A[3])]+['III' for i in range(A[3],A[4])]+['IV' for i in range(A[4],A[5])]
adata.obs_names=true_labs
adata.obs['stage']=true_labs
##数据标准化
adata.var_names_make_unique()
sc.pp.log1p(adata) ## X=log(X+1) natural  logarithm

def swap_num(data,mapping_list):
    #original_list = [0,1,2,3,4]
    #mapping = {0:4, 1:2, 2:1, 3:0, 4:3}
    mapping={i:mapping_list[i] for i in range(len(mapping_list))}
    new_list = [str(mapping[int(i)]) for i in data]
    return new_list

###########  探测吸引子个数
par = {"run_gene_analysis":False,"choice_distance":"cosine","write_anndata":True} 
#cluster_num=pm.plot_cluster_num(adata,par,save=savedir+'KIRC/KIRC_cluster_num.png')
cluster_num=pm.plot_cluster_num(adata,par)
cluster_num["attractor"]

##降维分析
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata,metric = 'cosine',n_neighbors=50,use_rep='X')#n_pcs=10,use_rep='X'
sc.tl.leiden(adata,resolution = 1.0)
sc.tl.tsne(adata)#, n_pcs=10
par = {"run_gene_analysis":False,"choice_distance":"cosine","perplex":100,
       "K_cluster":5.0,"trials":5,"reduction_coord":'tsne',"write_anndata":True,
        "display_legend":True,"legend_text":["0","1","2","3","4"],
       "scaleaxis":1.5,"scalevalue":1.5,"visual_dim":3,"mksize":20}
       #"start_set":[0,1,1,4,2],"end_set":[1,4,2,2,3]} 
adata = pm.dynamical_analysis(adata,par)
plt.figure(figsize=(12, 4))
fig = sc.pl.tsne(adata,color = ['stage','leiden','attractor'],legend_loc='on data',return_fig = True)
fig.savefig(savedir+"KIRC_cluster_tsne2.png",dpi=600)

ax.set_title('Number of stage by attractor')
ax.legend()
ax.grid(None)
plt.show()
sns.heatmap(a,cmap='Reds',annot=True,fmt="d")

########
P_hat = np.asarray(adata.uns['da_out']['P_hat'])
ax=sns.heatmap(P_hat,cmap='Reds')#,annot=True,fmt=".2f")
ax.set(xlabel="attractor", ylabel="attractor")
ax.set_title("transition probability")
ax.figure.set_size_inches(3.2, 3.2)  # 设置图像尺寸为10x8英寸
ax.figure.savefig(savedir+"KIRC_transition_prob_cluster.png",dpi=300)

################           infer lineage       ##################################
pm.infer_lineage(adata,si=0,sf = 3,method = "MPFT",flux_fraction = 0.95,
                 size_state = 0.2,state_labels=[0,1,3,4,2],size_text=16,
                 alpha_point = 0.7,arrow_scale=0.6,show_colorbar = True,
                 alpha_land = 1,save=savedir+'KIRC_lineage_MPFT_TA_IV.png')
pm.infer_lineage(adata,si=0,sf = 3,method = "MPPT",flux_fraction = 0.5,
                 size_point = 30,state_labels=[0,1,3,4,2],
                 size_state = 0.1,size_text=16,alpha_point = 0.7,show_colorbar = True,
                 alpha_land = 1,save=savedir+'KIRC_lineage_MPPT_TA_IV.png')

sc.tl.paga(adata, groups='attractor')
sc.pl.paga(adata, color=['attractor'])
sc.pl.paga_path(adata)

sc.tl.draw_graph(adata, init_pos='paga')
sc.pl.draw_graph(adata, color=['stage','leiden','attractor'], legend_loc='on data')


########################      gene analysis       ##################################
adata2 = sc.read_csv(datadir+'KIRC_data.csv').T
adata2.obs_names=true_labs
adata2.obs['stage']=true_labs
##数据标准化
adata2.var_names_make_unique()
#sc.pp.log1p(adata2) 
genes=list(adata2.var_names)
par = {"run_gene_analysis":False,"choice_distance":"cosine","write_anndata":True} 
####### 探测吸引子个数
##降维分析
par = {"run_gene_analysis":False,"choice_distance":"cosine","perplex":100.0,
       "K_cluster":5.0,"trials":5,"reduction_coord":'tsne',"write_anndata":True,
        "display_legend":True,"legend_text":["0","1","3","4","2"],
       "scaleaxis":1.2,"scalevalue":1.5,"visual_dim":3,
       "start_set": matlab.int64([0,1,1,4,2]),"end_set": matlab.int64([1,4,2,2,3]),"plot_landscape":False} 
genes=list(adata.var_names)
P_gene_result,P_before,barrier_height_1,barrier_height_2,barrier_height_before=pm.gene_analysis(adata,adata2,genes,par,flux_fraction = 0.5)
##保存数据
P_gene_result.to_csv(savedir+"KIRC_P_gene_result.csv")#P_gene_result['gene']=genes
pd.DataFrame(P_before).to_csv(savedir+"KIRC_P_before.csv")
barrier_height_1.to_csv(savedir+"KIRC_barrier_height_1.csv")#barrier_height_1['gene']=genes
barrier_height_2.to_csv(savedir+"KIRC_barrier_height_2.csv")#barrier_height_2['gene']=genes
pd.DataFrame(barrier_height_before).to_csv(savedir+"KIRC_barrier_height_before.csv")

##读取数据
P_gene_result=pd.read_csv(savedir+"KIRC_P_gene_result.csv",index_col=0)#P_gene_result['gene']=genes
P_before=pd.read_csv(savedir+"KIRC_P_before.csv",index_col=0)
barrier_height_1=pd.read_csv(savedir+"KIRC_barrier_height_1.csv",index_col=0)#barrier_height_1['gene']=genes
barrier_height_2=pd.read_csv(savedir+"KIRC_barrier_height_2.csv",index_col=0)#barrier_height_2['gene']=genes
barrier_height_before=pd.read_csv(savedir+"KIRC_barrier_height_before.csv",index_col=0)

##画图分析
name=list(P_gene_result.columns[1:])
name2=list(barrier_height_2.columns[1:])
name_map={0:'TA',1:'I',2:'III',3:'IV',4:'II'}
for i in range(len(name)):
  config = { "font.family": 'Arial', "font.size": 8,}
  P_gene_result2=P_gene_result[["gene",name[i]]]
  P_gene_result2[name[i]]=P_before['0'][i]-P_gene_result2[name[i]]
  barrier_height_12=barrier_height_1[["gene",name[i]]]
  barrier_height_12[name[i]]=barrier_height_before.iloc[0,i]-barrier_height_12[name[i]]
  barrier_height_22=barrier_height_2[["gene",name2[i]]]
  barrier_height_22[name[i]]=barrier_height_before.iloc[1,i]-barrier_height_22[name2[i]]
  #for i in range()

  gene_result = P_gene_result2.sort_values(by=name[i],ascending=True)  # by指定按哪列排序。ascending表示是否升序
  gene_result.reset_index(inplace=True,drop=True)

  gene_result2 = barrier_height_12.sort_values(by=name[i],ascending=True)  # by指定按哪列排序。ascending表示是否升序
  gene_result2.reset_index(inplace=True,drop=True)

  gene_result3 = barrier_height_22.sort_values(by=name2[i],ascending=True)  # by指定按哪列排序。ascending表示是否升序
  gene_result3.reset_index(inplace=True,drop=True)

  n=10#show gene number
  gene_show=pd.concat([gene_result[0:n],gene_result[-n:]])
  gene_show.reset_index(inplace=True,drop=True)

  gene_show2=pd.concat([gene_result2[0:n],gene_result2[-n:]])
  gene_show2.reset_index(inplace=True,drop=True)

  gene_show3=pd.concat([gene_result3[0:n],gene_result3[-n:]])
  gene_show3.reset_index(inplace=True,drop=True)

  plt.figure(figsize=(10, 4))
  plt.rcParams["font.sans-serif"]=['SimHei']
  plt.rcParams["axes.unicode_minus"]=False
  #plt.bar(gene_show["gene"],gene_show[name[i]])
  # 绘制柱状图并设置纵坐标颜色和标签
  for j in range(len(gene_show)):
      plt.bar(gene_show["gene"][j], gene_show[name[i]][j], color='#6464ff' if gene_show[name[i]][j] > 0 else '#ff6464')
      #plt.text(gene_show["gene"][i], gene_show["prob"][i]+0.5,{:.2f}gene_show["prob"][i]., ha='center')
  plt.grid (None)
  # 设置标题和坐标轴标签
  plt.xticks(gene_show["gene"],rotation=45,fontsize=10) 
  plt.title("from {} to {}:{:.1f} ".format(name_map[eval(name[i][0])],name_map[eval(name[i][-1])],P_before['0'][i]),fontsize=15)
  plt.xlabel("gene")
  plt.ylabel(r'$\Delta$P(%)')
  plt.savefig(savedir+"KIRC_transition_gene/P_"+name_map[eval(name[i][0])]+"_"+name_map[eval(name[i][-1])]+".png",dpi=300)
  plt.show()

  plt.figure(figsize=(10, 4))
  for j in range(len(gene_show2)):
      plt.bar(gene_show2["gene"][j], gene_show2[name[i]][j], color='#6464ff' if gene_show2[name[i]][j] > 0 else '#ff6464')
      #plt.text(gene_show["gene"][i], gene_show["prob"][i]+0.5,{:.2f}gene_show["prob"][i]., ha='center')
  plt.grid (None)
  # 设置标题和坐标轴标签
  plt.xticks(gene_show2["gene"],rotation=45,fontsize=10) 
  plt.title("from {} to {}:{:.1f} ".format(name_map[eval(name[i][0])],name_map[eval(name[i][-1])],barrier_height_before.iloc[0,i]),fontsize=15)
  plt.xlabel("gene")
  plt.ylabel(r'$\Delta$H')
  plt.savefig(savedir+"KIRC_transition_gene/H_"+name_map[eval(name[i][0])]+"_"+name_map[eval(name[i][-1])]+".png",dpi=300)
  plt.show()

  plt.figure(figsize=(10, 4))
  for j in range(len(gene_show3)):
      plt.bar(gene_show3["gene"][j], gene_show3[name2[i]][j], color='#6464ff' if gene_show3[name2[i]][j] > 0 else '#ff6464')
      #plt.text(gene_show["gene"][i], gene_show["prob"][i]+0.5,{:.2f}gene_show["prob"][i]., ha='center')
  plt.grid (None)
  # 设置标题和坐标轴标签
  plt.xticks(gene_show3["gene"],rotation=45,fontsize=10) 
  plt.title("from {} to {}:{:.1f} ".format(name_map[eval(name2[i][0])],name_map[eval(name2[i][-1])],barrier_height_before.iloc[1,i]),fontsize=15)
  plt.xlabel("gene")
  plt.ylabel(r'$\Delta$H')
  plt.savefig(savedir+"KIRC_transition_gene/H_"+name_map[eval(name2[i][0])]+"_"+name_map[eval(name2[i][-1])]+".png",dpi=300)
  plt.show()

###### heatmap分析
#marker_genes_dict
ax = sc.pl.heatmap(adata, gene_show3["gene"], groupby='stage',
                   cmap='Reds', dendrogram=True, standard_scale='var', 
                   figsize=(12, 6), swap_axes=False)
#######scanpy gene DEG分析
sc.tl.rank_genes_groups(adata, 'attractor', method='wilcoxon')
sc.pl.rank_genes_groups(adata, n_genes=10, sharey=True, ncols = 5)

fig=sc.pl.tsne(adata, color=['PIK3C2G','ACVRL1','GATA4','MC4R','MYBPC1'],vmax = 'p99',vmin = 'p1', ncols = 5,return_fig = True)

#小提琴图
sc.settings.set_figure_params(frameon=False,figsize=(4, 3))
fig=sc.pl.violin(adata, keys = ['PIK3C2G','ACVRL1','GATA4','MC4R','MYBPC1'], groupby = 'attractor',ncols = 5,return_fig = True)

sc.tl.rank_genes_groups(adata, 'stage', method='wilcoxon')
sc.pl.rank_genes_groups(adata, n_genes=10, sharey=True, ncols = 5)

fig=sc.pl.tsne(adata, color=['MUC15','ACVRL1','GATA4','MC4R','MYBPC1'],vmax = 'p99',vmin = 'p1', ncols = 5,return_fig = True)
#小提琴图
sc.settings.set_figure_params(frameon=False,figsize=(4, 3))
fig=sc.pl.violin(adata, keys = ['MUC15','ACVRL1','GATA4','MC4R','MYBPC1'], groupby = 'stage',ncols = 5,return_fig = True)

### TA_I critical gene KRT4 ATP12A MMP3 ADH4
fig=sc.pl.tsne(adata, color=['KRT4', 'ATP12A', 'MMP3', 'ADH4'],vmax = 'p99',vmin = 'p1', ncols = 4,return_fig = True)
###III_IV    CALCA INSM1 NR0B2 COL2A1
fig=sc.pl.tsne(adata, color=['CALCA', 'CPB2', 'NR0B2', 'COL2A1'],vmax = 'p99',vmin = 'p1', ncols = 4,return_fig = True)
#小提琴图