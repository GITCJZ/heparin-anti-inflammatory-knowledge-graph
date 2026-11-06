"""Rebuild Figure 8 and Cytoscape-compatible GraphML from Supplementary Data S1.

Dependencies: Python 3.10+, pandas, openpyxl, networkx, matplotlib, seaborn.
All layouts are deterministic (fixed seed 20260923). The figure is an original
evidence-informed synthesis and contains no reproduced published artwork.
The Edges sheet uses a submission-friendly flat table with headers in row 1.
"""
from __future__ import annotations

import math
import os
import shutil
import tempfile
import textwrap
from pathlib import Path

_mpl_cache = Path(tempfile.gettempdir()) / f"heparin_kg_mplcache_{os.getpid()}"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_mpl_cache)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

SEED = 20260923
np.random.seed(SEED)
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Supplementary_Data_S1_Heparin_Knowledge_Graph.xlsx"
OUT_SVG = ROOT / "Figure9_Heparin_Knowledge_Graph.svg"
OUT_PDF = ROOT / "Figure9_Heparin_Knowledge_Graph.pdf"
OUT_PNG = ROOT / "Figure9_Heparin_Knowledge_Graph_600dpi.png"
OUT_GRAPHML = ROOT / "Figure9_Heparin_Knowledge_Graph.graphml"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 7.2,
    "axes.titlesize": 8.2,
    "axes.labelsize": 7.4,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})
sns.set_style("white")

nodes = pd.read_excel(DATA, sheet_name="Nodes", header=2).fillna("")
edges = pd.read_excel(DATA, sheet_name="Edges", header=0).fillna("")
matrix = pd.read_excel(DATA, sheet_name="Format_Mechanism_Matrix", header=2).fillna("")
disease = pd.read_excel(DATA, sheet_name="Disease_Translation_Map", header=2).fillna("")

# Cytoscape-compatible complete network. Every value is scalar and every edge is cited.
G = nx.MultiDiGraph(name="Heparin anti-inflammatory pharmacology knowledge graph")
for _, row in nodes.iterrows():
    attrs = {k: (int(v) if isinstance(v, np.integer) else str(v)) for k, v in row.items() if k != "Node_ID"}
    G.add_node(str(row["Node_ID"]), **attrs)
for _, row in edges.iterrows():
    attrs = {}
    for k, v in row.items():
        if k in {"Source_Node", "Target_Node"}: continue
        if k == "Evidence_Tier": attrs[k] = int(v)
        else: attrs[k] = str(v)
    G.add_edge(str(row["Source_Node"]), str(row["Target_Node"]), key=str(row["Edge_ID"]), **attrs)
nx.write_graphml(G, OUT_GRAPHML, infer_numeric_types=True)

node_lookup = nodes.set_index("Node_ID").to_dict("index")
degree = dict(G.degree())

entity_colors = {
    "Heparin structure/form": "#4E79A7",
    "Structural determinant": "#255E91",
    "Molecular target/mediator": "#F28E2B",
    "Mechanism/process": "#59A14F",
    "Biological compartment": "#76B7B2",
    "Disease/clinical context": "#B07AA1",
    "Outcome/safety": "#E15759",
}
entity_markers = {
    "Heparin structure/form": "o",
    "Structural determinant": "D",
    "Molecular target/mediator": "s",
    "Mechanism/process": "^",
    "Biological compartment": "h",
    "Disease/clinical context": "P",
    "Outcome/safety": "X",
}

def edge_colour(direction: str, relationship: str) -> str:
    d = direction.lower(); r = relationship.lower()
    if "harm" in d or r in {"promotes", "increases_risk_of"}: return "#D55E00"
    if "favour" in d or "protect" in d or r in {"inhibits", "reduces", "preserves", "neutralizes", "fails_to_improve"}: return "#0072B2"
    if r in {"binds", "competes_with", "potentiates", "requires", "disrupts"}: return "#CC79A7"
    if r in {"localizes_to", "delivers_to", "associated_with", "tested_in"}: return "#6B7280"
    return "#009E73"

line_style = {"Direct": "-", "Indirect": "--", "Proposed": ":"}

def add_panel_label(ax, label):
    ax.text(-0.015, 1.025, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")

def add_box(ax, xy, wh, text, fc, ec="#64748B", size=7.1, radius=0.02, weight="normal"):
    x,y=xy; w,h=wh
    box=FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0.008,rounding_size={radius}",facecolor=fc,edgecolor=ec,linewidth=0.8)
    ax.add_patch(box)
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=size,fontweight=weight,color="#172033")
    return box

# Exactly 180 mm wide. A tall multi-panel layout keeps Panel B labels at >=7 pt.
fig = plt.figure(figsize=(180/25.4, 238/25.4), facecolor="white")
gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 4.35, 2.15], width_ratios=[1.25, 1.0], hspace=0.23, wspace=0.28,
                      left=0.125, right=0.985, top=0.966, bottom=0.125)
axA = fig.add_subplot(gs[0, :])
axB = fig.add_subplot(gs[1, :])
axC = fig.add_subplot(gs[2, 0])
axD = fig.add_subplot(gs[2, 1])

# Panel A: construction flow and data model.
axA.set_xlim(0,1); axA.set_ylim(0,1); axA.axis("off")
axA.text(-0.035,0.94,"A",fontsize=10,fontweight="bold",ha="left",va="top")
axA.text(0.0,0.94,"Knowledge graph schema and construction",fontsize=8.3,fontweight="bold",ha="left",va="top")
flow=["Manuscript evidence\nand cited studies","Entity and relationship\nextraction","Synonym\nnormalisation","Reference-level\nverification","Evidence\nclassification","Knowledge graph\nconstruction"]
xs=np.linspace(0.01,0.835,len(flow)); w=0.145; h=0.28
for i,(x,t) in enumerate(zip(xs,flow)):
    add_box(axA,(x,0.56),(w,h),t,"#F4F7FA",size=7.0,weight="bold")
    if i<len(flow)-1:
        axA.annotate("",xy=(xs[i+1]-0.004,0.70),xytext=(x+w+0.004,0.70),arrowprops=dict(arrowstyle="-|>",lw=0.9,color="#475569"))

hier=[
    ("Heparin structure/form","#4E79A7","o"),("Structural determinant","#255E91","D"),("Molecular target","#F28E2B","s"),
    ("Mechanism","#59A14F","^"),("Biological compartment","#76B7B2","h"),("Disease/context","#B07AA1","P"),("Outcome/safety","#E15759","X")]
hx=np.linspace(0.045,0.955,len(hier))
for i,(x,(t,c,m)) in enumerate(zip(hx,hier)):
    axA.scatter([x],[0.27],s=70,marker=m,c=c,edgecolors="#263238",linewidths=0.6,zorder=3)
    axA.text(x,0.07,textwrap.fill(t,20),ha="center",va="bottom",fontsize=7.0)
    if i<len(hier)-1:
        axA.annotate("",xy=(hx[i+1]-0.025,0.27),xytext=(x+0.025,0.27),arrowprops=dict(arrowstyle="->",lw=0.8,color="#64748B"))

# Panel B: curated 56-node, 60-edge core graph.
axB.text(-0.035,1.045,"B",transform=axB.transAxes,fontsize=10,fontweight="bold",ha="left",va="bottom")
axB.text(0.0,1.045,"Core reference-anchored network",transform=axB.transAxes,fontsize=8.2,fontweight="bold",ha="left",va="bottom")
axB.set_xlim(0,1); axB.set_ylim(0,1); axB.axis("off")
for y0,y1,label,fc in [(0.77,0.995,"Chemokine organisation and selectin recruitment","#F2F9F8"),(0.61,0.765,"Heparanase–glycocalyx injury","#F4F9F0"),(0.37,0.605,"Extracellular danger cargo and pyroptosis","#FFF6ED"),(0.035,0.365,"Complement, coagulation and translation","#F8F3FA")]:
    axB.add_patch(FancyBboxPatch((0.002,y0),0.996,y1-y0,boxstyle="round,pad=0.002,rounding_size=0.008",facecolor=fc,edgecolor="none",zorder=0))
    axB.text(0.006,y1-0.008,label,fontsize=7.0,fontstyle="italic",color="#64748B",ha="left",va="top")

layer_x={"left":0.07,"target":0.29,"mech":0.52,"comp":0.73,"right":0.90}
manual_order={
 "left":["F_DEFINED","F_HEP","S_6OS","F_LOCAL","F_NAH","F_RONE","S_GLY","F_ODSH","F_UFH","F_APAC","F_IMMOB","F_ADS","F_INHALED","F_LMWH","F_NECU"],
 "target":["T_CCL5","T_CXCL8","T_PSEL","T_LSEL","T_HS","T_HPSE","T_HMGB1","T_LPS","T_RAGE","T_HIST","T_NET","T_C1I","T_FH","T_COMP","T_PF4","T_PLT","T_COAG"],
 "mech":["M_CHEMPRES","M_PHASE","M_ROLL","M_RECRUIT","M_GLYCO","M_BARRIER","M_HMGBLPS","M_CYTLPS","M_CASP11","M_GSDMD","M_PYRO","M_PLTLEU","M_COAGX","M_IMMTH"],
 "comp":["C_VASC","C_GLYCO","C_CYTO","C_MAT","C_AIR"],
 "right":["D_SEPSIS","O_MORT","D_ARDS","D_COVID","O_THROMB","D_UC","D_CAI","O_SURV"]}
anchors={
 "left":(0.965,0.07),"target":(0.965,0.075),"mech":(0.95,0.09),"comp":(0.87,0.14),"right":(0.87,0.10)}
pos={}
core_edges=edges[edges["Panel_B"].astype(str).str.lower()=="yes"].copy()
connected=set(core_edges["Source_Node"]).union(set(core_edges["Target_Node"]))
for layer,ids in manual_order.items():
    ids=[i for i in ids if i in connected]
    top,bottom=anchors[layer]
    ys=np.linspace(top,bottom,len(ids)) if len(ids)>1 else [0.5]
    for i,y in zip(ids,ys): pos[i]=(layer_x[layer],float(y))

# All connected core nodes must have a deterministic layer position.
missing=connected-set(pos)
if missing:
    raise RuntimeError(f"Panel B nodes missing manual positions: {sorted(missing)}")

# Edges first, with small deterministic curvature to reduce overlap.
for j,(_,e) in enumerate(core_edges.iterrows()):
    u,v=str(e["Source_Node"]),str(e["Target_Node"])
    x1,y1=pos[u]; x2,y2=pos[v]
    rad=0.0
    if abs(x2-x1)<0.04: rad=0.16 if j%2==0 else -0.16
    elif abs(y2-y1)<0.025: rad=0.06 if j%2==0 else -0.06
    colour=edge_colour(str(e["Direction"]),str(e["Relationship"]))
    patch=FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=5.0,
                          connectionstyle=f"arc3,rad={rad}",linewidth=0.45+0.12*int(e["Evidence_Tier"]),
                          linestyle=line_style.get(str(e["Directness"]),"-"),color=colour,alpha=0.38,zorder=1,
                          shrinkA=5,shrinkB=5)
    axB.add_patch(patch)

# Nodes and labels; size reflects evidence-linked degree only.
for nid,(x,y) in pos.items():
    meta=node_lookup[nid]; typ=meta["Entity_Type"]
    size=min(145,34+10*degree.get(nid,1))
    axB.scatter([x],[y],s=size,marker=entity_markers[typ],c=entity_colors[typ],edgecolors="#263238",linewidths=0.55,zorder=3)
    label=str(meta["Preferred_Label"])
    label=textwrap.fill(label,18)
    dx=0.012 if x<0.84 else -0.012
    ha="left" if x<0.84 else "right"
    axB.text(x+dx,y,label,fontsize=7.0,ha=ha,va="center",color="#111827",zorder=4,
             bbox=dict(boxstyle="round,pad=0.08",fc="white",ec="none",alpha=0.72))

headers=[("Structures/formats",layer_x["left"]),("Targets/mediators",layer_x["target"]),("Mechanisms",layer_x["mech"]),("Compartments",layer_x["comp"]),("Diseases/outcomes",layer_x["right"])]
for t,x in headers: axB.text(x,1.005,t,ha="center",va="bottom",fontweight="bold",fontsize=7.2,color="#334155")

# Compact action and directness legend.
action_handles=[Line2D([0],[0],color="#0072B2",lw=1.5,label="inhibitory/protective"),Line2D([0],[0],color="#D55E00",lw=1.5,label="promoting/harmful"),Line2D([0],[0],color="#CC79A7",lw=1.5,label="binding/competition"),Line2D([0],[0],color="#6B7280",lw=1.5,label="context/localisation")]
style_handles=[Line2D([0],[0],color="#333",lw=1.2,ls="-",label="Direct"),Line2D([0],[0],color="#333",lw=1.2,ls="--",label="Indirect"),Line2D([0],[0],color="#333",lw=1.2,ls=":",label="Proposed")]
leg1=axB.legend(handles=action_handles,loc="lower left",bbox_to_anchor=(0.0,-0.01),ncol=4,frameon=False,handlelength=1.8,columnspacing=0.8)
axB.add_artist(leg1)
axB.legend(handles=style_handles,loc="lower right",bbox_to_anchor=(1.0,-0.065),ncol=3,frameon=False,handlelength=1.8,columnspacing=0.8)

# Panel C: evidence matrix, derived directly from the Edges sheet.
axC.text(-0.02,1.055,"C",transform=axC.transAxes,fontsize=10,fontweight="bold",ha="left",va="bottom")
axC.text(0.055,1.055,"Heparin format–mechanism evidence matrix",transform=axC.transAxes,fontsize=8.2,fontweight="bold",ha="left",va="bottom")
mat = matrix.set_index(matrix.columns[0]).replace("",np.nan).astype(float)
mat.index=["UFH","LMWH","NAH","ODSH","roneparstat","necuparanib","defined glycans","inhaled/local","immobilised/material"]
mat.columns=["Cytokine signalling","Chemokine organisation","Selectin recruitment","Glycocalyx protection","Histone/HMGB1/NET","Complement","Immunothrombosis"]
cmap=ListedColormap(["#FFFFFF","#EDF8FB","#B3CDE3","#8C96C6","#8856A7","#810F7C"])
norm=BoundaryNorm(np.arange(-0.5,6.5,1),cmap.N)
sns.heatmap(mat,ax=axC,cmap=cmap,norm=norm,mask=mat.isna(),annot=True,fmt=".0f",annot_kws={"fontsize":7.0},
            linewidths=0.5,linecolor="#D7DEE6",cbar_kws={"label":"Highest evidence tier","ticks":range(6),"shrink":0.78,"pad":0.02})
axC.set_xlabel(""); axC.set_ylabel("")
axC.tick_params(axis="x",rotation=38,pad=1); axC.tick_params(axis="y",rotation=0,pad=1)
axC.text(0,-0.46,"Colour indicates evidence depth, not effect size;\nblank = insufficient evidence identified in this review.",transform=axC.transAxes,fontsize=7.0,ha="left",va="top")

# Panel D: review-derived translation-gap map.
axD.text(-0.02,1.055,"D",transform=axD.transAxes,fontsize=10,fontweight="bold",ha="left",va="bottom")
axD.text(0.065,1.055,"Mechanistic–clinical translation gap",transform=axD.transAxes,fontsize=8.2,fontweight="bold",ha="left",va="bottom")
direction_colors={"favourable":"#009E73","mixed":"#E69F00","neutral/negative":"#D55E00","insufficient":"#9CA3AF"}
for _,r in disease.iterrows():
    x=float(r["Mechanistic_Evidence_Depth"]); y=float(r["Clinical_Evidence_Depth"]); n=float(r["Supporting_Human_Studies"])
    d=str(r["Clinical_Direction"]).lower(); c=direction_colors.get(d,"#9CA3AF")
    axD.scatter(x,y,s=90+70*n,c=c,edgecolors="white",linewidths=0.9,alpha=0.90,zorder=3)
    label=str(r["Disease"])
    display={"ARDS / acute lung injury":"ARDS / ALI","Cancer-associated inflammation":"Cancer inflammation"}.get(label,label)
    label_positions={"Sepsis":(4.55,4.78),"ARDS / acute lung injury":(3.25,3.58),"COVID-19":(4.15,3.35),"Acute pancreatitis":(1.35,4.68),"Ulcerative colitis":(2.70,2.95),"Cancer-associated inflammation":(3.05,5.10)}
    lx,ly=label_positions.get(label,(x+0.15,y+0.15))
    axD.annotate(textwrap.fill(display,18),(x,y),xytext=(lx,ly),textcoords="data",fontsize=7.0,ha="left",va="center",
                 arrowprops=dict(arrowstyle="-",lw=0.55,color="#6B7280",shrinkA=2,shrinkB=5))
axD.set_xlim(0,5.35); axD.set_ylim(0,5.35)
axD.set_xlabel("Mechanistic evidence depth (ordinal 0–5)")
axD.set_ylabel("Clinical evidence depth (ordinal 0–5)")
axD.grid(True,color="#E5E7EB",linewidth=0.6,zorder=0)
legend=[Line2D([0],[0],marker="o",color="none",markerfacecolor=c,markeredgecolor="white",markersize=7,label=k) for k,c in direction_colors.items()]
axD.legend(handles=legend,loc="lower left",frameon=False,ncol=2,handletextpad=0.3,columnspacing=0.8)
axD.text(0,-0.30,"Bubble area = number of supporting\nhuman/clinical references. Ordinal mapping\nis review-derived, not GRADE or meta-analysis.",transform=axD.transAxes,fontsize=7.0,ha="left",va="top")

fig.suptitle("Figure 9. Evidence-informed knowledge graph of heparin anti-inflammatory pharmacology and translational evidence",fontsize=8.8,fontweight="bold",y=0.995)
fig.savefig(OUT_SVG,format="svg")
fig.savefig(OUT_PDF,format="pdf")
fig.savefig(OUT_PNG,format="png",dpi=600,pil_kwargs={"compress_level":6})
plt.close(fig)
shutil.rmtree(_mpl_cache, ignore_errors=True)

print({"nodes":G.number_of_nodes(),"edges":G.number_of_edges(),"panel_b_nodes":len(connected),"panel_b_edges":len(core_edges),"seed":SEED})
