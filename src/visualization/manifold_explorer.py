"""
Zero-dependency interactive 3D manifold explorer (single self-contained HTML file).

Renders tens of thousands of paintings as a rotatable 3D point cloud on an HTML5 canvas
(no CDN, works offline): drag to rotate, wheel to zoom, hover for metadata, filter by
cluster / museum / century / outlier status, and free-text search.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>ArtStylometry — Collection Manifold Explorer</title>
<style>
 body{margin:0;background:#0b1120;color:#e2e8f0;font-family:Inter,Segoe UI,sans-serif;overflow:hidden}
 #ui{position:absolute;top:10px;left:10px;background:rgba(15,23,42,.92);padding:12px 14px;border-radius:10px;border:1px solid #334155;width:290px;font-size:12px;z-index:2}
 #ui h1{font-size:14px;margin:0 0 6px}#ui select,#ui input{width:100%;margin:3px 0 7px;background:#1e293b;color:#e2e8f0;border:1px solid #475569;border-radius:5px;padding:4px}
 #stats{color:#94a3b8;line-height:1.5}#tip{position:absolute;pointer-events:none;background:rgba(2,6,23,.95);border:1px solid #38bdf8;border-radius:8px;padding:8px;font-size:12px;display:none;max-width:280px;z-index:3}
 #tip img{max-width:260px;max-height:180px;display:block;margin-top:5px;border-radius:4px}
 .warn{color:#fbbf24}
</style></head><body>
<div id="ui"><h1>🌌 Collection Manifold (3D)</h1>
<div id="stats"></div>
<label>Colour by</label><select id="colorBy"><option value="cluster">HDBSCAN cluster</option><option value="century">Century</option><option value="museum">Museum</option></select>
<label>Museum</label><select id="museum"><option value="">All</option></select>
<label>Century</label><select id="century"><option value="">All</option></select>
<label><input type="checkbox" id="outliersOnly" style="width:auto"> Outliers (noise) only</label><br>
<label>Search title / artist</label><input id="q" placeholder="e.g. Rembrandt">
<div style="color:#64748b">Drag = rotate · Wheel = zoom · Hover = details</div></div>
<div id="tip"></div><canvas id="c"></canvas>
<script>
const DATA=__DATA__; const SUMMARY=__SUMMARY__;
const cols=DATA.cols, rows=DATA.rows, I={}; cols.forEach((c,i)=>I[c]=i);
const cv=document.getElementById('c'),ctx=cv.getContext('2d');let W,H;function rs(){W=cv.width=innerWidth;H=cv.height=innerHeight;draw()}addEventListener('resize',rs);
const pal=n=>`hsl(${(n*137.508)%360},70%,60%)`;
const uniq=k=>[...new Set(rows.map(r=>r[I[k]]))].sort();
const museums=uniq('museum'),cents=uniq('century').filter(c=>c>0);
museums.forEach(m=>museum.add(new Option(m,m)));cents.forEach(c=>century.add(new Option(c+'th c.',c)));
const mIdx=Object.fromEntries(museums.map((m,i)=>[m,i]));
// normalise coordinates
['x3','y3','z3'].forEach(k=>{const v=rows.map(r=>r[I[k]]),mn=Math.min(...v),mx=Math.max(...v);rows.forEach(r=>r[I[k]]=(r[I[k]]-(mn+mx)/2)/((mx-mn)||1)*2)});
let ax=0.4,ay=0.6,zoom=1,drag=null,proj=[]; window.proj=proj;
function color(r){const cb=colorBy.value;if(cb==='century')return r[I.century]?`hsl(${(r[I.century]-12)*40},75%,58%)`:'#475569';
 if(cb==='museum')return pal(mIdx[r[I.museum]]);return r[I.cluster_id]<0?'rgba(244,63,94,.85)':pal(r[I.cluster_id])}
function visible(r){if(museum.value&&r[I.museum]!==museum.value)return false;if(century.value&&String(r[I.century])!==century.value)return false;
 if(outliersOnly.checked&&r[I.cluster_id]>=0)return false;const q=document.getElementById('q').value.toLowerCase();
 if(q&&!(r[I.title]+' '+r[I.artist]).toLowerCase().includes(q))return false;return true}
function draw(){if(!W)return;ctx.fillStyle='#0b1120';ctx.fillRect(0,0,W,H);const s=Math.min(W,H)*0.38*zoom,ca=Math.cos(ax),sa=Math.sin(ax),cb=Math.cos(ay),sb=Math.sin(ay);
 proj=[]; window.proj=proj; let shown=0,out=0;for(const r of rows){if(!visible(r))continue;shown++;if(r[I.cluster_id]<0)out++;
  let x=r[I.x3],y=r[I.y3],z=r[I.z3];let x1=x*cb+z*sb,z1=-x*sb+z*cb;let y1=y*ca-z1*sa,z2=y*ca*0+y*sa+z1*ca;
  const p=1/(1+z2*0.35);proj.push([W/2+x1*s*p,H/2+y1*s*p,z2,r])}
 proj.sort((a,b)=>b[2]-a[2]);for(const [x,y,z,r] of proj){ctx.fillStyle=color(r);ctx.globalAlpha=.55+.45*(1-(z+1)/2);ctx.fillRect(x-1.6,y-1.6,3.2,3.2)}ctx.globalAlpha=1;
 stats.innerHTML=`<b>${shown.toLocaleString()}</b> paintings shown · ${out.toLocaleString()} outliers<br>${SUMMARY.n_clusters} clusters · noise ${(100*SUMMARY.noise_fraction).toFixed(1)}% · ${SUMMARY.projection.toUpperCase()}<br>
 AMI(cluster, museum) = <span class="${SUMMARY.ami_cluster_vs_museum>0.3?'warn':''}">${SUMMARY.ami_cluster_vs_museum}</span><br>AMI(cluster, century) = ${SUMMARY.ami_cluster_vs_century}`}
cv.onmousedown=e=>drag=[e.clientX,e.clientY];onmouseup=()=>drag=null;
const resolvePath=p=>{if(!p)return '';if(location.protocol.startsWith('http')){return p.replace(/^[a-zA-Z]:\/.*?\/ArtStylometry\//,'/').replace(/^\/?/,'/')}return p.startsWith('file:///')?p:('file:///'+p)};
cv.onmousemove=e=>{if(drag){ay+=(e.clientX-drag[0])*0.008;ax+=(e.clientY-drag[1])*0.008;drag=[e.clientX,e.clientY];draw();tip.style.display='none';return}
 let best=null,bd=36;for(const p of proj){const d=(p[0]-e.clientX)**2+(p[1]-e.clientY)**2;if(d<bd){bd=d;best=p}}
 if(!best){tip.style.display='none';return}const r=best[3];tip.style.display='block';tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY+14)+'px';
 tip.innerHTML=`<b>${r[I.title]}</b><br>${r[I.artist]} · ${r[I.creation_date]}<br>${r[I.museum]}<br>Cluster: ${r[I.cluster_id]<0?'<span style="color:#f43f5e">outlier</span>':r[I.cluster_id]} (p=${r[I.membership_probability]})<img src="${resolvePath(r[I.local_path])}" onerror="this.remove()">`};
cv.onwheel=e=>{zoom*=e.deltaY<0?1.1:0.9;draw()};[colorBy,museum,century,outliersOnly].forEach(el=>el.onchange=draw);q.oninput=draw;rs();
</script></body></html>"""


def export_manifold_explorer(df: pd.DataFrame, output_path: str | Path, summary: Dict[str, Any]) -> Path:
    """Writes the standalone explorer. ``df`` needs x3/y3/z3, cluster_id and metadata columns."""
    cols = ["title", "artist", "creation_date", "century", "museum", "cluster_id",
            "membership_probability", "x3", "y3", "z3", "local_path"]
    sub = df[cols].copy()
    sub["local_path"] = sub["local_path"].astype(str).str.replace("\\", "/", regex=False)
    payload = {"cols": cols, "rows": sub.values.tolist()}
    html = _TEMPLATE.replace("__DATA__", json.dumps(payload, ensure_ascii=False, default=str)).replace(
        "__SUMMARY__", json.dumps(summary, default=str))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
