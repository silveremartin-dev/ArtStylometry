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
 #ui{position:absolute;top:10px;left:10px;background:rgba(15,23,42,.92);padding:12px 14px;border-radius:10px;border:1px solid #334155;width:300px;font-size:12px;z-index:2;box-shadow:0 8px 25px rgba(0,0,0,.6)}
 #ui h1{font-size:14px;margin:0 0 6px;color:#38bdf8}#ui select,#ui input{width:100%;margin:3px 0 7px;background:#1e293b;color:#e2e8f0;border:1px solid #475569;border-radius:5px;padding:5px}
 #stats{color:#94a3b8;line-height:1.5;margin-bottom:6px}#tip{position:absolute;pointer-events:none;background:rgba(9,15,30,.96);border:1px solid #38bdf8;border-radius:10px;padding:10px;font-size:12px;display:none;width:270px;z-index:3;box-shadow:0 10px 30px rgba(0,0,0,.85)}
 #tip .tip-img-wrap{width:100%;height:175px;background:#030712;border-radius:6px;margin-top:6px;display:flex;align-items:center;justify-content:center;overflow:hidden;border:1px solid #1e293b;padding:4px;box-sizing:border-box}
 #tip img{width:100%;height:100%;object-fit:contain;display:block}
 .warn{color:#fbbf24}
 #mModal{position:fixed;inset:0;background:rgba(2,6,23,.92);backdrop-filter:blur(8px);z-index:99;display:none;align-items:center;justify-content:center;padding:24px}
 .mm-box{background:#0f172a;border:1px solid #334155;border-radius:12px;max-width:1200px;width:92vw;height:86vh;display:flex;overflow:hidden;box-shadow:0 20px 60px rgba(0,0,0,.9)}
 .mm-img{flex:1;background:#030712;display:flex;align-items:center;justify-content:center;padding:16px}
 .mm-img img{max-width:100%;max-height:100%;object-fit:contain;border-radius:4px}
 .mm-side{width:340px;padding:20px;border-left:1px solid #1e293b;display:flex;flex-direction:column;gap:10px}
</style></head><body>
<div id="ui"><h1>🌌 Collection Manifold (3D)</h1>
<div id="stats"></div>
<label title="Choose how points are colored in 3D space">Colour by</label><select id="colorBy"><option value="cluster">HDBSCAN Stylistic Cluster</option><option value="century">Historical Century</option><option value="museum">Holding Museum</option></select>
<label>Museum</label><select id="museum"><option value="">All Museums</option></select>
<label>Century</label><select id="century"><option value="">All Centuries</option></select>
<label title="Show only paintings whose brushwork falls outside any dense stylistic cluster"><input type="checkbox" id="outliersOnly" style="width:auto"> Statistical Outliers (Noise) Only</label><br>
<label>Search title / artist</label><input id="q" placeholder="e.g. Rembrandt, Vermeer, Goya...">
<div style="color:#64748b;margin-top:4px">Drag = rotate · Wheel = zoom · Hover = preview · <b>Click point = Full-Screen</b></div></div>
<div id="tip"></div><canvas id="c"></canvas>
<div id="mModal" onclick="if(event.target===this)this.style.display='none'">
 <div class="mm-box" onclick="event.stopPropagation()">
  <div class="mm-img"><img id="mmPic" src="" alt=""></div>
  <div class="mm-side">
   <div style="display:flex;justify-content:space-between;align-items:start"><h2 id="mmTitle" style="margin:0;font-size:17px;color:#fff"></h2><button onclick="document.getElementById('mModal').style.display='none'" style="background:#1e293b;border:1px solid #475569;color:#fff;border-radius:6px;padding:4px 9px;cursor:pointer">✕</button></div>
   <div id="mmArtist" style="color:#38bdf8;font-weight:700;font-size:14px"></div>
   <div id="mmMuseum" style="color:#94a3b8;font-size:13px"></div>
   <div id="mmCluster" style="background:#162035;border:1px solid #263554;padding:10px;border-radius:8px;font-size:12px;line-height:1.5"></div>
   <a id="mmRaw" href="#" target="_blank" style="margin-top:auto;background:#0284c7;color:#fff;text-decoration:none;text-align:center;padding:9px;border-radius:6px;font-weight:700;font-size:13px">🔍 Open Full-Resolution File</a>
  </div>
 </div>
</div>
<script>
const DATA=__DATA__; const SUMMARY=__SUMMARY__;
const cols=DATA.cols, rows=DATA.rows, I={}; cols.forEach((c,i)=>I[c]=i);
const cv=document.getElementById('c'),ctx=cv.getContext('2d');let W,H;function rs(){W=cv.width=innerWidth;H=cv.height=innerHeight;draw()}addEventListener('resize',rs);
const pal=n=>`hsl(${(n*137.508)%360},72%,60%)`;
const uniq=k=>[...new Set(rows.map(r=>r[I[k]]))].sort();
const museums=uniq('museum'),cents=uniq('century').filter(c=>c>0);
museums.forEach(m=>museum.add(new Option(m,m)));cents.forEach(c=>century.add(new Option(c+'th c.',c)));
const mIdx=Object.fromEntries(museums.map((m,i)=>[m,i]));
// normalise coordinates
['x3','y3','z3'].forEach(k=>{const v=rows.map(r=>r[I[k]]),mn=Math.min(...v),mx=Math.max(...v);rows.forEach(r=>r[I[k]]=(r[I[k]]-(mn+mx)/2)/((mx-mn)||1)*2)});
let ax=0.4,ay=0.6,zoom=1,drag=null,moved=false,hovered=null,proj=[]; window.proj=proj;
function color(r){const cb=colorBy.value;if(cb==='century')return r[I.century]?`hsl(${(r[I.century]-12)*40},75%,58%)`:'#475569';
 if(cb==='museum')return pal(mIdx[r[I.museum]]);return r[I.cluster_id]<0?'rgba(244,63,94,.75)':pal(r[I.cluster_id])}
function visible(r){if(museum.value&&r[I.museum]!==museum.value)return false;if(century.value&&String(r[I.century])!==century.value)return false;
 if(outliersOnly.checked&&r[I.cluster_id]>=0)return false;const q=document.getElementById('q').value.toLowerCase();
 if(q&&!(r[I.title]+' '+r[I.artist]).toLowerCase().includes(q))return false;return true}
function draw(){if(!W)return;ctx.fillStyle='#0b1120';ctx.fillRect(0,0,W,H);const s=Math.min(W,H)*0.38*zoom,ca=Math.cos(ax),sa=Math.sin(ax),cb=Math.cos(ay),sb=Math.sin(ay);
 proj=[]; window.proj=proj; let shown=0,out=0;for(const r of rows){if(!visible(r))continue;shown++;if(r[I.cluster_id]<0)out++;
  let x=r[I.x3],y=r[I.y3],z=r[I.z3];let x1=x*cb+z*sb,z1=-x*sb+z*cb;let y1=y*ca-z1*sa,z2=y*ca*0+y*sa+z1*ca;
  const p=1/(1+z2*0.35);proj.push([W/2+x1*s*p,H/2+y1*s*p,z2,r])}
 proj.sort((a,b)=>b[2]-a[2]);for(const [x,y,z,r] of proj){ctx.fillStyle=color(r);ctx.globalAlpha=.6+.4*(1-(z+1)/2);ctx.fillRect(x-1.8,y-1.8,3.6,3.6)}ctx.globalAlpha=1;
 stats.innerHTML=`<b>${shown.toLocaleString()}</b> paintings shown · <b>${(shown-out).toLocaleString()}</b> clustered · ${out.toLocaleString()} outliers<br>${SUMMARY.n_clusters} stylistic clusters · noise ${(100*SUMMARY.noise_fraction).toFixed(1)}% · ${SUMMARY.projection.toUpperCase()}<br>
 AMI(cluster, museum) = <span class="${SUMMARY.ami_cluster_vs_museum>0.3?'warn':''}">${SUMMARY.ami_cluster_vs_museum}</span><br>AMI(cluster, century) = ${SUMMARY.ami_cluster_vs_century}`}
cv.onmousedown=e=>{drag=[e.clientX,e.clientY];moved=false};
onmouseup=e=>{if(!moved&&hovered){openPointModal(hovered)}drag=null};
const resolvePath=p=>{
 if(!p)return '';
 let clean=String(p).replace(/\\/g,'/').replace(/^[a-zA-Z]:\/.*?\/ArtStylometry\//,'').replace(/^file:\/\/\//,'').replace(/^\/+/,'');
 if(clean.startsWith('data/')) clean='../'+clean.slice(5);
 return clean;
};
function openPointModal(r){
 const src=resolvePath(r[I.local_path]);
 document.getElementById('mmPic').src=src;
 document.getElementById('mmTitle').textContent=r[I.title];
 document.getElementById('mmArtist').textContent=r[I.artist]+(r[I.creation_date]?' ('+r[I.creation_date]+')':'');
 document.getElementById('mmMuseum').textContent='🏛️ '+r[I.museum];
 document.getElementById('mmCluster').innerHTML=r[I.cluster_id]<0
  ? '<b style="color:#f43f5e">⚠️ Statistical Manifold Outlier</b><br>This painting lies in a low-density region outside the main HDBSCAN stylistic clusters, indicating atypical facture or composition.'
  : `<b style="color:#10b981">✅ Stylistic Cluster #${r[I.cluster_id]}</b> (Confidence: ${(Number(r[I.membership_probability])*100).toFixed(1)}%)<br>Grouped by unsupervised micro-facture and structure tensor similarity.`;
 document.getElementById('mmRaw').href=src;
 document.getElementById('mModal').style.display='flex';
}
cv.onmousemove=e=>{if(drag){if(Math.hypot(e.clientX-drag[0],e.clientY-drag[1])>3)moved=true;ay+=(e.clientX-drag[0])*0.008;ax+=(e.clientY-drag[1])*0.008;drag=[e.clientX,e.clientY];draw();tip.style.display='none';return}
 let best=null,bd=49;for(const p of proj){const d=(p[0]-e.clientX)**2+(p[1]-e.clientY)**2;if(d<bd){bd=d;best=p}}
 if(!best){hovered=null;tip.style.display='none';cv.style.cursor='grab';return}
 cv.style.cursor='pointer';const r=best[3];hovered=r;tip.style.display='block';
 const tx=Math.min(W-290,e.clientX+16),ty=Math.min(H-280,e.clientY+16);
 tip.style.left=Math.max(10,tx)+'px';tip.style.top=Math.max(10,ty)+'px';
 const imgUrl=resolvePath(r[I.local_path]);
 tip.innerHTML=`<div style="font-weight:700;color:#fff;margin-bottom:2px">${r[I.title]}</div><div style="color:#38bdf8;font-weight:600">${r[I.artist]} (${r[I.creation_date]})</div><div style="color:#94a3b8;font-size:11px">🏛️ ${r[I.museum]}</div><div style="margin-top:3px;font-size:11px">Status: ${r[I.cluster_id]<0?'<b style="color:#f43f5e">⚠️ Manifold Outlier</b>':`<b style="color:#10b981">Cluster #${r[I.cluster_id]}</b> (p=${Number(r[I.membership_probability]).toFixed(2)})`}</div><div class="tip-img-wrap"><img src="${imgUrl}" alt="" onerror="this.parentElement.style.display='none'"></div><div style="font-size:10px;color:#64748b;margin-top:4px;text-align:center">🔍 Click point to inspect in Full-Screen</div>`};
cv.onwheel=e=>{e.preventDefault();zoom*=e.deltaY<0?1.12:0.89;draw()};[colorBy,museum,century,outliersOnly].forEach(el=>el.onchange=draw);q.oninput=draw;rs();
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
