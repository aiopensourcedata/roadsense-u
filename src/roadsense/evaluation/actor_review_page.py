# SPDX-License-Identifier: AGPL-3.0-only
"""Offline score-blind actor annotation page, with explicit JSON export only."""

import json


def render_actor_page(manifest: dict, manifest_sha256: str) -> str:
    """Embed only the blind frame manifest; predictions and outcomes are absent."""
    payload = json.dumps({"manifest": manifest, "sha256": manifest_sha256}).replace("<", "\\u003c")
    return PAGE.replace("__BLIND_DATA__", payload)


PAGE = r"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RoadSense-U · Actor review</title>
<style>
body{font:16px system-ui;background:#111827;color:#e5e7eb;margin:24px auto;max-width:1350px;padding:0 20px}
h1{margin-bottom:8px}p{line-height:1.5}button,select,input{font:inherit;padding:8px;margin:4px;border-radius:6px}
button{cursor:pointer;background:#dbeafe;color:#111827;border:0}label{display:inline-block;margin:5px}
canvas{width:100%;height:auto;touch-action:none;border:1px solid #64748b;background:#1e293b}
.notice{background:#26354c;padding:12px;border-radius:8px}.row{display:flex;flex-wrap:wrap;align-items:center}
li{margin:8px 0}#status{min-height:24px;color:#fbbf24}a{color:#93c5fd}
</style>
<h1>Actor review</h1><p class="notice">Development footage · Independent visual annotation.
Model boxes, scores and collision outcomes are hidden. Do not open predictions.json while annotating.
No results are saved automatically or uploaded. Export JSON before closing.</p>
<p>Draw the visible extent of every supported road user. Include small/partial objects when discernible.
Use the same temporary actor ID across frames of a clip. IDs describe local tracks, not people’s identities.
Mark uncertain objects explicitly. A depicted/reflected object is not a physical road user.</p>
<div class="row"><label>Reviewer ID <input id="reviewer" autocomplete="off"></label>
<button id="export">Export review JSON</button><label>Resume JSON <input id="import" type="file" accept=".json"></label></div>
<div class="row"><button id="prev">Previous</button><label>Frame <select id="frame"></select></label><button id="next">Next</button>
<label><input id="reviewed" type="checkbox">Entire frame reviewed (including empty frames)</label></div>
<p id="meta"></p><div class="row">
<label>Actor ID <input id="actor" placeholder="e.g. car-A"></label>
<label>Class <select id="class"><option>car</option><option>motorcycle</option><option>bus</option><option>truck</option><option>bicycle</option><option>person</option><option>unknown</option></select></label>
<label>Kind <select id="kind"><option>physical</option><option>depiction</option><option>uncertain</option></select></label>
<label>Visibility <select id="visibility"><option>clear</option><option>partial</option><option>uncertain</option></select></label>
</div><p id="status" role="status"></p><canvas id="canvas" aria-label="Draw road-user boxes" width="1280" height="720"></canvas>
<ol id="boxes"></ol><p>Boxes use normalized original-image coordinates. Annotation quality, actor association and adjudication require human review.</p>
<script id="blind-data" type="application/json">__BLIND_DATA__</script><script>
'use strict';
const payload=JSON.parse(document.getElementById('blind-data').textContent);
const manifest=payload.manifest, $=id=>document.getElementById(id);
let data={schema_version:1,manifest_sha256:payload.sha256,review_stage:'first_pass',reviewer_ids:[],
adjudicator_id:null,review_method:'human_visual',frames:manifest.frames.map(f=>({frame_key:f.frame_key,reviewed:false,actors:[]}))};
let index=0, start=null, draft=null, image=new Image(), imageReady=false;
const canvas=$('canvas'), ctx=canvas.getContext('2d');
manifest.frames.forEach((f,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${i+1}. ${f.frame_key}`;$('frame').append(o)});
function message(text){$('status').textContent=text}
function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);if(imageReady)ctx.drawImage(image,0,0);
 const boxes=data.frames[index].actors.map(a=>a.bbox);if(draft)boxes.push(draft);
 boxes.forEach((b,i)=>{ctx.strokeStyle=i===boxes.length-1&&draft?'#fbbf24':'#22d3ee';ctx.lineWidth=3;
 ctx.strokeRect(b[0]*canvas.width,b[1]*canvas.height,(b[2]-b[0])*canvas.width,(b[3]-b[1])*canvas.height);
 if(data.frames[index].actors[i]){ctx.fillStyle='#22d3ee';ctx.font='20px sans-serif';ctx.fillText(data.frames[index].actors[i].actor_id,b[0]*canvas.width,Math.max(22,b[1]*canvas.height-5))}})}
function list(){const target=$('boxes');target.replaceChildren();data.frames[index].actors.forEach((a,i)=>{
 const li=document.createElement('li');li.textContent=`${a.actor_id} · ${a.class_name} · ${a.kind} · ${a.visibility} `;
 const button=document.createElement('button');button.textContent='Remove';button.onclick=()=>{data.frames[index].actors.splice(i,1);data.frames[index].reviewed=false;$('reviewed').checked=false;list();draw()};li.append(button);target.append(li)})}
function show(){imageReady=false;start=null;draft=null;message('');$('frame').value=index;const f=manifest.frames[index];
 $('meta').textContent=`${index+1}/${manifest.frames.length} · clip ${f.clip_index} · source frame ${f.frame_index} · ${f.timestamp_s.toFixed(3)} media seconds`;
 $('reviewed').checked=data.frames[index].reviewed;image=new Image();const requested=index;
 image.onload=()=>{if(requested!==index)return;canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;imageReady=true;draw()};
 image.onerror=()=>{if(requested===index)message('No review image. Regenerate with --write-review-frames to annotate.')};
 if(f.image)image.src=f.image;else message('Images were not retained. Explicit frame export is required for this page.');list();draw()}
function point(e){const r=canvas.getBoundingClientRect();return [Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))]}
function box(a,b){return [Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])]}
canvas.onpointerdown=e=>{if(!imageReady)return;if(!$('actor').value.trim()){message('Enter a temporary actor ID first.');return}start=point(e);canvas.setPointerCapture(e.pointerId)};
canvas.onpointermove=e=>{if(start){draft=box(start,point(e));draw()}};
canvas.onpointerup=e=>{if(!start)return;const b=box(start,point(e));start=null;draft=null;
 const actor_id=$('actor').value.trim();if(data.frames[index].actors.some(a=>a.actor_id===actor_id)){message('This actor ID already has a box in the frame. Remove that box to redraw.');draw();return}
 if(b[2]-b[0]<1/canvas.width||b[3]-b[1]<1/canvas.height){message('Box is too small.');draw();return}
 data.frames[index].actors.push({actor_id,class_name:$('class').value,bbox:b,kind:$('kind').value,visibility:$('visibility').value});
 data.frames[index].reviewed=false;$('reviewed').checked=false;message('Box added. Review the entire frame before marking complete.');list();draw()};
canvas.onpointercancel=()=>{start=null;draft=null;draw()};
$('reviewed').onchange=()=>{data.frames[index].reviewed=$('reviewed').checked};
$('frame').onchange=()=>{index=Number($('frame').value);show()};
$('prev').onclick=()=>{index=Math.max(0,index-1);show()};$('next').onclick=()=>{index=Math.min(manifest.frames.length-1,index+1);show()};
$('export').onclick=()=>{const reviewer=$('reviewer').value.trim();if(!reviewer){message('Reviewer ID is required.');return}
 data.reviewer_ids=[reviewer];const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
 const link=document.createElement('a');link.href=url;link.download='actor-review.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 message(`Exported ${data.frames.filter(f=>f.reviewed).length}/${data.frames.length} reviewed frames. Partial exports are drafts, not evaluation truth.`)};
$('import').onchange=async e=>{try{const incoming=JSON.parse(await e.target.files[0].text());
 if(incoming.manifest_sha256!==payload.sha256||incoming.review_stage!=='first_pass'||incoming.review_method!=='human_visual'||incoming.frames.length!==data.frames.length)throw Error('Wrong bundle or review type');
 if(incoming.frames.some((f,i)=>f.frame_key!==manifest.frames[i].frame_key||typeof f.reviewed!=='boolean'||!Array.isArray(f.actors)||f.actors.some(a=>!Array.isArray(a.bbox)||a.bbox.length!==4||a.bbox.some(v=>!Number.isFinite(v)||v<0||v>1)||a.bbox[0]>=a.bbox[2]||a.bbox[1]>=a.bbox[3])))throw Error('Invalid frame or box');
 data=incoming;$('reviewer').value=incoming.reviewer_ids[0]||'';show();message('Review resumed locally.');
 }catch(error){message('Cannot import: '+error.message)}};
show();
</script></html>"""
