/* Adaptive Dual-Phase PSO-GWO Forecasting Lab — dashboard controller.
   Renders the full analytics suite from the benchmark result payload. */
"use strict";

const FLAGSHIP_INDEX = 15;
const state = { cacheKey:null, jobId:null, poll:null, final:null, models:[], selected:FLAGSHIP_INDEX, sortKey:"rmse", sortDir:1 };

/* ---------- theme ---------- */
function cssVar(n){ return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
function isDark(){ return document.documentElement.getAttribute("data-theme")==="dark"; }
function tpl(){
  const text=cssVar("--text"), muted=cssVar("--muted"), line=cssVar("--line");
  const grid = isDark()? "rgba(255,255,255,.07)" : "rgba(20,30,60,.08)";
  return {
    paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:"rgba(0,0,0,0)",
    font:{color:text, family:"-apple-system,Segoe UI,Roboto,sans-serif", size:12},
    margin:{l:54,r:18,t:38,b:42}, hovermode:"x unified",
    legend:{orientation:"h", y:-0.18, font:{size:11}},
    xaxis:{gridcolor:grid, zerolinecolor:line, linecolor:line},
    yaxis:{gridcolor:grid, zerolinecolor:line, linecolor:line},
    colorway:["#4f8cff","#22d3ee","#34d399","#fbbf24","#f87171","#a855f7","#f472b6","#60a5fa"]
  };
}
const COLORS = { actual:"#0ea5e9", pred:"#f87171", train:"#34d399", val:"#fbbf24",
  forecast:"#a855f7", band:"rgba(168,85,247,.18)", flagship:"#a855f7" };
const GROUP_COLOR = { "Baseline":"#64748b","PSO":"#4f8cff","GWO":"#22d3ee",
  "Hybrid PSO-GWO":"#34d399","Adaptive Dual-Phase":"#fbbf24","Ensemble":"#a855f7" };

function plot(el, traces, layout, extra){
  const base = tpl();
  const merged = Object.assign({}, base, layout||{});
  merged.xaxis = Object.assign({}, base.xaxis, (layout&&layout.xaxis)||{});
  merged.yaxis = Object.assign({}, base.yaxis, (layout&&layout.yaxis)||{});
  const config = { responsive:true, displaylogo:false,
    modeBarButtonsToAdd:[{ name:"Download SVG", icon:Plotly.Icons.disk,
      click:gd=>Plotly.downloadImage(gd,{format:"svg",filename:"chart"}) }],
    toImageButtonOptions:{ format:"png", scale:2, filename:"chart" } };
  Plotly.react(el, traces, merged, config);
}

/* ---------- small utils ---------- */
const $=id=>document.getElementById(id);
const fmt=(x,d=3)=> (x===null||x===undefined||isNaN(x))? "—" : Number(x).toFixed(d);
const pct=(x,d=1)=> (x===null||x===undefined||isNaN(x))? "—" : Number(x).toFixed(d)+"%";
function toast(msg,kind=""){ const t=$("toast"); t.textContent=msg; t.className="toast show "+kind; setTimeout(()=>t.className="toast",3200); }
function mean(a){ return a.reduce((s,x)=>s+x,0)/(a.length||1); }
function std(a){ const m=mean(a); return Math.sqrt(mean(a.map(x=>(x-m)**2))||0); }
function normInv(p){ // Acklam inverse normal CDF
  const a=[-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00];
  const b=[-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01];
  const c=[-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00];
  const d=[7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00];
  const pl=0.02425,ph=1-pl; let q,r;
  if(p<pl){q=Math.sqrt(-2*Math.log(p));return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);}
  if(p<=ph){q=p-0.5;r=q*q;return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);}
  q=Math.sqrt(-2*Math.log(1-p));return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
}

/* ---------- API ---------- */
async function api(url, opts){ const r=await fetch(url,opts); const j=await r.json(); if(!r.ok) throw new Error(j.error||("HTTP "+r.status)); return j; }

async function health(){
  try{ const h=await api("/api/health");
    $("redis-dot").className="dot "+(h.redis?"on":"err"); $("redis-text").textContent=h.redis?"Redis connected · "+h.models+" models":"Redis offline";
  }catch(e){ $("redis-dot").className="dot err"; $("redis-text").textContent="API offline"; }
}

async function fetchData(){
  const btn=$("btn-fetch"); btn.disabled=true; btn.textContent="…";
  try{
    const d=await api("/api/fetch_data",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({symbol:$("symbol").value,start:$("start").value})});
    onData(d);
  }catch(e){ toast(e.message,"err"); } finally{ btn.disabled=false; btn.textContent="Fetch"; }
}
async function uploadData(){
  const f=$("csv").files[0]; if(!f){ toast("Choose a CSV first","err"); return; }
  const fd=new FormData(); fd.append("file",f);
  try{ const d=await api("/api/upload_data",{method:"POST",body:fd}); onData(d); }
  catch(e){ toast(e.message,"err"); }
}
function onData(d){
  state.cacheKey=d.cache_key; const s=d.summary;
  $("d-sym").textContent=d.symbol; $("d-rows").textContent=s.rows; $("d-feat").textContent=s.features;
  $("d-from").textContent=s.start; $("d-to").textContent=s.end; $("d-close").textContent=fmt(s.lastClose,2);
  $("btn-run").disabled=false; $("prog-text").textContent="Data ready — "+s.rows+" rows, "+s.features+" features.";
  toast("Loaded "+d.symbol+" ("+s.rows+" rows)","ok");
}

function cfgFromForm(){
  return { popSize:+$("popSize").value, maxIter:+$("maxIter").value, c1:+$("c1").value, c2:+$("c2").value,
    w:+$("w").value, seqLen:+$("seqLen").value, testPct:+$("testPct").value, seed:+$("seed").value, effort:$("effort").value };
}

async function runModels(){
  if(!state.cacheKey){ toast("Fetch or upload data first","err"); return; }
  $("btn-run").disabled=true; $("live-card").classList.remove("hidden"); $("results").classList.add("hidden");
  initBoard();
  try{
    const d=await api("/api/run",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({cache_key:state.cacheKey,cfg:cfgFromForm()})});
    state.jobId=d.job_id; state.poll=setInterval(poll,1400); poll();
  }catch(e){ toast(e.message,"err"); $("btn-run").disabled=false; }
}

async function poll(){
  if(!state.jobId) return;
  let s; try{ s=await api("/api/status/"+state.jobId); }catch(e){ return; }
  $("prog-bar").style.width=Math.round((s.progress||0)*100)+"%";
  $("stage").textContent=s.stage||"running";
  $("prog-text").textContent=(s.stage||"")+" · "+Math.round((s.progress||0)*100)+"%";
  updateBoard(s);
  if(s.log){ const L=$("log"); L.innerHTML=s.log.map(x=>"› "+x.msg).join("<br>"); L.scrollTop=L.scrollHeight; }
  if(s.status==="done"){ clearInterval(state.poll); state.poll=null; await loadResults(); }
  else if(s.status==="error"){ clearInterval(state.poll); state.poll=null; $("btn-run").disabled=false; toast("Run failed: "+(s.error||"see log"),"err"); }
}

async function loadResults(){
  const r=await api("/api/results/"+state.jobId);
  state.final=r.final; state.models=r.results||[]; $("btn-run").disabled=false;
  if(!state.final){ toast("No results produced","err"); return; }
  state.selected = (state.models.find(m=>m.index===FLAGSHIP_INDEX)?FLAGSHIP_INDEX:(state.final.leaderboard[0]?.index??0));
  toast("Benchmark complete — "+state.models.length+" models","ok");
  renderResults();
}

/* ---------- live board ---------- */
const MODEL_NAMES=["Base LSTM","Base ELM","Base GBDT","LSTM + PSO","ELM + PSO","GBDT + PSO","LSTM + GWO","ELM + GWO","GBDT + GWO","LSTM + PSO-GWO","ELM + PSO-GWO","GBDT + PSO-GWO","Adaptive Dual-Phase PSO-GWO + LSTM","Adaptive Dual-Phase PSO-GWO + ELM","Adaptive Dual-Phase PSO-GWO + GBDT","Adaptive Dual-Phase PSO-GWO Expert Fusion Ensemble"];
function initBoard(){
  $("board").innerHTML=MODEL_NAMES.map((n,i)=>`<div class="mc ${i===FLAGSHIP_INDEX?'flagship':''}" id="mc-${i}">
    <div class="nm">${i+1}. ${n}</div><div class="st" id="st-${i}">Queued</div>
    <div class="mt"><span id="mr-${i}" class="muted mono"></span></div><div class="bar" id="bar-${i}" style="width:0"></div></div>`).join("");
}
function updateBoard(s){
  const st=s.model_status||{}; const sum=s.summary||[];
  MODEL_NAMES.forEach((n,i)=>{ const el=$("mc-"+i); if(!el)return; const stat=st[n];
    el.classList.remove("done","running","error");
    if(stat==="done")el.classList.add("done"); else if(stat==="running")el.classList.add("running"); else if(stat==="error")el.classList.add("error");
    if($("st-"+i)) $("st-"+i).textContent = stat? stat[0].toUpperCase()+stat.slice(1) : "Queued";
    if(stat==="running")$("bar-"+i).style.width="60%"; if(stat==="done")$("bar-"+i).style.width="100%";
  });
  sum.forEach(r=>{ if($("mr-"+r.index)) $("mr-"+r.index).textContent="RMSE "+fmt(r.rmse,3)+" · R² "+fmt(r.r2,2); });
}

/* ---------- results shell ---------- */
const TABS=[["overview","Overview & Forecast"],["predictions","Closing Price Predictions"],["comparison","Model Comparison"],["errors","Errors & Residuals"],
  ["optimization","Optimization"],["features","Features & Correlation"],["adaptive","Adaptive Dual-Phase"],["benchmarks","Complexity & Runtime"]];
function renderResults(){
  $("results").classList.remove("hidden");
  renderKPIs();
  $("tabs").innerHTML=TABS.map((t,i)=>`<div class="tab ${i===0?'active':''}" data-tab="${t[0]}" onclick="switchTab('${t[0]}')">${t[1]}</div>`).join("");
  switchTab("overview");
  $("results").scrollIntoView({behavior:"smooth"});
}
function renderKPIs(){
  const lb=state.final.leaderboard, f=state.final;
  const flag=lb.find(r=>r.index===FLAGSHIP_INDEX)||{};
  const best=lb[0]||{};
  $("kpis").innerHTML=`
    <div class="kpi flag"><div class="v" style="color:var(--flagship)">#${f.flagship_rank||"—"}</div><div class="l">Flagship Ensemble rank</div></div>
    <div class="kpi"><div class="v">${fmt(flag.rmse,3)}</div><div class="l">Flagship RMSE</div></div>
    <div class="kpi"><div class="v">${pct(flag.dir_acc,1)}</div><div class="l">Flagship Directional Acc.</div></div>
    <div class="kpi"><div class="v">${best.name?best.name.replace('Adaptive Dual-Phase PSO-GWO','ADP'):'—'}</div><div class="l">Leaderboard #1 (RMSE ${fmt(best.rmse,3)})</div></div>`;
}
function switchTab(name){
  document.querySelectorAll(".tab").forEach(t=>t.classList.toggle("active",t.dataset.tab===name));
  const renderers={overview:renderOverview,predictions:renderPredictions,comparison:renderComparison,errors:renderErrors,
    optimization:renderOptimization,features:renderFeatures,adaptive:renderAdaptive,benchmarks:renderBenchmarks};
  (renderers[name]||(()=>{}))();
}
function modelByIndex(i){ return state.models.find(m=>m.index===i); }
function modelSelector(onchange){
  const opts=[...state.models].sort((a,b)=>a.index-b.index).map(m=>`<option value="${m.index}" ${m.index===state.selected?'selected':''}>${m.index+1}. ${m.name} — RMSE ${fmt(m.rmse,3)}</option>`).join("");
  return `<select class="sel" id="msel" onchange="state.selected=+this.value;${onchange}()">${opts}</select>`;
}

/* ================= TAB 1: OVERVIEW & FORECAST ================= */
function renderOverview(){
  const m=modelByIndex(state.selected)||state.models[0];
  $("tab-body").innerHTML=`
    <div class="toolbar">${modelSelector("renderOverview")}
      <span class="muted">Selected model · ${m.optimizer?.method||m.group||""}</span></div>
    <div class="chips" style="margin-bottom:14px" id="ov-chips"></div>
    <div class="plot lg" id="c-ap"><div></div></div>
    <div class="legend-note">Actual vs predicted across the walk-forward validation window and the held-out test set. Predictions are one-step-ahead close prices reconstructed from forecast returns.</div>
    <div class="plot lg" id="c-fc"><div></div></div>
    <div class="legend-note">Recursive ${state.final.config?.horizon||11}-day forecast with 80 / 90 / 95% prediction intervals. Spotlight markers: D+1, D+2, D+3, D+5, D+7, D+11.</div>`;
  // metric chips
  const c=[["Train MSE",fmt(m.train_mse,4)],["Test MSE",fmt(m.test_mse,4)],["RMSE",fmt(m.rmse,3)],["MAE",fmt(m.mae,3)],["MAPE",pct(m.mape,2)],
    ["Theil U",fmt(m.theil_u,3)],["ARV",fmt(m.arv,3)],["R²",fmt(m.r2,3)],["Dir. Acc",pct(m.dir_acc,1)],["Hit Ratio",pct(m.hit_ratio,1)],
    ["Bias",fmt(m.bias,3)],["Tracking",fmt(m.tracking_signal,2)],["Features",m.n_selected||m.n_features]];
  $("ov-chips").innerHTML=c.map(x=>`<span class="chip">${x[0]} <b>${x[1]}</b></span>`).join("");

  // actual vs pred (train tail + val + test)
  const tr=[];
  const trainTail=Math.min(m.train.dates.length,120);
  tr.push({x:m.train.dates.slice(-trainTail),y:m.train.actual.slice(-trainTail),name:"Train actual",mode:"lines",line:{color:COLORS.actual,width:1.3}});
  tr.push({x:m.train.dates.slice(-trainTail),y:m.train.pred.slice(-trainTail),name:"Train fit",mode:"lines",line:{color:COLORS.train,width:1,dash:"dot"}});
  if(m.val&&m.val.dates.length){ tr.push({x:m.val.dates,y:m.val.pred,name:"Validation pred",mode:"lines",line:{color:COLORS.val,width:1.6}}); }
  tr.push({x:m.test.dates,y:m.test.actual,name:"Test actual",mode:"lines",line:{color:COLORS.actual,width:2.2}});
  tr.push({x:m.test.dates,y:m.test.pred,name:"Test predicted",mode:"lines",line:{color:COLORS.pred,width:2}});
  plot($("c-ap").firstChild,tr,{title:"Actual vs Predicted — "+m.name});

  // forecast with bands
  const f=m.forecast; const hist=state.final;
  const tail=80; const hx=hist.dates_full.slice(-tail), hy=hist.y_full.slice(-tail);
  const ft=[];
  const lvl=(p,c)=>({x:f.dates.concat([...f.dates].reverse()),
    y:f.intervals[p].upper.concat([...f.intervals[p].lower].reverse()),
    fill:"toself",fillcolor:c,line:{color:"rgba(0,0,0,0)"},name:p+"% interval",hoverinfo:"skip"});
  ft.push(lvl("95","rgba(168,85,247,.10)")); ft.push(lvl("90","rgba(168,85,247,.16)")); ft.push(lvl("80","rgba(168,85,247,.24)"));
  ft.push({x:hx,y:hy,name:"History",mode:"lines",line:{color:COLORS.actual,width:2}});
  const bridge=[hx[hx.length-1],...f.dates], bridgeY=[hy[hy.length-1],...f.values];
  ft.push({x:bridge,y:bridgeY,name:"Forecast",mode:"lines+markers",line:{color:COLORS.forecast,width:2.4},marker:{size:4}});
  const pts=f.points||[]; ft.push({x:pts.map(p=>p.date),y:pts.map(p=>p.value),mode:"markers+text",name:"Spotlight",
    text:pts.map(p=>"D+"+p.horizon),textposition:"top center",marker:{size:9,color:COLORS.forecast,symbol:"diamond",line:{color:"#fff",width:1}}});
  plot($("c-fc").firstChild,ft,{title:((state.final.config?.horizon)||11)+"-Day Recursive Forecast — "+m.name});
}

/* ================= TAB: CLOSING PRICE PREDICTIONS ================= */
/* One two-panel figure per model — left = training-set fit (in-sample),
   right = testing-set prediction (out-of-sample). Blue = actual close,
   red = predicted close. Mirrors the reference "Closing price prediction
   using <model>" layout and is included verbatim in the HTML report. */
function renderPredictions(){
  const models=[...state.models].sort((a,b)=>a.index-b.index);
  $("tab-body").innerHTML=
    `<div class="legend-note">Per-model closing-price prediction. Left panel: training-set fit (in-sample). Right panel: held-out testing-set prediction (out-of-sample). <b style="color:${COLORS.actual}">Blue</b> = actual close · <b style="color:${COLORS.pred}">Red</b> = predicted close. All ${models.length} models shown.</div>`
    + models.map(m=>`
      <div class="card">
        <h3>${(m.index===FLAGSHIP_INDEX?'★ ':'')}Closing price prediction using ${m.name}</h3>
        <div class="pgrid">
          <div class="plot md" id="cp-tr-${m.index}"><div></div></div>
          <div class="plot md" id="cp-te-${m.index}"><div></div></div>
        </div>
        <div class="legend-note">Train MSE ${fmt(m.train_mse,4)} · Test MSE ${fmt(m.test_mse,4)} · RMSE ${fmt(m.rmse,3)} · MAE ${fmt(m.mae,3)} · MAPE ${fmt(m.mape,2)}% · R² ${fmt(m.r2,3)} · Dir ${fmt(m.dir_acc,1)}%</div>
      </div>`).join("");
  models.forEach(m=>{
    const trActual={x:m.train.dates,y:m.train.actual,name:"Actual",mode:"lines",line:{color:COLORS.actual,width:1.2}};
    const trPred={x:m.train.dates,y:m.train.pred,name:"Predicted",mode:"lines",line:{color:COLORS.pred,width:1.2}};
    plot($("cp-tr-"+m.index).firstChild,[trActual,trPred],
      {title:"Training set — model fit (in-sample)",yaxis:{title:"Close price"}});
    const teActual={x:m.test.dates,y:m.test.actual,name:"Actual",mode:"lines",line:{color:COLORS.actual,width:1.7}};
    const tePred={x:m.test.dates,y:m.test.pred,name:"Predicted",mode:"lines",line:{color:COLORS.pred,width:1.7}};
    plot($("cp-te-"+m.index).firstChild,[teActual,tePred],
      {title:"Testing set — prediction (out-of-sample)",yaxis:{title:"Close price"}});
  });
}

/* ================= TAB 2: MODEL COMPARISON ================= */
function renderComparison(){
  $("tab-body").innerHTML=`
    <div class="plot lg" id="c-cmp"><div></div></div>
    <div class="legend-note">All models on the held-out test set vs. the actual close. The flagship ensemble is highlighted.</div>
    <div class="pgrid">
      <div class="plot md" id="c-radar"><div></div></div>
      <div class="plot md" id="c-cum"><div></div></div>
    </div>
    <div class="plot md" id="c-dir"><div></div></div>
    <div class="card"><h3>Leaderboard — all 16 models ranked</h3><div style="max-height:520px;overflow:auto"><table id="lb"></table></div></div>`;
  comparisonChart(); radarChart(); cumulativeChart(); directionalChart(); leaderboard();
}
function comparisonChart(){
  const any=state.models[0]; const dates=any.test.dates;
  const tr=[{x:dates,y:any.test.actual,name:"Actual",mode:"lines",line:{color:"#e2e8f0",width:3}}];
  [...state.models].sort((a,b)=>a.rmse-b.rmse).forEach(m=>{
    const flag=m.index===FLAGSHIP_INDEX;
    tr.push({x:m.test.dates,y:m.test.pred,name:m.name.replace('Adaptive Dual-Phase PSO-GWO','ADP'),mode:"lines",
      line:{color:flag?COLORS.flagship:GROUP_COLOR[m.group]||"#64748b",width:flag?2.6:1,dash:flag?"solid":"dot"},opacity:flag?1:.65});
  });
  plot($("c-cmp").firstChild,tr,{title:"All-Model Test Predictions vs Actual"});
}
function radarChart(){
  const lb=state.final.leaderboard;
  const pick=[...new Set([...lb.slice(0,3).map(r=>r.index),FLAGSHIP_INDEX])];
  const maxR=Math.max(...lb.map(r=>r.rmse)), maxM=Math.max(...lb.map(r=>r.mape));
  const axes=["R²","Dir. Acc","Hit Ratio","Low RMSE","Low MAPE"];
  const tr=pick.map(i=>{ const r=lb.find(x=>x.index===i); if(!r)return null;
    const v=[Math.max(0,r.r2),r.dir_acc/100,r.hit_ratio/100,1-r.rmse/(maxR||1),1-r.mape/(maxM||1)];
    return {type:"scatterpolar",r:v.concat([v[0]]),theta:axes.concat([axes[0]]),fill:"toself",name:r.name.replace('Adaptive Dual-Phase PSO-GWO','ADP'),
      opacity:i===FLAGSHIP_INDEX?.8:.45,line:{color:i===FLAGSHIP_INDEX?COLORS.flagship:undefined}};
  }).filter(Boolean);
  plot($("c-radar").firstChild,tr,{title:"Forecast Accuracy Radar",polar:{radialaxis:{visible:true,range:[0,1]}},hovermode:"closest"});
}
function cumulativeChart(){
  const tr=[...state.models].sort((a,b)=>a.rmse-b.rmse).map(m=>{
    let c=0; const cum=m.residuals.map(e=>{c+=Math.abs(e);return c;});
    const flag=m.index===FLAGSHIP_INDEX;
    return {x:m.test.dates,y:cum,name:m.name.replace('Adaptive Dual-Phase PSO-GWO','ADP'),mode:"lines",
      line:{color:flag?COLORS.flagship:GROUP_COLOR[m.group]||"#64748b",width:flag?2.6:1},opacity:flag?1:.6};
  });
  plot($("c-cum").firstChild,tr,{title:"Cumulative Absolute Error",hovermode:"closest"});
}
function directionalChart(){
  const lb=[...state.final.leaderboard].sort((a,b)=>b.dir_acc-a.dir_acc);
  plot($("c-dir").firstChild,[{type:"bar",x:lb.map(r=>r.name.replace('Adaptive Dual-Phase PSO-GWO','ADP')),y:lb.map(r=>r.dir_acc),
    marker:{color:lb.map(r=>r.index===FLAGSHIP_INDEX?COLORS.flagship:GROUP_COLOR[r.group]||"#64748b")}}],
    {title:"Directional Accuracy by Model (%)",xaxis:{tickangle:-35},margin:{b:150}});
}
function leaderboard(){
  const cols=[["rank","#"],["name","Model"],["group","Family"],
    ["train_mse","Train MSE"],["test_mse","Test MSE"],["rmse","RMSE"],["mae","MAE"],["mape","MAPE"],
    ["theil_u","Theil U"],["arv","ARV"],["r2","R²"],["dir_acc","DIR%"],["n_features","Features"]];
  const feat=r=>r.n_selected||r.n_features;
  const keyOf=r=>state.sortKey==="n_features"?feat(r):r[state.sortKey];
  const rows=[...state.final.leaderboard].sort((a,b)=>state.sortDir*((keyOf(a)>keyOf(b))?1:-1));
  $("lb").innerHTML=`<thead><tr>${cols.map(c=>`<th onclick="sortLB('${c[0]}')">${c[1]}</th>`).join("")}</tr></thead>
    <tbody>${rows.map(r=>`<tr class="${r.index===FLAGSHIP_INDEX?'flagship-row':''} ${r.rank===1?'best-row':''}">
      <td><span class="rankbadge ${r.rank===1?'r1':''}">${r.rank}</span></td><td>${r.name}</td><td>${r.group}</td>
      <td>${fmt(r.train_mse,4)}</td><td>${fmt(r.test_mse,4)}</td><td>${fmt(r.rmse,3)}</td><td>${fmt(r.mae,3)}</td><td>${fmt(r.mape,2)}</td>
      <td>${fmt(r.theil_u,3)}</td><td>${fmt(r.arv,3)}</td><td>${fmt(r.r2,3)}</td><td>${fmt(r.dir_acc,1)}</td>
      <td>${feat(r)}</td></tr>`).join("")}</tbody>`;
}
function sortLB(k){ state.sortDir = (state.sortKey===k)? -state.sortDir : 1; state.sortKey=k; leaderboard(); }

/* ================= TAB 3: ERRORS & RESIDUALS ================= */
function renderErrors(){
  const m=modelByIndex(state.selected)||state.models[0];
  $("tab-body").innerHTML=`
    <div class="toolbar">${modelSelector("renderErrors")}<span class="muted">Residual = actual − predicted (test set)</span></div>
    <div class="pgrid">
      <div class="plot md" id="e-time"><div></div></div>
      <div class="plot md" id="e-vs"><div></div></div>
      <div class="plot md" id="e-hist"><div></div></div>
      <div class="plot md" id="e-qq"><div></div></div>
    </div>
    <div class="plot md" id="e-bars"><div></div></div>
    <div class="legend-note">Error-distribution comparison: per-model RMSE and MAE across the test set.</div>`;
  const res=m.residuals;
  plot($("e-time").firstChild,[{x:m.test.dates,y:res,mode:"lines",name:"Residual",line:{color:COLORS.pred}},
    {x:m.test.dates,y:res.map(()=>0),mode:"lines",line:{color:"#888",dash:"dash",width:1},hoverinfo:"skip",showlegend:false}],
    {title:"Residuals vs Time — "+m.name});
  plot($("e-vs").firstChild,[{x:m.test.pred,y:res,mode:"markers",marker:{color:COLORS.accent||"#4f8cff",size:6,opacity:.7}}],
    {title:"Residuals vs Predicted",xaxis:{title:"Predicted"},yaxis:{title:"Residual"},hovermode:"closest"});
  // histogram + KDE
  const s=std(res)||1,mu=mean(res); const grid=[]; const lo=Math.min(...res),hi=Math.max(...res);
  for(let i=0;i<=60;i++){const x=lo+(hi-lo)*i/60; let dsum=0; res.forEach(r=>{const u=(x-r)/(1.06*s*Math.pow(res.length,-0.2));dsum+=Math.exp(-0.5*u*u);});
    grid.push([x,dsum/(res.length*1.06*s*Math.pow(res.length,-0.2)*Math.sqrt(2*Math.PI))]);}
  plot($("e-hist").firstChild,[{type:"histogram",x:res,histnorm:"probability density",name:"Residuals",marker:{color:"rgba(79,140,255,.55)"},nbinsx:24},
    {x:grid.map(g=>g[0]),y:grid.map(g=>g[1]),mode:"lines",name:"KDE",line:{color:COLORS.forecast,width:2}}],
    {title:"Residual Histogram + KDE",hovermode:"closest"});
  // QQ
  const sorted=[...res].sort((a,b)=>a-b); const sm=mean(sorted),ss=std(sorted)||1;
  const theo=sorted.map((_,i)=>normInv((i+0.5)/sorted.length)); const samp=sorted.map(v=>(v-sm)/ss);
  const lim=Math.max(Math.abs(theo[0]),Math.abs(theo[theo.length-1]));
  plot($("e-qq").firstChild,[{x:theo,y:samp,mode:"markers",name:"Quantiles",marker:{color:COLORS.val,size:6,opacity:.75}},
    {x:[-lim,lim],y:[-lim,lim],mode:"lines",name:"Normal",line:{color:"#888",dash:"dash"}}],
    {title:"Normal Q–Q Plot",xaxis:{title:"Theoretical"},yaxis:{title:"Sample"},hovermode:"closest"});
  const lb=[...state.final.leaderboard].sort((a,b)=>a.rmse-b.rmse);
  plot($("e-bars").firstChild,[
    {type:"bar",name:"RMSE",x:lb.map(r=>r.name.replace('Adaptive Dual-Phase PSO-GWO','ADP')),y:lb.map(r=>r.rmse),marker:{color:"#4f8cff"}},
    {type:"bar",name:"MAE",x:lb.map(r=>r.name.replace('Adaptive Dual-Phase PSO-GWO','ADP')),y:lb.map(r=>r.mae),marker:{color:"#22d3ee"}}],
    {title:"RMSE & MAE by Model",barmode:"group",xaxis:{tickangle:-35},margin:{b:150}});
}

/* ================= TAB 4: OPTIMIZATION ================= */
function renderOptimization(){
  const optimized=state.models.filter(m=>m.optimizer&&m.optimizer.telemetry&&(m.optimizer.telemetry.best||[]).length);
  if(!state.selected||!(modelByIndex(state.selected)?.optimizer?.telemetry?.best||[]).length){
    const first=optimized[0]; if(first) state.selected=first.index;
  }
  const m=modelByIndex(state.selected)||optimized[0];
  const opts=optimized.sort((a,b)=>a.index-b.index).map(x=>`<option value="${x.index}" ${x.index===state.selected?'selected':''}>${x.index+1}. ${x.name}</option>`).join("");
  $("tab-body").innerHTML=`
    <div class="toolbar"><select class="sel" id="msel" onchange="state.selected=+this.value;renderOptimization()">${opts}</select>
      <span class="muted">${m?.optimizer?.method||""}</span></div>
    <div class="pgrid">
      <div class="plot md" id="o-conv"><div></div></div>
      <div class="plot md" id="o-expl"><div></div></div>
      <div class="plot md" id="o-lam"><div></div></div>
      <div class="plot md" id="o-traj"><div></div></div>
    </div>
    <div class="legend-note">Convergence envelope shows best / mean / worst population fitness per iteration. λ(t)=1−(t/T)² governs the adaptive PSO↔GWO balance. Trajectories trace particle/wolf positions through the first two search dimensions.</div>`;
  const t=m.optimizer.telemetry;
  const it=t.best.map((_,i)=>i);
  plot($("o-conv").firstChild,[
    {x:it,y:t.worst,name:"Worst",mode:"lines",line:{color:"#f87171",width:1},fill:"tonexty",fillcolor:"rgba(248,113,113,.06)"},
    {x:it,y:t.mean,name:"Mean",mode:"lines",line:{color:"#fbbf24",width:1.6}},
    {x:it,y:t.best,name:"Best",mode:"lines",line:{color:"#34d399",width:2.6}}],
    {title:"Convergence (best/mean/worst) — "+m.name,xaxis:{title:"Iteration"},yaxis:{title:"Fitness (val RMSE)"}});
  plot($("o-expl").firstChild,[
    {x:it,y:t.exploration,name:"Exploration",mode:"lines",line:{color:"#4f8cff",width:2},fill:"tozeroy",fillcolor:"rgba(79,140,255,.12)"},
    {x:it,y:t.exploitation,name:"Exploitation",mode:"lines",line:{color:"#a855f7",width:2}}],
    {title:"Exploration vs Exploitation",xaxis:{title:"Iteration"},yaxis:{title:"Ratio",range:[0,1]}});
  if(t.lam&&t.lam.length){
    plot($("o-lam").firstChild,[{x:t.lam.map((_,i)=>i),y:t.lam,mode:"lines",name:"λ(t)",line:{color:"#22d3ee",width:2.4}},
      ...(t.a&&t.a.length?[{x:t.a.map((_,i)=>i),y:t.a.map(v=>v/2),mode:"lines",name:"a(t)/2 (GWO)",line:{color:"#fbbf24",width:1.6,dash:"dot"}}]:[])],
      {title:"Adaptive λ(t) & GWO pressure",xaxis:{title:"Iteration"},yaxis:{title:"Value",range:[0,1]}});
  } else { plot($("o-lam").firstChild,[{x:it,y:t.diversity,mode:"lines",name:"Diversity",line:{color:"#22d3ee",width:2}}],{title:"Population Diversity",xaxis:{title:"Iteration"}}); }
  // trajectories
  const trj=t.trajectories||[]; const lines=[];
  const nP=trj.length?trj[0].length:0;
  for(let p=0;p<nP;p++){ const xs=[],ys=[]; trj.forEach(step=>{ if(step[p]){xs.push(step[p][0]);ys.push(step[p][1]);} });
    lines.push({x:xs,y:ys,mode:"lines+markers",line:{width:1},marker:{size:3},showlegend:false,opacity:.55}); }
  plot($("o-traj").firstChild,lines.length?lines:[{x:[],y:[]}],{title:"Particle / Wolf Trajectories (dims 0×1)",xaxis:{title:"dim 0",range:[0,1]},yaxis:{title:"dim 1",range:[0,1]},hovermode:"closest"});
}

/* ================= TAB 5: FEATURES & CORRELATION ================= */
function renderFeatures(){
  const gbdt=state.models.filter(m=>m.feature_importance&&m.feature_importance.length);
  $("tab-body").innerHTML=`
    <div class="pgrid">
      <div class="plot md" id="f-imp"><div></div></div>
      <div class="plot md" id="f-tcorr"><div></div></div>
    </div>
    <div class="plot lg" id="f-corr"><div></div></div>
    <div class="legend-note">Feature/feature correlation (training window only — leak-free). Below: Phase-1 selection-frequency stability across multi-seed runs.</div>
    <div class="plot md" id="f-stab"><div></div></div>`;
  // importance: pick a tree model (prefer Adaptive GBDT)
  const tree=gbdt.find(m=>m.index===14)||gbdt.find(m=>m.group==="Adaptive Dual-Phase")||gbdt[0];
  if(tree){ const imp=tree.feature_importance.slice(0,18).reverse();
    plot($("f-imp").firstChild,[{type:"bar",orientation:"h",y:imp.map(d=>d.feature),x:imp.map(d=>d.importance),marker:{color:"#34d399"}}],
      {title:"Feature Importance — "+tree.name,margin:{l:130}});
  } else plot($("f-imp").firstChild,[{x:[],y:[]}],{title:"Feature Importance (n/a)"});
  // target corr
  const tc=(state.final.diagnostics?.target_corr||[]).slice(0,18).reverse();
  plot($("f-tcorr").firstChild,[{type:"bar",orientation:"h",y:tc.map(d=>d.feature),x:tc.map(d=>d.corr),
    marker:{color:tc.map(d=>d.corr>=0?"#4f8cff":"#f87171")}}],{title:"Feature → Target Correlation",margin:{l:130}});
  // correlation heatmap
  const dg=state.final.diagnostics;
  if(dg&&dg.matrix){ plot($("f-corr").firstChild,[{type:"heatmap",z:dg.matrix,x:dg.feature_names,y:dg.feature_names,
    colorscale:"RdBu",zmid:0,zmin:-1,zmax:1,colorbar:{title:"ρ"}}],{title:"Feature Correlation Heatmap",margin:{l:110,b:110},xaxis:{tickangle:-45}}); }
  // stability heatmap from flagship phase1
  const fl=modelByIndex(FLAGSHIP_INDEX)||modelByIndex(12);
  const p1=fl?.phase1;
  if(p1&&p1.select_freq_by_seed&&p1.select_freq_by_seed.length){
    const names=p1.feature_names; const sel=new Set(p1.selected||[]);
    // order by mean frequency, show top 30
    const meanf=names.map((_,i)=>mean(p1.select_freq_by_seed.map(r=>r[i])));
    const order=names.map((_,i)=>i).sort((a,b)=>meanf[b]-meanf[a]).slice(0,30);
    const z=p1.select_freq_by_seed.map(row=>order.map(i=>row[i]));
    plot($("f-stab").firstChild,[{type:"heatmap",z:z,x:order.map(i=>names[i]+(sel.has(i)?" ✓":"")),y:p1.seeds.map(s=>"seed "+s),
      colorscale:"Viridis",zmin:0,zmax:1,colorbar:{title:"freq"}}],{title:"Phase-1 Feature Selection Stability (multi-seed)",margin:{b:130},xaxis:{tickangle:-45}});
  } else plot($("f-stab").firstChild,[{x:[],y:[]}],{title:"Feature Selection Stability (n/a)"});
}

/* ================= TAB 6: ADAPTIVE DUAL-PHASE ================= */
function renderAdaptive(){
  const fl=modelByIndex(FLAGSHIP_INDEX);
  const fam={12:modelByIndex(12),13:modelByIndex(13),14:modelByIndex(14)};
  $("tab-body").innerHTML=`
    <div class="chips" style="margin-bottom:14px" id="a-chips"></div>
    <div class="pgrid">
      <div class="plot md" id="a-p1"><div></div></div>
      <div class="plot md" id="a-p2"><div></div></div>
      <div class="plot md" id="a-w-pie"><div></div></div>
      <div class="plot md" id="a-w-bar"><div></div></div>
    </div>
    <div class="legend-note">Phase 1 = binary feature selection convergence; Phase 2 = continuous hyper-parameter search convergence. Fusion weights are non-negative, sum to 1, and calibrated by temporal cross-validation.</div>
    <div class="card"><h3>Selected features (${fl?.n_selected||0})</h3><div class="chips" id="a-feats"></div></div>`;
  if(!fl){ $("tab-body").innerHTML+="<p class='muted'>Ensemble result unavailable.</p>"; return; }
  const chips=[["Flagship rank","#"+(state.final.flagship_rank||"—")],["Experts fused",(fl.fusion_weights||[]).length],
    ["Selected features",fl.n_selected||0],["Champion",(fl.best_hyper?.champion_mode||fl.optimizer?.method||"").replace('Adaptive Dual-Phase PSO-GWO','ADP')]];
  $("a-chips").innerHTML=chips.map(c=>`<span class="chip">${c[0]} <b>${c[1]}</b></span>`).join("");

  // phase 1 convergence across families
  const p1tr=[]; Object.values(fam).forEach((m,i)=>{ if(m&&m.phase1&&m.phase1.history){ p1tr.push({x:m.phase1.history.map((_,j)=>j),y:m.phase1.history,mode:"lines",name:m.name.replace('Adaptive Dual-Phase PSO-GWO + ',''),line:{width:2}}); }});
  plot($("a-p1").firstChild,p1tr.length?p1tr:[{x:[],y:[]}],{title:"Phase 1 — Feature Selection Convergence",xaxis:{title:"Iteration"},yaxis:{title:"Fitness"},hovermode:"closest"});
  // phase 2 convergence + lambda
  const m12=fam[12];
  const p2tr=[]; Object.values(fam).forEach(m=>{ if(m&&m.phase2&&m.phase2.history){ p2tr.push({x:m.phase2.history.map((_,j)=>j),y:m.phase2.history,mode:"lines",name:m.name.replace('Adaptive Dual-Phase PSO-GWO + ',''),line:{width:2}}); }});
  if(m12&&m12.phase2&&m12.phase2.lam_curve&&m12.phase2.lam_curve.length){ p2tr.push({x:m12.phase2.lam_curve.map((_,j)=>j),y:m12.phase2.lam_curve,mode:"lines",name:"λ(t)",yaxis:"y2",line:{color:"#22d3ee",dash:"dot"}}); }
  plot($("a-p2").firstChild,p2tr.length?p2tr:[{x:[],y:[]}],{title:"Phase 2 — Hyper-parameter Convergence",xaxis:{title:"Iteration"},yaxis:{title:"Fitness"},yaxis2:{overlaying:"y",side:"right",range:[0,1],title:"λ"},hovermode:"closest"});
  // fusion weights
  const fw=(fl.fusion_weights||[]).filter(w=>w.weight>0);
  plot($("a-w-pie").firstChild,[{type:"pie",labels:fw.map(w=>w.name),values:fw.map(w=>w.weight),hole:.5,textinfo:"label+percent",
    marker:{colors:["#4f8cff","#22d3ee","#34d399","#fbbf24","#f87171","#a855f7","#f472b6","#60a5fa","#818cf8"]}}],{title:"Ensemble Weight Distribution",showlegend:false});
  plot($("a-w-bar").firstChild,[{type:"bar",x:fw.map(w=>w.name),y:fw.map(w=>w.weight),marker:{color:"#a855f7"},
    text:fw.map(w=>"val "+fmt(w.validation_rmse,3)),textposition:"outside"}],{title:"Expert Weights & Validation RMSE",xaxis:{tickangle:-25},margin:{b:120}});
  $("a-feats").innerHTML=(fl.selected_features||[]).map(f=>`<span class="chip">${f}</span>`).join("")||"<span class='muted'>—</span>";
}

/* ================= TAB 7: COMPLEXITY & RUNTIME ================= */
function renderBenchmarks(){
  const lstm=state.models.find(m=>m.learning_curve&&m.learning_curve.train&&m.learning_curve.train.length);
  $("tab-body").innerHTML=`
    <div class="pgrid">
      <div class="plot md" id="b-acc-rt"><div></div></div>
      <div class="plot md" id="b-acc-feat"><div></div></div>
    </div>
    <div class="plot md" id="b-lc"><div></div></div>
    <div class="pgrid">
      <div class="card"><h3>Runtime benchmark</h3><div style="max-height:420px;overflow:auto"><table id="rt-table"></table></div></div>
      <div class="card"><h3>Memory consumption</h3><div style="max-height:420px;overflow:auto"><table id="mem-table"></table></div></div>
    </div>`;
  const lb=state.final.leaderboard;
  // accuracy vs runtime
  plot($("b-acc-rt").firstChild,[{x:lb.map(r=>r.timing_sec),y:lb.map(r=>r.rmse),mode:"markers+text",
    text:lb.map(r=>r.index+1),textposition:"top center",marker:{size:11,color:lb.map(r=>r.index===FLAGSHIP_INDEX?COLORS.flagship:GROUP_COLOR[r.group]||"#64748b")}}],
    {title:"Accuracy vs Runtime",xaxis:{title:"Training time (s)"},yaxis:{title:"RMSE (lower better)"},hovermode:"closest"});
  // accuracy vs features
  plot($("b-acc-feat").firstChild,[{x:lb.map(r=>r.n_selected||r.n_features),y:lb.map(r=>r.r2),mode:"markers+text",
    text:lb.map(r=>r.index+1),textposition:"top center",marker:{size:11,color:lb.map(r=>r.index===FLAGSHIP_INDEX?COLORS.flagship:GROUP_COLOR[r.group]||"#64748b")}}],
    {title:"Accuracy vs Feature Count",xaxis:{title:"# features used"},yaxis:{title:"R²"},hovermode:"closest"});
  // learning curve
  if(lstm){ const lc=lstm.learning_curve; plot($("b-lc").firstChild,[
    {x:lc.train.map((_,i)=>i),y:lc.train,mode:"lines",name:"Training loss",line:{color:"#34d399",width:2}},
    {x:lc.val.map((_,i)=>i),y:lc.val,mode:"lines",name:"Validation loss",line:{color:"#fbbf24",width:2}}],
    {title:"LSTM Learning Curve — "+lstm.name,xaxis:{title:"Epoch"},yaxis:{title:"MSE (scaled returns)"},hovermode:"x"});
  } else plot($("b-lc").firstChild,[{x:[],y:[]}],{title:"Learning Curve (n/a)"});
  // tables
  const rt=[...state.final.runtime_table].sort((a,b)=>b.timing_sec-a.timing_sec);
  $("rt-table").innerHTML=`<thead><tr><th>Model</th><th>Time (s)</th><th>Features</th></tr></thead><tbody>${rt.map(r=>`<tr><td>${r.name}</td><td>${fmt(r.timing_sec,2)}</td><td>${r.n_features}</td></tr>`).join("")}</tbody>`;
  const mem=[...state.final.runtime_table].sort((a,b)=>b.memory_mb-a.memory_mb);
  $("mem-table").innerHTML=`<thead><tr><th>Model</th><th>Peak Python mem (MB)</th></tr></thead><tbody>${mem.map(r=>`<tr><td>${r.name}</td><td>${fmt(r.memory_mb,2)}</td></tr>`).join("")}</tbody>`;
}

/* ---------- export (self-contained interactive report) ---------- */
function safeName(s){ return String(s).replace(/[^A-Za-z0-9_.-]/g,"_"); }
function triggerDownload(blob,name){
  const a=document.createElement("a"); a.href=URL.createObjectURL(blob); a.download=name;
  document.body.appendChild(a); a.click(); setTimeout(()=>{URL.revokeObjectURL(a.href); a.remove();},1500);
}
function downloadJSON(){
  if(!state.final){ toast("Run a benchmark first","err"); return; }
  const sym=($("d-sym")&&$("d-sym").textContent)||"results";
  triggerDownload(new Blob([JSON.stringify({final:state.final,models:state.models},null,2)],{type:"application/json"}),
    "ADP_results_"+safeName(sym)+".json");
  toast("Results JSON downloaded ✓","ok");
}
async function downloadReport(){
  if(!state.final){ toast("Run a benchmark first","err"); return; }
  toast("Building self-contained report…");
  const sym=($("d-sym")&&$("d-sym").textContent)||"report";
  const O="<scr"+"ipt>", C="</scr"+"ipt>";
  const css=Array.from(document.querySelectorAll("style")).map(s=>s.outerHTML).join("\n");
  let appjs=""; try{ appjs=await (await fetch("/static/app.js")).text(); }catch(e){}
  let plotly=""; try{ plotly=await (await fetch("/static/vendor/plotly.min.js")).text(); }catch(e){}
  // Drop the offline-polling line so the saved file doesn't spam failed requests.
  appjs=appjs.replace(/health\(\);\s*setInterval\(health,\s*15000\);/, "");
  const plotlyTag = plotly ? (O+plotly+C) : ('<scr'+'ipt src="https://cdn.plot.ly/plotly-2.35.2.min.js">'+C);
  const payload=JSON.stringify({final:state.final,models:state.models,symbol:sym});
  const skeleton=
    '<header><div class="brand"><div class="logo">Φ</div><div>'+
    '<h1>Adaptive Dual-Phase PSO-GWO — Report</h1>'+
    '<div class="sub">'+sym+' · offline snapshot · click tabs &amp; charts to interact</div></div></div>'+
    '<div class="spacer"></div>'+
    '<span class="pill"><span class="dot on" id="redis-dot"></span><span id="redis-text">offline report</span></span>'+
    '<button class="btn icon-btn btn-ghost" id="theme-btn" title="Toggle theme">◐</button></header>'+
    '<div class="wrap"><div id="results"><div class="kpis" id="kpis"></div>'+
    '<div class="tabs" id="tabs"></div><div id="tab-body"></div></div></div>'+
    '<div class="toast" id="toast"></div>';
  const boot=O+
    'try{state.final=window.__REPORT__.final;state.models=window.__REPORT__.models;'+
    'state.selected=(state.models.find(function(m){return m.index===15})?15:'+
    '((state.final.leaderboard[0]||{}).index||0));renderResults();window.scrollTo(0,0);}'+
    'catch(e){document.body.innerHTML="<pre style=\\"padding:24px;color:#f87171\\">Report render error: "+e+"</pre>";}'+C;
  const doc="<!DOCTYPE html><html lang=\"en\" data-theme=\"dark\"><head><meta charset=\"UTF-8\">"+
    "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"+
    "<title>ADP Report — "+sym+"</title>"+plotlyTag+css+"</head><body>"+skeleton+
    O+"window.__REPORT__="+payload+";"+C+O+appjs+C+boot+"</body></html>";
  triggerDownload(new Blob([doc],{type:"text/html;charset=utf-8"}), "ADP_report_"+safeName(sym)+".html");
  toast("Report downloaded ✓ — open it in any browser, fully interactive","ok");
}

/* ---------- boot ---------- */
$("theme-btn").onclick=()=>{ const d=isDark(); document.documentElement.setAttribute("data-theme",d?"light":"dark");
  localStorage.setItem("theme",d?"light":"dark"); if(state.final){ switchTab(document.querySelector(".tab.active")?.dataset.tab||"overview"); } };
(function(){ const t=localStorage.getItem("theme"); if(t)document.documentElement.setAttribute("data-theme",t); })();
health(); setInterval(health,15000);
