//app.js

document.addEventListener('DOMContentLoaded', () => {

  const providerBtn = document.getElementById('btn_provider');
  const claimBtn = document.getElementById('btn_claim');
  const providerSection = document.getElementById('provider_section');
  const claimSection = document.getElementById('claim_section');
  const modeButtons = document.querySelectorAll('.mode');

  // Switch main section
  providerBtn.addEventListener('click', () => {
    providerSection.classList.remove('hidden');
    claimSection.classList.add('hidden');
  });
  claimBtn.addEventListener('click', () => {
    claimSection.classList.remove('hidden');
    providerSection.classList.add('hidden');
  });

  // Switch single/batch mode per section
  modeButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const mode = btn.dataset.mode;
      const type = btn.dataset.type;
      const panel = document.querySelector(`#${type}_section`);
      panel.querySelectorAll('.batch-panel, .single-panel').forEach(el => el.classList.add('hidden'));
      panel.querySelector(`.${mode}-panel[data-type="${type}"]`).classList.remove('hidden');

      // Highlight active mode
      panel.querySelectorAll('.mode').forEach(m => m.classList.remove('active'));
      btn.classList.add('active');
    });
  });

  // --- Provider Single ---
  const providerSingleForm = document.getElementById('provider_single_form');
  providerSingleForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const data = Object.fromEntries(new FormData(providerSingleForm).entries());
    Object.keys(data).forEach(k => data[k] = parseFloat(data[k]));
    const res = await fetch('/predict_provider_single', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(data)
    });
    const json = await res.json();
    typeOutSummary('provider_single_result', json);
  });

  // --- Claim Single ---
  const claimSingleForm = document.getElementById('claim_single_form');
  claimSingleForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const data = Object.fromEntries(new FormData(claimSingleForm).entries());
    Object.keys(data).forEach(k => data[k] = parseFloat(data[k]));
    const res = await fetch('/predict_claim_single', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(data)
    });
    const json = await res.json();
    typeOutSummary('claim_single_result', json);
  });

  // --- Provider Batch ---
  document.getElementById('provider_batch_analyze').addEventListener('click', async () => {
    const fileInput = document.getElementById('provider_batch_file');
    if (!fileInput.files.length) { alert('Please choose a file.'); return; }
    const fd = new FormData();
    fd.append('file', fileInput.files[0]);
    const res = await fetch('/upload_provider_batch', { method:'POST', body: fd });
    const json = await res.json();
    handleBatchResult(json, 'provider', 'provider_batch_result');
  });

  // --- Claim Batch ---
  document.getElementById('claim_batch_analyze').addEventListener('click', async () => {
    const fileInput = document.getElementById('claim_batch_file');
    if (!fileInput.files.length) { alert('Please choose a file.'); return; }
    const fd = new FormData();
    fd.append('file', fileInput.files[0]);
    const res = await fetch('/upload_claim_batch', { method:'POST', body: fd });
    const json = await res.json();
    handleBatchResult(json, 'claim', 'claim_batch_result');
  });

  // --- Helpers ---
  function handleBatchResult(json, type, containerId){
    const container = document.getElementById(containerId);
    if(json.error){
      container.innerHTML = `<pre style="color:red">${json.error}</pre>`;
    } else {
      container.innerHTML = `<p>${json.message}. Total records: ${json.total_records}</p>
        <p>AI Summary:<br>${json.ai_summary}</p>
        <button onclick="openReport('${json.report_path}')">Download Report</button>
        <div><h4>Preview (first 10 rows)</h4>
        <pre style="max-height:300px;overflow:auto">${JSON.stringify(json.summary_preview.slice(0,10), null, 2)}</pre></div>`;
    }
  }

  window.openReport = function(path){
    window.open(path, '_blank');
  }

  // --- Type out AI summary line by line ---
  async function typeOutSummary(containerId, json){
    const container = document.getElementById(containerId);
    if(json.error){
      container.innerHTML = `<pre style="color:red">${json.error}</pre>`;
      return;
    }

    const lines = [
      `Prediction: ${json.prediction}`,
      `Fraud Probability: ${json.fraud_probability}`,
      `Risk Level: ${json.risk_level}`,
      `Potential Saving: ${json.potential_saving}`,
      `AI Summary:\n${json.ai_summary}`
    ];

    container.innerHTML = '';
    for(let line of lines){
      await typeLine(container, line);
    }
  }

  function typeLine(container, text){
    return new Promise(resolve => {
      const p = document.createElement('p');
      container.appendChild(p);
      let i=0;
      const interval = setInterval(() => {
        p.textContent += text[i];
        i++;
        if(i>=text.length){
          clearInterval(interval);
          resolve();
        }
      }, 15); // 15ms per character
    });
  }

});