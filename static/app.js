// UI wiring
document.addEventListener('DOMContentLoaded', () => {
  const providerBtn = document.getElementById('btn_provider');
  const claimBtn = document.getElementById('btn_claim');
  const providerSection = document.getElementById('provider_section');
  const claimSection = document.getElementById('claim_section');
  const modeButtons = document.querySelectorAll('.mode');

  providerBtn.addEventListener('click', () => {
    providerSection.classList.remove('hidden');
    claimSection.classList.add('hidden');
  });
  claimBtn.addEventListener('click', () => {
    claimSection.classList.remove('hidden');
    providerSection.classList.add('hidden');
  });

  // mode switch
  modeButtons.forEach(btn => {
    btn.addEventListener('click', (e) => {
      const mode = btn.dataset.mode;
      const type = btn.dataset.type; // provider or claim
      const panel = document.querySelector(`#${type}_section`);
      panel.querySelectorAll('.batch-panel').forEach(el => el.classList.add('hidden'));
      panel.querySelectorAll('.single-panel').forEach(el => el.classList.add('hidden'));
      if (mode === 'batch') {
        panel.querySelector(`.batch-panel[data-type="${type}"]`).classList.remove('hidden');
      } else {
        panel.querySelector(`.single-panel[data-type="${type}"]`).classList.remove('hidden');
      }
    });
  });

  // Provider single submit
  const providerSingleForm = document.getElementById('provider_single_form');
  providerSingleForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const data = Object.fromEntries(new FormData(providerSingleForm).entries());
    // convert values to numbers
    Object.keys(data).forEach(k => data[k] = parseFloat(data[k]));
    const res = await fetch('/predict_provider_single', {
      method:'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(data)
    });
    const json = await res.json();
    document.getElementById('provider_single_result').innerHTML = resultHtml(json);
  });

  // Claim single submit
  const claimSingleForm = document.getElementById('claim_single_form');
  claimSingleForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const data = Object.fromEntries(new FormData(claimSingleForm).entries());
    Object.keys(data).forEach(k => data[k] = parseFloat(data[k]));
    const res = await fetch('/predict_claim_single', {
      method:'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(data)
    });
    const json = await res.json();
    document.getElementById('claim_single_result').innerHTML = resultHtml(json);
  });

  // Provider batch
  document.getElementById('provider_batch_analyze').addEventListener('click', async () => {
    const fileInput = document.getElementById('provider_batch_file');
    if (!fileInput.files.length) {
      alert('Please choose a file.');
      return;
    }
    const fd = new FormData();
    fd.append('file', fileInput.files[0]);
    const res = await fetch('/upload_provider_batch', { method:'POST', body: fd });
    const json = await res.json();
    if (json.error) {
      document.getElementById('provider_batch_result').innerHTML = `<pre style="color:red">${json.error}</pre>`;
    } else {
      document.getElementById('provider_batch_result').innerHTML = `<p>${json.message}. Total records: ${json.total_records}</p>
        <button onclick="generateReport('${json.analysis_id}', 'provider')">Generate AI Report</button>
        <button onclick="openDashboard()">Open Dashboard</button>
        <div><h4>Preview</h4><pre style='max-height:300px;overflow:auto'>${JSON.stringify(json.summary_preview.slice(0,10), null, 2)}</pre></div>`;
    }
  });

  // Claim batch
  document.getElementById('claim_batch_analyze').addEventListener('click', async () => {
    const fileInput = document.getElementById('claim_batch_file');
    if (!fileInput.files.length) {
      alert('Please choose a file.');
      return;
    }
    const fd = new FormData();
    fd.append('file', fileInput.files[0]);
    const res = await fetch('/upload_claim_batch', { method:'POST', body: fd });
    const json = await res.json();
    if (json.error) {
      document.getElementById('claim_batch_result').innerHTML = `<pre style="color:red">${json.error}</pre>`;
    } else {
      document.getElementById('claim_batch_result').innerHTML = `<p>${json.message}. Total records: ${json.total_records}</p>
        <button onclick="generateReport('${json.analysis_id}', 'claim', ${JSON.stringify(json.summary_preview)})">Generate AI Report</button>
        <button onclick="openDashboard()">Open Dashboard</button>
        <div><h4>Preview</h4><pre style='max-height:300px;overflow:auto'>${JSON.stringify(json.summary_preview.slice(0,10), null, 2)}</pre></div>`;
    }
  });

  // helper functions
  window.openDashboard = function() {
    window.location.href = '/dashboard';
  }

  window.generateReport = async function(analysis_id, type, summary_preview) {
    // If we have the summary preview we pass it; otherwise backend will create minimal report
    const payload = { analysis_id, type, summary: summary_preview || [] };
    const res = await fetch('/generate_report', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    const json = await res.json();
    if (json.error) {
      alert('Report generation failed: ' + json.error);
    } else {
      alert('Report generated: ' + json.report_file);
      document.getElementById('last_report_span').innerHTML = `Last report: <a href="${json.report_path}" target="_blank">${json.report_file}</a>`;
    }
  }

  function resultHtml(json) {
    if (json.error) {
      return `<pre style="color:red">${json.error}</pre>`;
    }
    return `<div>
      <p><strong>Prediction:</strong> ${json.prediction}</p>
      <p><strong>Fraud Probability:</strong> ${json.fraud_probability}</p>
      <p><strong>Risk Level:</strong> ${json.risk_level}</p>
      <p><strong>Confidence:</strong> ${json.confidence}</p>
      <p><strong>Potential Saving:</strong> ${json.potential_saving}</p>
      <div style="margin-top:8px">
        <button onclick='generateReport("${Date.now()}","single", [${JSON.stringify(json)}])'>Generate AI Report</button>
        <button onclick='openDashboard()'>Open Dashboard</button>
      </div>
    </div>`;
  }
});
