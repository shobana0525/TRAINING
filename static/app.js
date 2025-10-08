// static/app.js

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
    const singleBtn = providerSection.querySelector('.mode[data-mode="single"]');
    if (singleBtn) singleBtn.click();
  });

  claimBtn.addEventListener('click', () => {
    claimSection.classList.remove('hidden');
    providerSection.classList.add('hidden');
    const singleBtn = claimSection.querySelector('.mode[data-mode="single"]');
    if (singleBtn) singleBtn.click();
  });

  // Switch single/batch mode per section
  modeButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const mode = btn.dataset.mode;
      const type = btn.dataset.type;
      const panel = document.querySelector(`#${type}_section`);
      panel.querySelectorAll('.batch-panel, .single-panel').forEach(el => el.classList.add('hidden'));
      panel.querySelector(`.${mode}-panel[data-type="${type}"]`).classList.remove('hidden');

      panel.querySelectorAll('.mode').forEach(m => m.classList.remove('active'));
      btn.classList.add('active');
    });
  });

  // ---------- Helpers ----------
  function gatherFormData(form) {
    const data = Object.fromEntries(new FormData(form).entries());
    Object.keys(data).forEach(k => {
      const v = data[k];
      const n = parseFloat(v);
      data[k] = isNaN(n) ? v : n;
    });
    return data;
  }

  function showPrediction(containerId, json) {
    const container = document.getElementById(containerId);
    if (json.error) {
      container.innerHTML = `<pre style="color:red">${json.error}</pre>`;
      return;
    }
    container.innerHTML = `
      <div style="padding:12px;border-radius:8px;background:linear-gradient(90deg, rgba(74,144,226,0.12), rgba(53,122,183,0.06));">
        <p><strong>Prediction:</strong> ${json.prediction}</p>
        <p><strong>Fraud Probability:</strong> ${json.fraud_probability}</p>
        <p><strong>Risk Level:</strong> ${json.risk_level}</p>
        <p><strong>Potential Saving:</strong> ${json.potential_saving}</p>
        ${json.report_path ? `<p><button onclick="openReport('${json.report_path}')">Download Report</button></p>` : ''}
      </div>
    `;
  }

  async function typeOutSummary(containerId, json) {
    const container = document.getElementById(containerId);
    if (json.error) {
      container.innerHTML = `<pre style="color:red">${json.error}</pre>`;
      return;
    }

    const lines = [
      json.prediction ? `Prediction: ${json.prediction}` : '',
      `Fraud Probability: ${json.fraud_probability}`,
      `Risk Level: ${json.risk_level}`,
      `Potential Saving: ${json.potential_saving}`,
      `AI Summary:\n${json.ai_summary}`
    ].filter(Boolean);

    container.innerHTML = '';
    for (let line of lines) {
      await typeLine(container, line);
    }
  }

  function typeLine(container, text) {
    return new Promise(resolve => {
      const p = document.createElement('p');
      container.appendChild(p);
      let i = 0;
      const interval = setInterval(() => {
        p.textContent += text[i];
        i++;
        if (i >= text.length) {
          clearInterval(interval);
          resolve();
        }
      }, 15);
    });
  }

  // ---------- Provider Single ----------
  const providerForm = document.getElementById('provider_single_form');
  document.getElementById('provider_single_analyze').addEventListener('click', async () => {
    const data = gatherFormData(providerForm);
    try {
      const res = await fetch('/predict_provider_single', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });
      const json = await res.json();
      showPrediction('provider_single_result', json);
    } catch (e) {
      document.getElementById('provider_single_result').innerHTML = `<pre style="color:red">${e}</pre>`;
    }
  });

  document.getElementById('provider_single_ai_btn').addEventListener('click', async () => {
    const data = gatherFormData(providerForm);
    const resultContainer = document.getElementById('provider_single_result');

    let prediction = '';
    let fraud_probability = 0;

    if (resultContainer.innerText.includes('Prediction:')) {
      const lines = resultContainer.innerText.split('\n');
      lines.forEach(line => {
        if (line.startsWith('Prediction:')) prediction = line.replace('Prediction:', '').trim();
        if (line.startsWith('Fraud Probability:')) fraud_probability = parseFloat(line.replace('Fraud Probability:', '').trim());
      });
    } else {
      alert("Please run Analyze first to get prediction.");
      return;
    }

    data.prediction = prediction;
    data.fraud_probability = fraud_probability;

    try {
      const res = await fetch('/provider_single_ai_summary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });

      if (!res.ok) {
        const text = await res.text();
        console.error("Server error:", text);
        alert("Server error. Check console.");
        return;
      }

      const json = await res.json();
      await typeOutSummary('provider_single_result', json);
    } catch (e) {
      resultContainer.innerHTML = `<pre style="color:red">${e}</pre>`;
    }
  });

  document.getElementById('provider_single_view_dashboard').addEventListener('click', () => {
    window.open('/dashboard', '_blank');
  });

  // ---------- Claim Single ----------
  const claimForm = document.getElementById('claim_single_form');
  document.getElementById('claim_single_analyze').addEventListener('click', async () => {
    const data = gatherFormData(claimForm);
    try {
      const res = await fetch('/predict_claim_single', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });
      const json = await res.json();
      showPrediction('claim_single_result', json);
    } catch (e) {
      document.getElementById('claim_single_result').innerHTML = `<pre style="color:red">${e}</pre>`;
    }
  });

  document.getElementById('claim_single_ai_btn').addEventListener('click', async () => {
    const data = gatherFormData(claimForm);
    try {
      const res = await fetch('/claim_single_ai_summary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });

      if (!res.ok) {
        const text = await res.text();
        console.error("Server error:", text);
        alert("Server error. Check console.");
        return;
      }

      const json = await res.json();
      await typeOutSummary('claim_single_result', json);
    } catch (e) {
      document.getElementById('claim_single_result').innerHTML = `<pre style="color:red">${e}</pre>`;
    }
  });

  document.getElementById('claim_single_view_dashboard').addEventListener('click', () => {
    window.open('/dashboard', '_blank');
  });

  // ---------- NEW: JSON to HTML table helper ----------
  function jsonToTable(jsonArray) {
    if (!jsonArray || !jsonArray.length) return '<p>No data</p>';
    const columns = Object.keys(jsonArray[0]);
    let table = '<table border="1" style="border-collapse:collapse; width:100%">';
    table += '<thead><tr>';
    columns.forEach(col => table += `<th style="padding:6px; background:#f0f0f0">${col}</th>`);
    table += '</tr></thead>';
    table += '<tbody>';
    jsonArray.forEach(row => {
      table += '<tr>';
      columns.forEach(col => table += `<td style="padding:6px; text-align:center">${row[col]}</td>`);
      table += '</tr>';
    });
    table += '</tbody></table>';
    return table;
  }

  // ---------- Provider Batch ----------
  document.getElementById('provider_batch_analyze').addEventListener('click', async () => {
    const fileInput = document.getElementById('provider_batch_file');
    if (!fileInput.files.length) { alert('Please choose a file.'); return; }
    const fd = new FormData();
    fd.append('file', fileInput.files[0]);
    const res = await fetch('/upload_provider_batch', { method: 'POST', body: fd });
    const json = await res.json();
    handleBatchResult(json, 'provider', 'provider_batch_result');
  });

  document.getElementById('provider_batch_ai').addEventListener('click', async () => {
    const container = document.getElementById('provider_batch_result');
    const reportFile = container.dataset.reportFile;
    if (!reportFile) {
      alert('No batch report available. Please run Analyse first.');
      return;
    }

    try {
      const res = await fetch('/provider_batch_ai_summary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ report_file: reportFile })
      });

      if (!res.ok) {
        const text = await res.text();
        console.error("Server error:", text);
        alert("Server error. Check console.");
        return;
      }

      const json = await res.json();
      await typeOutSummary('provider_batch_result', json);
    } catch (e) {
      container.innerHTML = `<pre style="color:red">${e}</pre>`;
    }
  });

  // ---------- Claim Batch ----------
  document.getElementById('claim_batch_analyze').addEventListener('click', async () => {
    const fileInput = document.getElementById('claim_batch_file');
    if (!fileInput.files.length) { alert('Please choose a file.'); return; }
    const fd = new FormData();
    fd.append('file', fileInput.files[0]);
    const res = await fetch('/upload_claim_batch', { method: 'POST', body: fd });
    const json = await res.json();
    handleBatchResult(json, 'claim', 'claim_batch_result');
  });

  document.getElementById('claim_batch_ai').addEventListener('click', async () => {
    const container = document.getElementById('claim_batch_result');
    const reportFile = container.dataset.reportFile;
    if (!reportFile) {
      alert('No batch report available. Please run Analyse first.');
      return;
    }

    try {
      const res = await fetch('/claim_batch_ai_summary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ report_file: reportFile })
      });

      if (!res.ok) {
        const text = await res.text();
        console.error("Server error:", text);
        alert("Server error. Check console.");
        return;
      }

      const json = await res.json();
      await typeOutSummary('claim_batch_result', json);
    } catch (e) {
      container.innerHTML = `<pre style="color:red">${e}</pre>`;
    }
  });

  // ---------- Shared batch handler ----------
  function handleBatchResult(json, type, containerId) {
    const container = document.getElementById(containerId);
    if (json.error) {
      container.innerHTML = `<pre style="color:red">${json.error}</pre>`;
      return;
    } else {
      if (json.report_path) container.dataset.reportFile = json.report_path.split('/').pop();

      container.innerHTML = `
        <div style="padding:12px;border-radius:8px;background:rgba(255,255,255,0.9)">
          <p><strong>${json.message}</strong></p>
          <p>Total records: ${json.total_records}</p>
          <div style="margin-top:8px;">
            <h4>Preview (first 10 rows)</h4>
            <div style="max-height:300px;overflow:auto">
              ${jsonToTable((json.summary_preview || []).slice(0, 10))}
            </div>
          </div>
          <div style="margin-top:10px;display:flex;gap:10px;">
            <button onclick="document.getElementById('${containerId}').dispatchEvent(new CustomEvent('requestAI'))" class="cta ai">AI Summary</button>
            <button onclick="window.open('/dashboard','_blank')" class="cta dashboard">View Dashboard</button>
            ${json.report_path ? `<button onclick="openReport('${json.report_path}')" class="cta">Download Report</button>` : ''}
          </div>
        </div>
      `;
    }
  }

  // Listen for requestAI custom event
  document.getElementById('provider_batch_result').addEventListener('requestAI', async function () {
    const reportFile = this.dataset.reportFile;
    if (!reportFile) { alert('No report found. Please run Analyse first.'); return; }
    try {
      const res = await fetch('/provider_batch_ai_summary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ report_file: reportFile })
      });
      const json = await res.json();
      await typeOutSummary('provider_batch_result', json);
    } catch (e) { this.innerHTML = `<pre style="color:red">${e}</pre>`; }
  });

  document.getElementById('claim_batch_result').addEventListener('requestAI', async function () {
    const reportFile = this.dataset.reportFile;
    if (!reportFile) { alert('No report found. Please run Analyse first.'); return; }
    try {
      const res = await fetch('/claim_batch_ai_summary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ report_file: reportFile })
      });
      const json = await res.json();
      await typeOutSummary('claim_batch_result', json);
    } catch (e) { this.innerHTML = `<pre style="color:red">${e}</pre>`; }
  });

  // global helper to open report links
  window.openReport = function (path) {
    window.open(path, '_blank');
  };
});
