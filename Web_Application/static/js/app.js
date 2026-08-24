// Stitch AI Data Workspace - Precision Engine JS Engine

document.addEventListener('DOMContentLoaded', () => {
  initTabNavigation();
  initCorrelationCanvas();
  initDistributionCanvas();
  initLogStreamer();
});

// 1. Tab Navigation between 4 Dashboards
function initTabNavigation() {
  const tabs = document.querySelectorAll('.nav-tab');
  const views = document.querySelectorAll('.dashboard-view');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetView = tab.getAttribute('data-tab');

      tabs.forEach(t => t.classList.remove('active'));
      views.forEach(v => v.classList.remove('active'));

      tab.classList.add('active');
      document.getElementById(targetView).classList.add('active');

      if (targetView === 'eda-view') {
        initCorrelationCanvas();
        initDistributionCanvas();
      }
    });
  });
}

// 2. Correlation Matrix Heatmap Renderer (Canvas)
function initCorrelationCanvas() {
  const canvas = document.getElementById('correlationCanvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const width = canvas.width = canvas.parentElement.clientWidth || 500;
  const height = canvas.height = 300;

  ctx.clearRect(0, 0, width, height);

  const features = ['Revenue', 'Units', 'Discount', 'Customer_Age', 'Margin'];
  const grid = [
    [1.00, 0.85, -0.42, 0.12, 0.78],
    [0.85, 1.00, -0.35, 0.08, 0.65],
    [-0.42, -0.35, 1.00, -0.05, -0.58],
    [0.12, 0.08, -0.05, 1.00, 0.15],
    [0.78, 0.65, -0.58, 0.15, 1.00]
  ];

  const size = Math.min((width - 100) / 5, (height - 60) / 5);
  const startX = 90;
  const startY = 20;

  for (let i = 0; i < 5; i++) {
    // Row Labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px JetBrains Mono';
    ctx.fillText(features[i], 10, startY + i * size + size / 1.6);

    for (let j = 0; j < 5; j++) {
      // Column Labels (top row)
      if (i === 0) {
        ctx.fillText(features[j].substring(0, 4), startX + j * size + 8, 15);
      }

      const val = grid[i][j];
      let color;
      if (val > 0) {
        const intensity = Math.floor(val * 220);
        color = `rgba(0, 242, 255, ${val})`;
      } else {
        const intensity = Math.floor(Math.abs(val) * 220);
        color = `rgba(244, 63, 94, ${Math.abs(val)})`;
      }

      ctx.fillStyle = color;
      ctx.fillRect(startX + j * size, startY + i * size, size - 2, size - 2);

      ctx.fillStyle = val > 0.5 || val < -0.4 ? '#000' : '#fff';
      ctx.font = '10px JetBrains Mono';
      ctx.fillText(val.toFixed(2), startX + j * size + 6, startY + i * size + size / 1.6);
    }
  }
}

// 3. Distribution & Scatter Bar Chart Renderer (Canvas)
function initDistributionCanvas() {
  const canvas = document.getElementById('distributionCanvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const width = canvas.width = canvas.parentElement.clientWidth || 500;
  const height = canvas.height = 300;

  ctx.clearRect(0, 0, width, height);

  const data = [420, 680, 1100, 950, 1420, 1850, 2100, 1650, 1200, 800, 510];
  const maxVal = 2500;
  const barWidth = (width - 60) / data.length;
  const startX = 40;
  const startY = height - 40;

  // Grid lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  ctx.lineWidth = 1;
  for (let y = 0; y <= 4; y++) {
    const yPos = startY - (y / 4) * (height - 60);
    ctx.beginPath();
    ctx.moveTo(startX, yPos);
    ctx.lineTo(width - 20, yPos);
    ctx.stroke();
  }

  // Draw Bars with Cyan Gradient
  data.forEach((val, i) => {
    const barHeight = (val / maxVal) * (height - 60);
    const x = startX + i * barWidth;
    const y = startY - barHeight;

    const grad = ctx.createLinearGradient(0, y, 0, startY);
    grad.addColorStop(0, '#00f2ff');
    grad.addColorStop(1, 'rgba(0, 242, 255, 0.1)');

    ctx.fillStyle = grad;
    ctx.fillRect(x + 4, y, barWidth - 8, barHeight);

    ctx.strokeStyle = '#00f2ff';
    ctx.strokeRect(x + 4, y, barWidth - 8, barHeight);
  });
}

// 4. CrewAI Agent Execution Logs Streamer
function initLogStreamer() {
  const logContainer = document.getElementById('terminalLogs');
  const runBtn = document.getElementById('runCrewBtn');
  if (!logContainer || !runBtn) return;

  const sampleLogs = [
    { time: '00:01.12', agent: 'DataCleanerAgent', msg: 'Started ingestion for sales_data.csv [12,450 rows]', class: 'log-cyan' },
    { time: '00:01.45', agent: 'DataCleanerAgent', msg: 'Missing value audit completed: 0.4% imputed via median strategy.', class: 'log-green' },
    { time: '00:02.10', agent: 'EDASpecialistAgent', msg: 'Computing Pearson correlation matrix across 18 numerical features...', class: 'log-cyan' },
    { time: '00:02.85', agent: 'EDASpecialistAgent', msg: 'Strong positive correlation detected between Revenue and Units (r=0.85).', class: 'log-green' },
    { time: '00:03.40', agent: 'ModelingExpertAgent', msg: 'Training Random Forest Regressor on churn risk indicators...', class: 'log-amber' },
    { time: '00:04.22', agent: 'ModelingExpertAgent', msg: 'Model evaluation score: R² = 0.941, MAE = 4.12.', class: 'log-green' },
    { time: '00:05.00', agent: 'ExecutiveReporterAgent', msg: 'Synthesizing automated executive summary and action items.', class: 'log-cyan' },
    { time: '00:05.60', agent: 'CrewAI Engine', msg: 'Workflow execution completed successfully in 5.60s.', class: 'log-green' }
  ];

  runBtn.addEventListener('click', () => {
    logContainer.innerHTML = '';
    let index = 0;
    runBtn.disabled = true;
    runBtn.textContent = '⏳ Execution Running...';

    const interval = setInterval(() => {
      if (index >= sampleLogs.length) {
        clearInterval(interval);
        runBtn.disabled = false;
        runBtn.textContent = '▶ Run Crew Workflow';
        return;
      }

      const log = sampleLogs[index];
      const div = document.createElement('div');
      div.className = 'log-line';
      div.innerHTML = `<span class="log-time">[${log.time}]</span> <span class="log-agent">${log.agent}:</span> <span class="log-msg ${log.class}">${log.msg}</span>`;
      logContainer.appendChild(div);
      logContainer.scrollTop = logContainer.scrollHeight;
      index++;
    }, 600);
  });
}
