const fileInput = document.getElementById('fileInput');
const uploadBox = document.getElementById('uploadBox');
const uploadContent = document.getElementById('uploadContent');
const fileInfoBox = document.getElementById('fileInfoBox');
const fileThumbnail = document.getElementById('fileThumbnail');
const fileName = document.getElementById('fileName');
const fileMeta = document.getElementById('fileMeta');
const removeFileBtn = document.getElementById('removeFileBtn');
const chooseImgBtn = document.getElementById('chooseImgBtn');

const analyzeBtn = document.getElementById('analyzeBtn');
const apiUrlInput = document.getElementById('apiUrl');
const thresholdInput = document.getElementById('threshold');
const statusEl = document.getElementById('status');

// Elements hiển thị
const originalImg = document.getElementById('originalImg');
const originalPlaceholder = document.getElementById('originalPlaceholder');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const canvasWrap = document.getElementById('canvasWrap');
const resultPlaceholder = document.getElementById('resultPlaceholder');

const detectedCountBadge = document.getElementById('detectedCountBadge');
const confidenceBar = document.getElementById('confidenceBar');
const avgConfidenceEl = document.getElementById('avgConfidence');
const confidenceProgress = document.getElementById('confidenceProgress');

const totalDetectedEl = document.getElementById('totalDetected');
const totalRecognizedEl = document.getElementById('totalRecognized');
const totalProductsEl = document.getElementById('totalProducts');
const totalTypesEl = document.getElementById('totalTypes');
const countsTableBody = document.querySelector('#countsTable tbody');

let currentImage = null;   
let currentFile = null;    
let lastResult = null;     
let donutChart = null;

const colors = [
  '#ef4444', '#f59e0b', '#22c55e', '#3b82f6', '#8b5cf6', 
  '#ec4899', '#14b8a6', '#f97316', '#6366f1', '#84cc16'
];

function setStatus(msg, type) {
  statusEl.textContent = msg;
  statusEl.style.color = type === 'error' ? '#ef4444' : '#22c55e';
}

function formatBytes(bytes, decimals = 2) {
    if (!+bytes) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}

function loadImageToUI(file) {
  const img = new Image();
  const reader = new FileReader();
  reader.onload = (e) => {
    img.onload = () => {
      currentImage = img;
      
      // Hiển thị ảnh gốc ở khung bên cạnh
      originalImg.src = e.target.result;
      originalPlaceholder.style.display = 'none';
      fileInfoBox.style.display = 'block';

      // Chuẩn bị canvas (xóa kết quả cũ nếu có)
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.drawImage(img, 0, 0);
      resultPlaceholder.style.display = 'none';
      canvas.style.display = 'block';
      detectedCountBadge.style.display = 'none';
      confidenceBar.style.display = 'none';
      
      analyzeBtn.disabled = false;
      setStatus('', '');
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

// Xử lý nút chọn ảnh
chooseImgBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) {
    currentFile = fileInput.files[0];
    loadImageToUI(currentFile);
  }
});

removeFileBtn.addEventListener('click', () => {
  currentFile = null;
  currentImage = null;
  fileInput.value = '';
  
  fileInfoBox.style.display = 'none';
  originalPlaceholder.style.display = 'flex';
  originalImg.src = '';
  
  canvas.style.display = 'none';
  resultPlaceholder.style.display = 'block';
  
  analyzeBtn.disabled = true;
});

// Xử lý Drag & Drop
['dragover', 'dragenter'].forEach(evt =>
  uploadBox.addEventListener(evt, (e) => { e.preventDefault(); uploadBox.classList.add('dragover'); })
);
['dragleave', 'drop'].forEach(evt =>
  uploadBox.addEventListener(evt, (e) => { e.preventDefault(); uploadBox.classList.remove('dragover'); })
);
uploadBox.addEventListener('drop', (e) => {
  const file = e.dataTransfer.files[0];
  if (file) {
    currentFile = file;
    loadImageToUI(file);
  }
});

function getColorForLabel(label, index) {
  return colors[index % colors.length];
}

function redrawWithBoxes(products, minReliability) {
  ctx.drawImage(currentImage, 0, 0);
  ctx.lineWidth = Math.max(3, canvas.width / 300);
  ctx.font = `bold ${Math.max(14, canvas.width / 60)}px sans-serif`;

  // Nhóm các EAN để lấy index màu thống nhất
  const uniqueLabels = [...new Set(products.map(p => p.EAN))];

  products.forEach(p => {
    if (p.reliability < minReliability) return;
    const [x1, y1, x2, y2] = p.box;
    const colorIdx = uniqueLabels.indexOf(p.EAN);
    const color = getColorForLabel(p.EAN, colorIdx !== -1 ? colorIdx : 0);

    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const label = `${p.EAN} ${(p.reliability*100).toFixed(0)}%`;
    const textWidth = ctx.measureText(label).width;
    const pad = 6;
    const h = Math.max(20, canvas.width / 40);

    ctx.fillStyle = color;
    ctx.fillRect(x1, Math.max(0, y1 - h), textWidth + pad * 2, h);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x1 + pad, Math.max(h - 4, y1 - 4));
  });
}

function renderStatsAndChart(counts) {
  countsTableBody.innerHTML = '';
  
  const labels = [];
  const dataVals = [];
  const bgColors = [];
  
  const sortedEntries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const totalItems = sortedEntries.reduce((sum, [_, count]) => sum + count, 0);

  sortedEntries.forEach(([ean, count], idx) => {
      const color = getColorForLabel(ean, idx);
      const ratio = totalItems > 0 ? Math.round((count / totalItems) * 100) : 0;
      
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><span class="label-dot" style="background:${color}"></span>${ean}</td>
        <td class="text-right">${count}</td>
        <td class="text-right">${ratio}%</td>
      `;
      countsTableBody.appendChild(tr);

      labels.push(ean);
      dataVals.push(count);
      bgColors.push(color);
  });

  // Vẽ biểu đồ
  const ctxChart = document.getElementById('donutChart').getContext('2d');
  if (donutChart) donutChart.destroy();

  donutChart = new Chart(ctxChart, {
      type: 'doughnut',
      data: {
          labels: labels,
          datasets: [{
              data: dataVals,
              backgroundColor: bgColors,
              borderWidth: 0
          }]
      },
      options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
              legend: { position: 'right', labels: { boxWidth: 12, font: { size: 11 } } }
          },
          cutout: '65%'
      }
  });
}

analyzeBtn.addEventListener('click', async () => {
  if (!currentFile) return;

  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang phân tích...';
  setStatus('Đang gửi ảnh tới backend...', '');

  const apiUrl = apiUrlInput.value.trim();
  const formData = new FormData();
  formData.append('file', currentFile);

  try {
    const res = await fetch(apiUrl, { method: 'POST', body: formData });
    if (!res.ok) throw new Error(`Server trả lỗi ${res.status}`);
    const data = await res.json();
    lastResult = data;

    totalDetectedEl.textContent = data.total_detected || 0;
    totalRecognizedEl.textContent = data.total_recognized || 0;
    
    // Tính thêm 2 chỉ số
    const counts = data.counts || {};
    const totalTypes = Object.keys(counts).length;
    const totalProducts = Object.values(counts).reduce((a, b) => a + b, 0);
    
    totalTypesEl.textContent = totalTypes;
    totalProductsEl.textContent = totalProducts;

    renderStatsAndChart(counts);

    const threshold = parseFloat(thresholdInput.value) || 0;
    redrawWithBoxes(data.products || [], threshold);
    
    // Update badge & confidence
    detectedCountBadge.textContent = `Đã phát hiện: ${totalTypes} nhãn`;
    detectedCountBadge.style.display = 'inline-block';
    
    if (data.products && data.products.length > 0) {
      const avg = data.products.reduce((acc, p) => acc + p.reliability, 0) / data.products.length;
      const avgPct = Math.round(avg * 100);
      avgConfidenceEl.textContent = `${avgPct}%`;
      confidenceProgress.style.width = `${avgPct}%`;
      confidenceBar.style.display = 'block';
    }

    setStatus('', '');
  } catch (err) {
    setStatus(`Lỗi: ${err.message}`, 'error');
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Phân tích';
  }
});

function updateThreshold(val) {
  let currentVal = parseFloat(thresholdInput.value) || 0;
  let newVal = currentVal + val;
  // Giới hạn từ -1 đến 1
  newVal = Math.max(-1, Math.min(1, newVal));
  thresholdInput.value = newVal.toFixed(1);
  thresholdInput.dispatchEvent(new Event('change'));
}

document.getElementById('decThreshold').addEventListener('click', () => updateThreshold(-0.1));
document.getElementById('incThreshold').addEventListener('click', () => updateThreshold(0.1));

thresholdInput.addEventListener('change', () => {
  if (lastResult) {
    const threshold = parseFloat(thresholdInput.value) || 0;
    redrawWithBoxes(lastResult.products || [], threshold);
  }
});