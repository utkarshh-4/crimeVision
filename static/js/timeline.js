/* ==========================================================================
   CrimeVision AI - Timeline & Temporal Charts
   ========================================================================== */

window.CV = window.CV || {};

CV.timeline = (function() {
  let timelineChartInstance = null;
  let countsChartInstance = null;
  let currentPlayheadSec = null;

  // Custom Chart.js Playhead Plugin
  const playheadPlugin = {
    id: 'playhead',
    afterDatasetsDraw(chart) {
      if (typeof currentPlayheadSec !== 'number') return;
      const ctx = chart.ctx;
      const xAxis = chart.scales.x;
      const yAxis = chart.scales.y;
      if (!xAxis || !yAxis) return;

      const xPos = xAxis.getPixelForValue(currentPlayheadSec);
      if (xPos >= xAxis.left && xPos <= xAxis.right) {
        ctx.save();
        ctx.beginPath();
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.moveTo(xPos, yAxis.top);
        ctx.lineTo(xPos, yAxis.bottom);
        ctx.stroke();

        // Playhead handle
        ctx.fillStyle = '#38bdf8';
        ctx.beginPath();
        ctx.arc(xPos, yAxis.top + 4, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
    }
  };

  /**
   * Render horizontal floating bar chart of multimodal fused timeline.
   */
  function renderTimelineChart(events) {
    const canvas = document.getElementById('timelineChart');
    if (!canvas || typeof Chart === 'undefined') return;

    if (timelineChartInstance) {
      timelineChartInstance.destroy();
      timelineChartInstance = null;
    }

    if (!Array.isArray(events) || events.length === 0) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    const labels = [];
    const data = [];
    const bgColors = [];
    const borderColors = [];
    const borderWidths = [];

    const modalityColors = {
      fused: 'rgba(168, 85, 247, 0.75)',
      video: 'rgba(59, 130, 246, 0.75)',
      audio: 'rgba(245, 158, 11, 0.75)',
      image: 'rgba(20, 184, 166, 0.75)'
    };

    events.forEach(function(ev) {
      const eid = ev.id || 'E';
      const cat = ev.category || 'Event';
      const st = typeof ev.start_raw === 'number' ? ev.start_raw : 0;
      let et = typeof ev.end_raw === 'number' ? ev.end_raw : st;
      // Visual minimum span for zero-length events
      if (et <= st) {
        et = st + 0.25;
      }

      const label = `[${eid}] ${cat}: ${(ev.description || '').substring(0, 30)}`;
      labels.push(label);
      data.push([st, et]);

      const mod = (ev.modality || 'video').toLowerCase();
      bgColors.push(modalityColors[mod] || 'rgba(148, 163, 184, 0.6)');

      if (ev.conflict) {
        borderColors.push('#ef4444');
        borderWidths.push(2.5);
      } else if (ev.is_anchor) {
        borderColors.push('#ffffff');
        borderWidths.push(2.5);
      } else {
        borderColors.push('rgba(255, 255, 255, 0.1)');
        borderWidths.push(1);
      }
    });

    timelineChartInstance = new Chart(canvas, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Event Duration (s)',
          data: data,
          backgroundColor: bgColors,
          borderColor: borderColors,
          borderWidth: borderWidths,
          borderRadius: 4,
          borderSkipped: false,
          barPercentage: 0.65
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: function(items) {
                if (!items.length) return '';
                const idx = items[0].dataIndex;
                const ev = events[idx];
                return `[${ev.id}] ${ev.category} (${ev.modality})`;
              },
              label: function(item) {
                const idx = item.dataIndex;
                const ev = events[idx];
                return [
                  `Time: ${ev.start_raw}s - ${ev.end_raw}s`,
                  `Confidence: ${ev.confidence}`,
                  `Detail: ${ev.description}`
                ];
              }
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Timeline (Seconds)', color: '#94a3b8', font: { size: 11 } },
            grid: { color: 'rgba(148, 163, 184, 0.08)' },
            ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10.5 } }
          },
          y: {
            grid: { display: false },
            ticks: { color: '#cbd5e1', font: { size: 11 } }
          }
        },
        onClick: function(evt) {
          const points = timelineChartInstance.getElementsAtEventForMode(evt, 'nearest', { intersect: true }, true);
          if (points.length) {
            const idx = points[0].index;
            const ev = events[idx];
            if (ev && typeof ev.start_raw === 'number' && typeof CV.seekMedia === 'function') {
              CV.seekMedia(ev.start_raw);
            }
          }
        }
      }
    });
  }

  /**
   * Render people and weapon tracking line chart with video playhead tracker.
   */
  function renderCountsChart(frames) {
    const canvas = document.getElementById('countsChart');
    if (!canvas || typeof Chart === 'undefined') return;

    if (countsChartInstance) {
      countsChartInstance.destroy();
      countsChartInstance = null;
    }

    if (!Array.isArray(frames) || frames.length === 0) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    const labels = [];
    const peopleData = [];
    const weaponData = [];

    frames.forEach(function(f) {
      const t = typeof f.t === 'number' ? f.t : 0;
      labels.push(t);
      peopleData.push(typeof f.people_count === 'number' ? f.people_count : 0);
      weaponData.push(typeof f.weapon_count === 'number' ? f.weapon_count : 0);
    });

    countsChartInstance = new Chart(canvas, {
      type: 'line',
      plugins: [playheadPlugin],
      data: {
        labels: labels,
        datasets: [
          {
            label: 'People Count',
            data: peopleData,
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            borderWidth: 2,
            tension: 0.2,
            pointRadius: 2,
            fill: true
          },
          {
            label: 'Weapons Detected',
            data: weaponData,
            borderColor: '#ef4444',
            backgroundColor: 'rgba(239, 68, 68, 0.15)',
            borderWidth: 2,
            tension: 0.2,
            pointRadius: 3,
            fill: true
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            position: 'top',
            labels: { color: '#94a3b8', font: { size: 11 }, boxWidth: 12 }
          },
          tooltip: {
            mode: 'index',
            intersect: false,
            callbacks: {
              title: function(items) {
                return items.length ? `At t=${items[0].label}s` : '';
              }
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Time (Seconds)', color: '#94a3b8', font: { size: 11 } },
            grid: { color: 'rgba(148, 163, 184, 0.08)' },
            ticks: {
              color: '#94a3b8',
              font: { family: 'JetBrains Mono', size: 10.5 },
              callback: function(val) {
                return `${val}s`;
              }
            }
          },
          y: {
            beginAtZero: true,
            suggestedMax: 3,
            grid: { color: 'rgba(148, 163, 184, 0.08)' },
            ticks: {
              stepSize: 1,
              color: '#94a3b8',
              font: { family: 'JetBrains Mono', size: 10.5 }
            }
          }
        },
        onClick: function(evt) {
          const points = countsChartInstance.getElementsAtEventForMode(evt, 'index', { intersect: false }, true);
          if (points.length) {
            const idx = points[0].index;
            const t = labels[idx];
            if (typeof t === 'number' && typeof CV.seekMedia === 'function') {
              CV.seekMedia(t);
            }
          }
        }
      }
    });
  }

  /**
   * Update the vertical playhead tracker position on the counts chart.
   */
  function updatePlayhead(timeSec) {
    currentPlayheadSec = timeSec;
    if (countsChartInstance) {
      countsChartInstance.update('none');
    }
  }

  return {
    renderTimelineChart: renderTimelineChart,
    renderCountsChart: renderCountsChart,
    updatePlayhead: updatePlayhead
  };
})();
