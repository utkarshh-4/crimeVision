/* ==========================================================================
   CrimeVision AI - Main Client Application Orchestrator
   ========================================================================== */

window.CV = window.CV || {};

CV.state = {
  currentSessionId: null,
  sessionData: null,
  pollingTimer: null,
  activeTab: 'evidence',
  chatHistory: []
};

CV.ui = (function() {
  const toastContainer = document.getElementById('toastContainer');
  const overallStatusBadge = document.getElementById('overallStatusBadge');
  const btnAnalyze = document.getElementById('btnAnalyze');
  const sessionText = document.getElementById('sessionText');

  /**
   * Display a temporary non-blocking toast notification.
   */
  function showToast(message, type = 'info') {
    if (!toastContainer) return;
    const toast = document.createElement('div');
    toast.className = `cv-toast ${type}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);

    setTimeout(() => {
      if (toast.parentNode) {
        toast.parentNode.removeChild(toast);
      }
    }, 4000);
  }

  /**
   * Update header device and Groq LLM status indicators.
   */
  async function fetchHealth() {
    try {
      const res = await fetch('/api/health');
      if (!res.ok) return;
      const data = await res.json();

      const devicePill = document.getElementById('devicePill');
      if (devicePill) {
        const dot = devicePill.querySelector('.cv-pill-dot');
        const text = devicePill.querySelector('.cv-pill-text');
        if (dot) dot.className = 'cv-pill-dot active';
        if (text) text.textContent = `Device: ${String(data.device || 'CPU').toUpperCase()}`;
      }

      const groqPill = document.getElementById('groqPill');
      if (groqPill) {
        const dot = groqPill.querySelector('.cv-pill-dot');
        const text = groqPill.querySelector('.cv-pill-text');
        if (data.groq_configured) {
          if (dot) dot.className = 'cv-pill-dot active';
          if (text) text.textContent = 'Groq LLM: Active';
        } else {
          if (dot) dot.className = 'cv-pill-dot warning';
          if (text) text.textContent = 'Groq LLM: Fallback';
        }
      }
    } catch (err) {
      console.warn('Health check failed:', err);
    }
  }

  /**
   * Switch center view navigation tabs.
   */
  function switchTab(tabName) {
    CV.state.activeTab = tabName;
    const tabButtons = document.querySelectorAll('.cv-tab');
    const tabViews = document.querySelectorAll('.cv-tab-view');

    tabButtons.forEach(btn => {
      if (btn.dataset.tab === tabName) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });

    tabViews.forEach(view => {
      if (view.id === `view${tabName.charAt(0).toUpperCase() + tabName.slice(1)}`) {
        view.classList.add('active');
      } else {
        view.classList.remove('active');
      }
    });

    // Re-render / resize specific tab components
    if (tabName === 'timeline' && CV.state.sessionData) {
      if (CV.state.sessionData.fusion && Array.isArray(CV.state.sessionData.fusion.timeline)) {
        CV.timeline.renderTimelineChart(CV.state.sessionData.fusion.timeline);
      }
      if (CV.state.sessionData.video && Array.isArray(CV.state.sessionData.video.frames)) {
        CV.timeline.renderCountsChart(CV.state.sessionData.video.frames);
      }
    } else if (tabName === 'evidence' && CV.state.sessionData) {
      drawDetectionBoxes();
    } else if (tabName === 'scene3d') {
      if (typeof window.initScene3D === 'function') {
        window.initScene3D('scene3d', CV.state.sessionData);
      }
    }
  }

  /**
   * Update progress badges for pipeline stages.
   */
  function updateProgress(progress, status) {
    if (overallStatusBadge) {
      overallStatusBadge.className = `cv-stage-badge ${status || 'pending'}`;
      overallStatusBadge.textContent = (status || 'pending').toUpperCase();
    }

    const stages = ['image', 'audio', 'video', 'fusion'];
    stages.forEach(st => {
      const el = document.getElementById(`stage${st.charAt(0).toUpperCase() + st.slice(1)}`);
      if (!el) return;
      const state = (progress && progress[st]) ? progress[st] : 'pending';
      el.className = `cv-stage-item ${state}`;
      const badge = el.querySelector('.cv-stage-badge');
      if (badge) {
        badge.className = `cv-stage-badge ${state}`;
        badge.textContent = state.toUpperCase();
      }
    });
  }

  /**
   * Draw RT-DETR detection bounding boxes on the overlay canvas.
   */
  function drawDetectionBoxes() {
    const img = document.getElementById('evidenceImg');
    const canvas = document.getElementById('detCanvas');
    if (!img || !canvas || !CV.state.sessionData || !CV.state.sessionData.image) return;

    const detections = CV.state.sessionData.image.detections || [];
    if (!Array.isArray(detections)) return;

    const ctx = canvas.getContext('2d');
    const displayW = img.clientWidth || img.naturalWidth;
    const displayH = img.clientHeight || img.naturalHeight;

    if (!displayW || !displayH) return;

    canvas.width = displayW;
    canvas.height = displayH;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const naturalW = img.naturalWidth || displayW;
    const naturalH = img.naturalHeight || displayH;

    const scaleX = displayW / naturalW;
    const scaleY = displayH / naturalH;

    detections.forEach(det => {
      const box = det.box || [];
      if (box.length < 4) return;

      const x1 = box[0] * scaleX;
      const y1 = box[1] * scaleY;
      const x2 = box[2] * scaleX;
      const y2 = box[3] * scaleY;
      const w = Math.max(0, x2 - x1);
      const h = Math.max(0, y2 - y1);

      // Color scheme: Red=Weapon, Green=Human, Blue=Object
      let strokeColor = '#3b82f6';
      let bgColor = 'rgba(59, 130, 246, 0.85)';

      if (det.is_weapon || String(det.label).toLowerCase() === 'weapon') {
        strokeColor = '#ef4444';
        bgColor = 'rgba(239, 68, 68, 0.85)';
      } else if (String(det.label).toLowerCase() === 'human') {
        strokeColor = '#10b981';
        bgColor = 'rgba(16, 185, 129, 0.85)';
      }

      // Box outline
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, w, h);

      // Tag label
      const tagText = `${det.raw_label || det.label} ${(det.confidence * 100).toFixed(0)}%`;
      ctx.font = '600 10.5px JetBrains Mono, monospace';
      const textMetrics = ctx.measureText(tagText);
      const tagW = textMetrics.width + 8;
      const tagH = 16;

      ctx.fillStyle = bgColor;
      ctx.fillRect(x1, Math.max(0, y1 - tagH), tagW, tagH);

      ctx.fillStyle = '#ffffff';
      ctx.fillText(tagText, x1 + 4, Math.max(11, y1 - 4));
    });
  }

  /**
   * Render complete forensic session into the dashboard in place.
   */
  function renderSession(session) {
    if (!session || typeof session !== 'object') return;
    CV.state.sessionData = session;

    if (typeof window.initScene3D === 'function') {
      window.initScene3D(session);
    }

    if (sessionText) {
      sessionText.textContent = `Session: ${(session.id || '').substring(0, 8)}...`;
    }

    // 1. Metric Badges
    const badgeScene = document.getElementById('badgeScene');
    const badgeDuration = document.getElementById('badgeDuration');
    const badgePeople = document.getElementById('badgePeople');
    const badgeWeapons = document.getElementById('badgeWeapons');
    const badgeCrime = document.getElementById('badgeCrime');

    let sceneLabel = 'Unknown';
    if (session.image && session.image.scene_classification) {
      sceneLabel = session.image.scene_classification.label || 'Unknown';
    } else if (session.audio && session.audio.facts && session.audio.facts.location_inference) {
      sceneLabel = session.audio.facts.location_inference.label || 'Unknown';
    }
    if (badgeScene) badgeScene.textContent = sceneLabel;

    let dur = 0;
    if (session.video && session.video.duration) dur = session.video.duration;
    else if (session.audio && session.audio.duration) dur = session.audio.duration;
    if (badgeDuration) badgeDuration.textContent = `${dur.toFixed(1)}s`;

    let peopleCnt = 0;
    let weaponCnt = 0;
    if (session.video && Array.isArray(session.video.frames)) {
      peopleCnt = Math.max(0, ...session.video.frames.map(f => f.people_count || 0));
      weaponCnt = Math.max(0, ...session.video.frames.map(f => f.weapon_count || 0));
    } else if (session.image) {
      peopleCnt = session.image.people_count || 0;
      weaponCnt = session.image.weapon_count || 0;
    }
    if (badgePeople) badgePeople.textContent = String(peopleCnt);
    if (badgeWeapons) badgeWeapons.textContent = String(weaponCnt);

    if (badgeCrime) {
      const cType = (session.video && session.video.crime_type) ? session.video.crime_type : 'No Crime Detected';
      const cConf = (session.video && typeof session.video.crime_confidence === 'number') ? (session.video.crime_confidence * 100).toFixed(0) : '0';
      badgeCrime.textContent = `${cType} (${cConf}%)`;
    }

    // 2. Media Viewers
    const imgEl = document.getElementById('evidenceImg');
    const imgEmpty = document.getElementById('imgEmptyMsg');
    const imgCaption = document.getElementById('imgCaptionText');
    const imgDim = document.getElementById('imgDimensions');

    if (session.media && session.media.image && session.id) {
      if (imgEl) {
        imgEl.style.display = 'block';
        imgEl.src = `/api/media/${session.id}/image?t=${Date.now()}`;
        imgEl.onload = function() {
          if (imgDim) imgDim.textContent = `${imgEl.naturalWidth}x${imgEl.naturalHeight}`;
          drawDetectionBoxes();
        };
      }
      if (imgEmpty) imgEmpty.style.display = 'none';
      if (imgCaption && session.image && session.image.caption) {
        imgCaption.style.display = 'block';
        imgCaption.textContent = `Caption: "${session.image.caption}"`;
      }
    } else {
      if (imgEl) imgEl.style.display = 'none';
      if (imgEmpty) imgEmpty.style.display = 'block';
      if (imgCaption) imgCaption.style.display = 'none';
    }

    const vidEl = document.getElementById('videoPlayer');
    const vidEmpty = document.getElementById('vidEmptyMsg');
    if (session.media && session.media.video && session.id) {
      if (vidEl) {
        vidEl.style.display = 'block';
        vidEl.src = `/api/media/${session.id}/video`;
      }
      if (vidEmpty) vidEmpty.style.display = 'none';
    } else {
      if (vidEl) vidEl.style.display = 'none';
      if (vidEmpty) vidEmpty.style.display = 'block';
    }

    const audEl = document.getElementById('audioPlayer');
    const audEmpty = document.getElementById('audEmptyMsg');
    if (session.media && session.media.audio && session.id) {
      if (audEl) {
        audEl.style.display = 'block';
        audEl.src = `/api/media/${session.id}/audio`;
      }
      if (audEmpty) audEmpty.style.display = 'none';
    } else {
      if (audEl) audEl.style.display = 'none';
      if (audEmpty) audEmpty.style.display = 'block';
    }

    // 3. Transcript Segments
    const transcriptEl = document.getElementById('transcriptSegments');
    if (transcriptEl) {
      transcriptEl.innerHTML = '';
      const segments = (session.audio && Array.isArray(session.audio.segments)) ? session.audio.segments : [];
      if (segments.length === 0) {
        const emptyDiv = document.createElement('div');
        emptyDiv.className = 'cv-media-empty';
        emptyDiv.style.padding = '10px';
        emptyDiv.textContent = session.audio ? 'No spoken dialogue transcribed.' : 'No audio evidence uploaded.';
        transcriptEl.appendChild(emptyDiv);
      } else {
        segments.forEach(seg => {
          const item = document.createElement('div');
          item.className = 'cv-transcript-segment';
          item.title = 'Click to seek media';

          const timeSpan = document.createElement('span');
          timeSpan.className = 'cv-segment-time';
          timeSpan.textContent = `[${(seg.start_raw || 0).toFixed(1)}s - ${(seg.end_raw || 0).toFixed(1)}s]`;

          const textSpan = document.createElement('span');
          textSpan.className = 'cv-segment-text';
          textSpan.textContent = seg.text || '';

          item.appendChild(timeSpan);
          item.appendChild(textSpan);

          item.addEventListener('click', () => {
            CV.seekMedia(seg.start_raw || 0);
          });

          transcriptEl.appendChild(item);
        });
      }
    }

    // 4. Crime Probability Distribution Bars
    const crimeProbEl = document.getElementById('crimeProbBars');
    if (crimeProbEl) {
      crimeProbEl.innerHTML = '';
      const probs = (session.video && session.video.crime_probabilities) ? session.video.crime_probabilities : {};
      const primaryType = (session.video && session.video.crime_type) ? session.video.crime_type : '';

      Object.keys(probs).forEach(cat => {
        const p = probs[cat] || 0;
        const item = document.createElement('div');
        item.className = `cv-prob-item ${cat === primaryType ? 'primary' : ''}`;

        const header = document.createElement('div');
        header.className = 'cv-prob-header';

        const nameSpan = document.createElement('span');
        nameSpan.textContent = cat;

        const valSpan = document.createElement('span');
        valSpan.textContent = `${(p * 100).toFixed(1)}%`;

        header.appendChild(nameSpan);
        header.appendChild(valSpan);

        const track = document.createElement('div');
        track.className = 'cv-prob-bar-track';

        const fill = document.createElement('div');
        fill.className = 'cv-prob-bar-fill';
        fill.style.width = `${Math.min(100, Math.max(0, p * 100))}%`;

        track.appendChild(fill);
        item.appendChild(header);
        item.appendChild(track);

        crimeProbEl.appendChild(item);
      });
    }

    // 5. Hypotheses Cards
    const hypothesesEl = document.getElementById('hypothesesList');
    if (hypothesesEl) {
      hypothesesEl.innerHTML = '';
      let hypoList = [];
      if (session.fusion && Array.isArray(session.fusion.hypotheses)) {
        hypoList = session.fusion.hypotheses;
      } else if (session.image && Array.isArray(session.image.hypotheses)) {
        hypoList = session.image.hypotheses;
      } else if (session.audio && session.audio.facts && Array.isArray(session.audio.facts.hypotheses)) {
        hypoList = session.audio.facts.hypotheses;
      }

      hypoList.forEach(h => {
        const card = document.createElement('div');
        card.className = 'cv-hypothesis-card';

        const top = document.createElement('div');
        top.className = 'cv-hypothesis-top';

        const title = document.createElement('span');
        title.className = 'cv-hypothesis-title';
        title.textContent = h.title || 'Hypothesis';

        const prob = document.createElement('span');
        prob.className = 'cv-hypothesis-prob';
        prob.textContent = `${((h.probability || 0) * 100).toFixed(0)}%`;

        top.appendChild(title);
        top.appendChild(prob);

        const exp = document.createElement('p');
        exp.className = 'cv-hypothesis-exp';
        exp.textContent = h.explanation || '';

        card.appendChild(top);
        card.appendChild(exp);

        hypothesesEl.appendChild(card);
      });
    }

    // 6. Conflicts Card
    const conflictsCard = document.getElementById('cardConflicts');
    const conflictsList = document.getElementById('conflictsList');
    if (conflictsCard && conflictsList) {
      const conflicts = (session.fusion && Array.isArray(session.fusion.conflicts)) ? session.fusion.conflicts : [];
      if (conflicts.length > 0) {
        conflictsCard.style.display = 'block';
        conflictsList.innerHTML = '';
        conflicts.forEach(c => {
          const item = document.createElement('div');
          item.className = 'cv-conflict-item';

          const mark = document.createElement('span');
          mark.textContent = '⚠';
          mark.style.color = '#ef4444';

          const desc = document.createElement('span');
          desc.textContent = `[${c.type || 'Discrepancy'}] ${c.description || ''}`;

          item.appendChild(mark);
          item.appendChild(desc);
          conflictsList.appendChild(item);
        });
      } else {
        conflictsCard.style.display = 'none';
      }
    }

    // 7. Report Tab Elements
    const reportSummary = document.getElementById('reportSummaryText');
    const reportConfidence = document.getElementById('reportConfidenceVal');
    const reportAnchor = document.getElementById('reportAnchorCount');
    const reportConflict = document.getElementById('reportConflictCount');

    if (reportSummary && session.fusion) {
      reportSummary.textContent = session.fusion.summary || 'Summary unavailable.';
    }
    if (reportConfidence && session.fusion) {
      reportConfidence.textContent = (session.fusion.overall_confidence || 0).toFixed(3);
    }
    if (reportAnchor && session.fusion && Array.isArray(session.fusion.anchor_events)) {
      reportAnchor.textContent = String(session.fusion.anchor_events.length);
    }
    if (reportConflict && session.fusion && Array.isArray(session.fusion.conflicts)) {
      reportConflict.textContent = String(session.fusion.conflicts.length);
    }

    // Download PDF button
    const btnDownloadReport = document.getElementById('btnDownloadReport');
    if (btnDownloadReport) {
      btnDownloadReport.onclick = function() {
        if (!session.id) return;
        window.open(`/api/report/${session.id}`, '_blank');
      };
    }

    // 8. Timeline Tab Charts
    if (session.fusion && Array.isArray(session.fusion.timeline)) {
      CV.timeline.renderTimelineChart(session.fusion.timeline);
    }
    if (session.video && Array.isArray(session.video.frames)) {
      CV.timeline.renderCountsChart(session.video.frames);
    }

    // 9. 3D Scene Tab
    if (typeof window.initScene3D === 'function') {
      window.initScene3D('scene3d', session);
    }
  }

  return {
    showToast: showToast,
    fetchHealth: fetchHealth,
    switchTab: switchTab,
    updateProgress: updateProgress,
    renderSession: renderSession,
    drawDetectionBoxes: drawDetectionBoxes
  };
})();

/**
 * Global helper to synchronize media seek across video and audio players.
 */
CV.seekMedia = function(timeSec) {
  const vid = document.getElementById('videoPlayer');
  const aud = document.getElementById('audioPlayer');

  if (vid && vid.style.display !== 'none') {
    vid.currentTime = Math.max(0, timeSec);
    vid.play().catch(() => {});
  } else if (aud && aud.style.display !== 'none') {
    aud.currentTime = Math.max(0, timeSec);
    aud.play().catch(() => {});
  }
};

/**
 * Main application bootstrap and event bindings.
 */
document.addEventListener('DOMContentLoaded', function() {
  CV.ui.fetchHealth();
  CV.chat.initChat();

  // 1. Navigation Tab Clicks
  const tabButtons = document.querySelectorAll('.cv-tab');
  tabButtons.forEach(btn => {
    btn.addEventListener('click', function() {
      CV.ui.switchTab(this.dataset.tab);
    });
  });

  // 2. File Input change previews
  ['image', 'video', 'audio'].forEach(type => {
    const input = document.getElementById(`${type}Input`);
    const nameSpan = document.getElementById(`${type}Name`);
    if (input && nameSpan) {
      input.addEventListener('change', function() {
        if (this.files && this.files.length > 0) {
          nameSpan.textContent = this.files[0].name;
          nameSpan.classList.add('selected');
        } else {
          nameSpan.textContent = `Select ${type.charAt(0).toUpperCase() + type.slice(1)} File...`;
          nameSpan.classList.remove('selected');
        }
      });
    }
  });

  // 3. Upload Form Submission
  const uploadForm = document.getElementById('uploadForm');
  const btnAnalyze = document.getElementById('btnAnalyze');

  if (uploadForm) {
    uploadForm.addEventListener('submit', async function(e) {
      e.preventDefault();

      const imgInput = document.getElementById('imageInput');
      const vidInput = document.getElementById('videoInput');
      const audInput = document.getElementById('audioInput');
      const langSelect = document.getElementById('languageSelect');

      const formData = new FormData();
      let hasFile = false;

      if (imgInput && imgInput.files.length > 0) {
        formData.append('image', imgInput.files[0]);
        hasFile = true;
      }
      if (vidInput && vidInput.files.length > 0) {
        formData.append('video', vidInput.files[0]);
        hasFile = true;
      }
      if (audInput && audInput.files.length > 0) {
        formData.append('audio', audInput.files[0]);
        hasFile = true;
      }

      if (!hasFile) {
        CV.ui.showToast('Please select at least one media file (image, video, or audio).', 'warning');
        return;
      }

      formData.append('language', langSelect ? langSelect.value : 'auto');

      try {
        if (btnAnalyze) {
          btnAnalyze.disabled = true;
          btnAnalyze.textContent = 'Uploading & Initializing...';
        }

        const res = await fetch('/api/analyze', {
          method: 'POST',
          body: formData
        });

        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || `Upload failed with status ${res.status}`);
        }

        const sid = data.session_id;
        CV.state.currentSessionId = sid;
        try {
          localStorage.setItem('cv_last_session_id', sid);
        } catch (_) {}

        CV.ui.showToast(`Session ${sid.substring(0, 8)} started. Running forensic analysis...`, 'success');
        startPollingStatus(sid);

      } catch (err) {
        console.error('Upload error:', err);
        CV.ui.showToast(err.message, 'error');
        if (btnAnalyze) {
          btnAnalyze.disabled = false;
          btnAnalyze.textContent = 'Analyse Evidence';
        }
      }
    });
  }

  // 4. Polling pipeline progress
  function startPollingStatus(sessionId) {
    if (CV.state.pollingTimer) {
      clearInterval(CV.state.pollingTimer);
    }

    CV.state.pollingTimer = setInterval(async function() {
      try {
        const res = await fetch(`/api/status/${sessionId}`);
        if (!res.ok) return;
        const data = await res.json();

        CV.ui.updateProgress(data.progress, data.status);

        // Fetch intermediate session data to show partial results
        try {
          const sessRes = await fetch(`/api/session/${sessionId}`);
          if (sessRes.ok) {
            const sessData = await sessRes.json();
            CV.ui.renderSession(sessData);
          }
        } catch (_) {}

        if (data.status === 'done' || data.status === 'error') {
          clearInterval(CV.state.pollingTimer);
          CV.state.pollingTimer = null;

          if (btnAnalyze) {
            btnAnalyze.disabled = false;
            btnAnalyze.textContent = 'Analyse Evidence';
          }

          if (data.status === 'done') {
            CV.ui.showToast('Forensic analysis pipeline completed successfully!', 'success');
          } else {
            CV.ui.showToast('Forensic analysis concluded with errors. Check session details.', 'error');
          }
        }
      } catch (err) {
        console.warn('Status polling error:', err);
      }
    }, 2000);
  }

  // 5. Load last session button
  const btnLoadLast = document.getElementById('btnLoadLastSession');
  if (btnLoadLast) {
    let lastId = null;
    try {
      lastId = localStorage.getItem('cv_last_session_id');
    } catch (_) {}

    if (!lastId) {
      btnLoadLast.style.opacity = '0.5';
    }

    btnLoadLast.addEventListener('click', async function() {
      let savedSid = null;
      try {
        savedSid = localStorage.getItem('cv_last_session_id');
      } catch (_) {}

      if (!savedSid) {
        CV.ui.showToast('No saved session found in local storage.', 'warning');
        return;
      }

      try {
        btnLoadLast.textContent = 'Loading...';
        const res = await fetch(`/api/session/${savedSid}`);
        if (!res.ok) {
          throw new Error('Previous session not found on server.');
        }

        const data = await res.json();
        CV.state.currentSessionId = savedSid;
        CV.ui.renderSession(data);
        CV.ui.updateProgress(data.progress, data.status);
        CV.ui.showToast(`Loaded previous session ${savedSid.substring(0, 8)}`, 'success');
      } catch (err) {
        CV.ui.showToast(err.message, 'error');
      } finally {
        btnLoadLast.textContent = 'Load Last Session';
      }
    });
  }

  // 6. Video timeupdate tracking
  const videoPlayer = document.getElementById('videoPlayer');
  const playbackTime = document.getElementById('videoPlaybackTime');

  if (videoPlayer) {
    videoPlayer.addEventListener('timeupdate', function() {
      const cur = videoPlayer.currentTime || 0;
      const dur = videoPlayer.duration || 0;

      const formatTime = sec => {
        const m = Math.floor(sec / 60);
        const s = Math.floor(sec % 60);
        return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
      };

      if (playbackTime) {
        playbackTime.textContent = `${formatTime(cur)} / ${formatTime(dur)}`;
      }

      CV.timeline.updatePlayhead(cur);

      if (typeof window.updateScene3D === 'function') {
        window.updateScene3D(cur);
      }
    });
  }

  // Window resize re-draws canvas boxes
  window.addEventListener('resize', function() {
    if (CV.state.activeTab === 'evidence') {
      CV.ui.drawDetectionBoxes();
    }
  });
});
