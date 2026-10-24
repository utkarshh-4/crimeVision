/* ==========================================================================
   CrimeVision AI - Interactive 3D Forensic Scene Reconstruction
   Three.js r128 (Global THREE build + OrbitControls)
   ========================================================================== */

(function(window) {
  'use strict';

  // Module state
  let scene = null;
  let camera = null;
  let renderer = null;
  let controls = null;
  let containerEl = null;
  let animId = null;
  const clock = new THREE.Clock();

  let currentSession = null;
  let currentTime = 0;
  let isPlaying = false;
  let playTimer = null;

  // Active scene objects
  let dynamicObjectsGroup = null;
  let trailsGroup = null;
  let anchorMarkersGroup = null;
  let sceneTypeSprite = null;

  // Materials & Pulsing weapon references
  const pulsingWeaponMaterials = [];
  const raycastTargets = [];
  let raycaster = null;
  const mouse = new THREE.Vector2(-999, -999);
  let tooltipEl = null;
  let sliderEl = null;
  let playBtnEl = null;
  let timeLabelEl = null;
  let clipBadgeEl = null;

  /**
   * Safe check for WebGL context availability.
   */
  function isWebGLAvailable() {
    try {
      const canvas = document.createElement('canvas');
      return !!(
        window.WebGLRenderingContext &&
        (canvas.getContext('webgl') || canvas.getContext('experimental-webgl'))
      );
    } catch (e) {
      return false;
    }
  }

  /**
   * Dispose all geometries, materials, and textures associated with an Object3D.
   */
  function disposeNode(node) {
    if (!node) return;
    if (node.geometry) {
      node.geometry.dispose();
    }
    if (node.material) {
      if (Array.isArray(node.material)) {
        node.material.forEach(m => {
          if (m.map) m.map.dispose();
          m.dispose();
        });
      } else {
        if (node.material.map) node.material.map.dispose();
        node.material.dispose();
      }
    }
  }

  function clearGroup(group) {
    if (!group) return;
    while (group.children.length > 0) {
      const child = group.children[0];
      group.remove(child);
      child.traverse(disposeNode);
    }
  }

  /**
   * Create an un-pixelated canvas texture sprite for floating 3D text labels.
   */
  function createTextSprite(text, bgColor = 'rgba(15, 23, 42, 0.85)', textColor = '#38bdf8', borderColor = '#0284c7') {
    const canvas = document.createElement('canvas');
    canvas.width = 300;
    canvas.height = 70;
    const ctx = canvas.getContext('2d');

    // Rounded pill background
    ctx.fillStyle = bgColor;
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 3;
    const r = 14;
    const w = canvas.width - 6;
    const h = canvas.height - 6;
    const x = 3;
    const y = 3;

    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Text label
    ctx.fillStyle = textColor;
    ctx.font = 'bold 26px "JetBrains Mono", Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, canvas.width / 2, canvas.height / 2);

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const spriteMat = new THREE.SpriteMaterial({ map: texture, depthTest: false, transparent: true });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.scale.set(1.9, 0.45, 1.0);
    return sprite;
  }

  /**
   * 2D-to-Pseudo-3D Projection Logic:
   * ---------------------------------------------------------------------------
   * Camera looks down onto ground plane (-7 to +7 along Z, -10 to +10 along X).
   * In 2D surveillance/image frames:
   *   - X: Normalized horizontal box center maps from -10 to +10.
   *        x = ((x1 + x2) / 2 / W - 0.5) * 20
   *   - Z: The bottom of the bounding box (y2) represents ground contact.
   *        Objects lower in the image (larger y2) are closer to the camera.
   *        z = (1 - y2 / H) * 14 - 7
   *   - Height: Box height scaled relative to camera perspective.
   *        h = clamp((y2 - y1) / H, 0.15, 1.0) * 2.2
   */
  function project2Dto3D(box, imgW, imgH) {
    const W = imgW || 1280;
    const H = imgH || 720;
    const x1 = box[0];
    const y1 = box[1];
    const x2 = box[2];
    const y2 = box[3];

    const cx = (x1 + x2) / 2.0;
    const x = (cx / W - 0.5) * 20.0;
    const z = (1.0 - y2 / H) * 14.0 - 7.0;

    const rawH = (y2 - y1) / H;
    const clampedH = Math.max(0.15, Math.min(1.0, rawH));
    const h = clampedH * 2.2;

    return { x, z, h, rawH };
  }

  /**
   * Check if a detection is a synthetic full-frame CLIP detection (covering ~whole image).
   */
  function isSyntheticClipBox(box, imgW, imgH, rawLabel) {
    if (rawLabel && String(rawLabel).toLowerCase().includes('clip')) return true;
    const W = imgW || 1280;
    const H = imgH || 720;
    const widthRatio = (box[2] - box[0]) / W;
    const heightRatio = (box[3] - box[1]) / H;
    return widthRatio >= 0.88 && heightRatio >= 0.88;
  }

  /**
   * Build Human mesh (Cylinder body + Sphere head - r128 compatible compound capsule).
   */
  function createHumanMesh(h) {
    const group = new THREE.Group();
    const bodyH = Math.max(0.4, h * 0.72);
    const radius = 0.32;

    // Body Cylinder
    const bodyGeo = new THREE.CylinderGeometry(radius, radius, bodyH, 16);
    const bodyMat = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      roughness: 0.35,
      metalness: 0.2,
    });
    const bodyMesh = new THREE.Mesh(bodyGeo, bodyMat);
    bodyMesh.position.y = bodyH / 2.0;
    bodyMesh.castShadow = true;
    group.add(bodyMesh);

    // Head Sphere
    const headRadius = 0.24;
    const headGeo = new THREE.SphereGeometry(headRadius, 16, 16);
    const headMat = new THREE.MeshStandardMaterial({
      color: 0x7dd3fc,
      roughness: 0.3,
      metalness: 0.1,
    });
    const headMesh = new THREE.Mesh(headGeo, headMat);
    headMesh.position.y = bodyH + headRadius;
    headMesh.castShadow = true;
    group.add(headMesh);

    // Ground Contact Shadow Disc
    const discGeo = new THREE.CylinderGeometry(radius * 1.25, radius * 1.25, 0.02, 16);
    const discMat = new THREE.MeshBasicMaterial({ color: 0x0284c7, transparent: true, opacity: 0.45 });
    const discMesh = new THREE.Mesh(discGeo, discMat);
    discMesh.position.y = 0.01;
    group.add(discMesh);

    group.userData.meshHeight = bodyH + headRadius * 2;
    return group;
  }

  /**
   * Build Weapon mesh (Elongated box with pulsing red emissive material).
   */
  function createWeaponMesh() {
    const group = new THREE.Group();
    const boxGeo = new THREE.BoxGeometry(0.24, 0.7, 0.24);
    const boxMat = new THREE.MeshStandardMaterial({
      color: 0xef4444,
      emissive: 0xef4444,
      emissiveIntensity: 0.6,
      roughness: 0.25,
      metalness: 0.4,
    });
    const boxMesh = new THREE.Mesh(boxGeo, boxMat);
    boxMesh.position.y = 0.55;
    boxMesh.rotation.z = Math.PI / 6;
    boxMesh.castShadow = true;
    group.add(boxMesh);

    // Pulsing ground indicator
    const ringGeo = new THREE.RingGeometry(0.2, 0.45, 16);
    ringGeo.rotateX(-Math.PI / 2);
    const ringMat = new THREE.MeshBasicMaterial({ color: 0xef4444, side: THREE.DoubleSide, transparent: true, opacity: 0.6 });
    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
    ringMesh.position.y = 0.02;
    group.add(ringMesh);

    pulsingWeaponMaterials.push(boxMat);
    group.userData.meshHeight = 1.0;
    return group;
  }

  /**
   * Build Object mesh (Compact slate grey box).
   */
  function createObjectMesh() {
    const group = new THREE.Group();
    const boxGeo = new THREE.BoxGeometry(0.5, 0.5, 0.5);
    const boxMat = new THREE.MeshStandardMaterial({
      color: 0x64748b,
      roughness: 0.7,
      metalness: 0.1,
    });
    const boxMesh = new THREE.Mesh(boxGeo, boxMat);
    boxMesh.position.y = 0.25;
    boxMesh.castShadow = true;
    group.add(boxMesh);

    group.userData.meshHeight = 0.6;
    return group;
  }

  /**
   * Rebuild or interpolate 3D meshes for target second t.
   */
  function renderFrameAtTime(t) {
    if (!dynamicObjectsGroup) return;

    pulsingWeaponMaterials.length = 0;
    raycastTargets.length = 0;

    let detections = [];
    let imgW = 1280;
    let imgH = 720;
    let hasClipFirearm = false;

    // 1. Determine detections from Video frames or Static Image
    if (currentSession && currentSession.video && Array.isArray(currentSession.video.frames) && currentSession.video.frames.length > 0) {
      const frames = currentSession.video.frames;
      imgW = currentSession.video.width || 1280;
      imgH = currentSession.video.height || 720;

      // Find frame nearest to second t
      let bestFrame = frames[0];
      let minDiff = 9999;
      frames.forEach(f => {
        const diff = Math.abs((f.t || 0) - t);
        if (diff < minDiff) {
          minDiff = diff;
          bestFrame = f;
        }
      });
      detections = bestFrame.detections || [];
    } else if (currentSession && currentSession.image && Array.isArray(currentSession.image.detections)) {
      detections = currentSession.image.detections;
      imgW = currentSession.image.width || 800;
      imgH = currentSession.image.height || 600;
    }

    // 2. Clear old meshes
    clearGroup(dynamicObjectsGroup);

    // 3. Project each detection into 3D
    detections.forEach(det => {
      const box = det.box || [0, 0, 0, 0];
      const label = det.label || 'Object';
      const rawLabel = det.raw_label || label;
      const conf = typeof det.confidence === 'number' ? det.confidence : 0.8;

      // Check synthetic CLIP detection
      if (isSyntheticClipBox(box, imgW, imgH, rawLabel)) {
        hasClipFirearm = true;
        return; // Skip drawing huge 3D box covering entire scene
      }

      const proj = project2Dto3D(box, imgW, imgH);
      let meshGroup = null;

      if (label === 'Human') {
        meshGroup = createHumanMesh(proj.h);
      } else if (label === 'Weapon' || det.is_weapon) {
        meshGroup = createWeaponMesh();
      } else {
        meshGroup = createObjectMesh();
      }

      meshGroup.position.set(proj.x, 0, proj.z);

      // Attach floating label sprite
      const labelText = `${label} ${Math.round(conf * 100)}%`;
      const labelColor = label === 'Weapon' ? '#ef4444' : (label === 'Human' ? '#38bdf8' : '#94a3b8');
      const borderColor = label === 'Weapon' ? '#dc2626' : (label === 'Human' ? '#0284c7' : '#475569');
      const sprite = createTextSprite(labelText, 'rgba(15, 23, 42, 0.85)', labelColor, borderColor);
      sprite.position.y = (meshGroup.userData.meshHeight || 1.2) + 0.55;
      meshGroup.add(sprite);

      // Attach metadata for raycaster hover
      meshGroup.userData = {
        label: label,
        raw_label: rawLabel,
        confidence: conf,
        box: box,
        coords: proj,
      };

      dynamicObjectsGroup.add(meshGroup);

      // Collect all child meshes for raycasting
      meshGroup.traverse(child => {
        if (child.isMesh) {
          child.userData.parentGroup = meshGroup;
          raycastTargets.push(child);
        }
      });
    });

    // Update CLIP firearm badge overlay
    if (clipBadgeEl) {
      clipBadgeEl.style.display = hasClipFirearm ? 'flex' : 'none';
    }

    // Update Anchor Event markers
    updateAnchorMarkers(t);
  }

  /**
   * Draw faint motion trails across seconds for humans in video evidence.
   */
  function buildMotionTrails() {
    if (!trailsGroup) return;
    clearGroup(trailsGroup);

    if (!currentSession || !currentSession.video || !Array.isArray(currentSession.video.frames)) {
      return;
    }

    const frames = currentSession.video.frames;
    const imgW = currentSession.video.width || 1280;
    const imgH = currentSession.video.height || 720;

    // Collect human points across seconds
    const humanPositions = [];
    frames.forEach(f => {
      (f.detections || []).forEach(d => {
        if (d.label === 'Human' && d.box) {
          const proj = project2Dto3D(d.box, imgW, imgH);
          humanPositions.push(new THREE.Vector3(proj.x, 0.04, proj.z));
        }
      });
    });

    if (humanPositions.length > 1) {
      const lineGeo = new THREE.BufferGeometry().setFromPoints(humanPositions);
      const lineMat = new THREE.LineBasicMaterial({
        color: 0x06b6d4,
        transparent: true,
        opacity: 0.35,
        linewidth: 2,
      });
      const lineMesh = new THREE.Line(lineGeo, lineMat);
      trailsGroup.add(lineMesh);
    }
  }

  /**
   * Build floating Anchor Event markers shown only when |t - event.start_raw| <= 1.5s.
   */
  function buildAnchorMarkers() {
    if (!anchorMarkersGroup) return;
    clearGroup(anchorMarkersGroup);

    if (!currentSession) return;
    const events = [];

    // Combine anchor events from fusion and video
    if (currentSession.fusion && Array.isArray(currentSession.fusion.timeline)) {
      currentSession.fusion.timeline.forEach(ev => {
        if (ev.is_anchor || (currentSession.fusion.anchor_events || []).includes(ev.id)) {
          events.push(ev);
        }
      });
    }
    if (currentSession.video && Array.isArray(currentSession.video.events)) {
      currentSession.video.events.forEach(ev => events.push(ev));
    }

    events.forEach(ev => {
      const st = typeof ev.start_raw === 'number' ? ev.start_raw : 0.0;
      const cat = ev.category || 'Anchor';
      const sprite = createTextSprite(`★ ${cat}`, 'rgba(217, 119, 6, 0.9)', '#fef3c7', '#f59e0b');
      // Semi-randomized location or mid-ground if spatial not provided
      sprite.position.set((Math.sin(st * 3) * 6), 3.2, (Math.cos(st * 2) * 4));
      sprite.userData.start_raw = st;
      sprite.visible = false;
      anchorMarkersGroup.add(sprite);
    });
  }

  function updateAnchorMarkers(t) {
    if (!anchorMarkersGroup) return;
    anchorMarkersGroup.children.forEach(sprite => {
      const st = sprite.userData.start_raw;
      if (typeof st === 'number') {
        sprite.visible = Math.abs(t - st) <= 1.5;
        // Subtle floating bounce
        if (sprite.visible) {
          sprite.position.y = 3.2 + Math.sin(clock.getElapsedTime() * 3.0) * 0.15;
        }
      }
    });
  }

  /**
   * Update floating scene classification billboard.
   */
  function updateSceneBillboard() {
    if (sceneTypeSprite) {
      scene.remove(sceneTypeSprite);
      disposeNode(sceneTypeSprite);
      sceneTypeSprite = null;
    }

    let sceneLabel = 'Forensic Reconstruction Area';
    if (currentSession) {
      if (currentSession.image && currentSession.image.scene && currentSession.image.scene.label) {
        sceneLabel = `Scene: ${currentSession.image.scene.label.toUpperCase()}`;
      } else if (currentSession.audio && currentSession.audio.facts && currentSession.audio.facts.location_inference) {
        sceneLabel = `Location: ${String(currentSession.audio.facts.location_inference.label).toUpperCase()}`;
      }
    }

    sceneTypeSprite = createTextSprite(`◈ ${sceneLabel}`, 'rgba(15, 23, 42, 0.9)', '#38bdf8', '#0284c7');
    sceneTypeSprite.scale.set(3.2, 0.7, 1);
    sceneTypeSprite.position.set(0, 6.0, -6.8);
    scene.add(sceneTypeSprite);
  }

  /**
   * Raycaster hover inspection logic.
   */
  function onPointerMove(event) {
    if (!renderer || !tooltipEl) return;
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    if (!raycaster || !camera) return;
    raycaster.setFromCamera(mouse, camera);

    const intersects = raycaster.intersectObjects(raycastTargets, false);
    if (intersects.length > 0) {
      const hit = intersects[0];
      const parent = hit.object.userData.parentGroup;
      if (parent && parent.userData) {
        const d = parent.userData;
        const box = d.box || [0, 0, 0, 0];
        tooltipEl.style.display = 'block';
        tooltipEl.style.left = `${event.clientX - rect.left + 15}px`;
        tooltipEl.style.top = `${event.clientY - rect.top + 15}px`;
        tooltipEl.innerHTML = `
          <div style="font-weight:700; color:#38bdf8; margin-bottom:2px;">${d.label} (${Math.round(d.confidence * 100)}%)</div>
          <div><b>Raw:</b> ${d.raw_label}</div>
          <div><b>Box:</b> [${box.map(v => Math.round(v)).join(', ')}]</div>
          <div><b>3D (X, Z):</b> (${d.coords.x.toFixed(2)}, ${d.coords.z.toFixed(2)})</div>
        `;
        return;
      }
    }
    tooltipEl.style.display = 'none';
  }

  /**
   * Animation & Render Loop (PAUSES when 3D tab is hidden).
   */
  function animate() {
    const view3D = document.getElementById('viewScene3D');
    // If the 3D tab is not active, cancel loop to preserve CPU / GPU resources
    if (!view3D || !view3D.classList.contains('active')) {
      animId = null;
      return;
    }

    animId = requestAnimationFrame(animate);

    const delta = clock.getDelta();
    const elapsedTime = clock.getElapsedTime();

    if (controls) {
      controls.update();
    }

    // Pulse red weapon emissive intensity
    if (pulsingWeaponMaterials.length > 0) {
      const pulse = 0.4 + 0.35 * Math.sin(elapsedTime * 6.0);
      pulsingWeaponMaterials.forEach(mat => {
        mat.emissiveIntensity = pulse;
      });
    }

    // Bob scene billboard gently
    if (sceneTypeSprite) {
      sceneTypeSprite.position.y = 6.0 + Math.sin(elapsedTime * 1.5) * 0.1;
    }

    if (renderer && scene && camera) {
      renderer.render(scene, camera);
    }
  }

  /**
   * Resize Handler ensuring aspect-ratio preservation.
   */
  function onWindowResize() {
    if (!containerEl || !renderer || !camera) return;
    const width = containerEl.clientWidth || 800;
    const height = containerEl.clientHeight || 520;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
  }

  /**
   * Build the UI HUD controls (Video playback scrubber, play/pause, CLIP warning).
   */
  function ensureHudOverlay(container) {
    if (container.querySelector('.cv-3d-hud')) return;

    const hud = document.createElement('div');
    hud.className = 'cv-3d-hud';
    hud.style.cssText = `
      position: absolute; bottom: 12px; left: 16px; right: 16px;
      display: flex; align-items: center; gap: 12px;
      background: rgba(15, 23, 42, 0.88); backdrop-filter: blur(8px);
      border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 8px;
      padding: 8px 14px; z-index: 10; font-family: 'JetBrains Mono', monospace; font-size: 11px;
    `;

    playBtnEl = document.createElement('button');
    playBtnEl.className = 'cv-btn-secondary';
    playBtnEl.style.cssText = 'padding: 4px 10px; font-weight: 700; min-width: 32px;';
    playBtnEl.textContent = '▶';
    playBtnEl.title = 'Play / Pause Video Replay';

    sliderEl = document.createElement('input');
    sliderEl.type = 'range';
    sliderEl.min = '0';
    sliderEl.max = '10';
    sliderEl.step = '1';
    sliderEl.value = '0';
    sliderEl.style.cssText = 'flex: 1; accent-color: #38bdf8; cursor: pointer;';

    timeLabelEl = document.createElement('span');
    timeLabelEl.style.cssText = 'color: #94a3b8; min-width: 80px; text-align: right;';
    timeLabelEl.textContent = 't = 0.0s';

    hud.appendChild(playBtnEl);
    hud.appendChild(sliderEl);
    hud.appendChild(timeLabelEl);
    container.appendChild(hud);

    // Floating CLIP Firearm badge
    clipBadgeEl = document.createElement('div');
    clipBadgeEl.className = 'cv-3d-clip-badge';
    clipBadgeEl.style.cssText = `
      position: absolute; top: 12px; right: 16px; z-index: 10;
      background: rgba(239, 68, 68, 0.2); border: 1px solid #ef4444; color: #ef4444;
      padding: 5px 12px; border-radius: 6px; font-weight: 700; font-size: 11px;
      display: none; align-items: center; gap: 6px; backdrop-filter: blur(4px);
    `;
    clipBadgeEl.textContent = '⚠ Firearm likely (CLIP)';
    container.appendChild(clipBadgeEl);

    // Tooltip Element
    tooltipEl = document.createElement('div');
    tooltipEl.className = 'cv-3d-tooltip';
    tooltipEl.style.cssText = `
      position: absolute; z-index: 20; display: none; pointer-events: none;
      background: rgba(15, 23, 42, 0.95); border: 1px solid #38bdf8; color: #f8fafc;
      padding: 6px 10px; border-radius: 6px; font-size: 11px; font-family: 'JetBrains Mono', monospace;
      box-shadow: 0 4px 14px rgba(0,0,0,0.5);
    `;
    container.appendChild(tooltipEl);

    // Scrubber interaction
    sliderEl.addEventListener('input', () => {
      const val = parseFloat(sliderEl.value);
      currentTime = val;
      if (timeLabelEl) timeLabelEl.textContent = `t = ${val.toFixed(1)}s`;
      renderFrameAtTime(val);

      // Sync video player if present
      const vid = document.getElementById('videoPlayer');
      if (vid && Math.abs(vid.currentTime - val) > 0.5) {
        vid.currentTime = val;
      }
    });

    // Play / Pause toggle
    playBtnEl.addEventListener('click', () => {
      isPlaying = !isPlaying;
      playBtnEl.textContent = isPlaying ? '❚❚' : '▶';

      const vid = document.getElementById('videoPlayer');
      if (vid) {
        if (isPlaying) vid.play();
        else vid.pause();
      }

      if (isPlaying) {
        if (playTimer) clearInterval(playTimer);
        const maxT = parseFloat(sliderEl.max || 10);
        playTimer = setInterval(() => {
          if (!isPlaying) {
            clearInterval(playTimer);
            return;
          }
          currentTime += 1;
          if (currentTime > maxT) currentTime = 0;
          sliderEl.value = String(currentTime);
          if (timeLabelEl) timeLabelEl.textContent = `t = ${currentTime.toFixed(1)}s`;
          renderFrameAtTime(currentTime);
        }, 1000);
      } else {
        if (playTimer) clearInterval(playTimer);
      }
    });
  }

  // =========================================================================
  // PUBLIC EXPORTS
  // =========================================================================

  /**
   * Initialize or update the 3D scene with the given session data.
   * Can be called as window.initScene3D(session) or window.initScene3D(containerId, session).
   */
  window.initScene3D = function(arg1, arg2) {
    let session = null;
    let targetId = 'scene3d';

    if (typeof arg1 === 'string') {
      targetId = arg1;
      session = arg2 || null;
    } else {
      session = arg1 || null;
    }

    containerEl = document.getElementById(targetId);
    if (!containerEl) return;

    if (!isWebGLAvailable()) {
      containerEl.innerHTML = `
        <div style="height:480px; display:flex; flex-direction:column; align-items:center; justify-content:center; color:#94a3b8; font-size:13px; text-align:center; padding:20px;">
          <div style="font-size:32px; color:#ef4444; margin-bottom:10px;">⚠</div>
          <div style="font-weight:700; color:#f8fafc; margin-bottom:6px;">WebGL Unavailable</div>
          <div>Hardware acceleration or WebGL is disabled in your browser. 3D spatial reconstruction cannot be rendered.</div>
        </div>
      `;
      return;
    }

    currentSession = session;
    containerEl.style.position = 'relative';
    containerEl.style.height = '540px';
    containerEl.style.overflow = 'hidden';

    // 1. First-time Three.js scene setup
    if (!renderer) {
      containerEl.innerHTML = '';

      const width = containerEl.clientWidth || 800;
      const height = containerEl.clientHeight || 540;

      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x080b11);
      scene.fog = new THREE.FogExp2(0x080b11, 0.02);

      camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
      camera.position.set(0, 9, 14);
      camera.lookAt(0, 0, 0);

      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
      renderer.setPixelRatio(window.devicePixelRatio || 1);
      renderer.setSize(width, height);
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      containerEl.appendChild(renderer.domElement);

      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.minDistance = 3;
      controls.maxDistance = 40;
      controls.maxPolarAngle = Math.PI / 2 - 0.05; // Prevent camera sinking under ground
      controls.target.set(0, 0, 0);

      // Lighting
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
      scene.add(ambientLight);

      const dirLight = new THREE.DirectionalLight(0xffffff, 0.85);
      dirLight.position.set(10, 22, 12);
      dirLight.castShadow = true;
      dirLight.shadow.mapSize.width = 1024;
      dirLight.shadow.mapSize.height = 1024;
      scene.add(dirLight);

      const fillLight = new THREE.DirectionalLight(0x38bdf8, 0.35);
      fillLight.position.set(-12, 15, -10);
      scene.add(fillLight);

      // Ground plane (20 x 14 units)
      const groundGeo = new THREE.PlaneGeometry(20, 14);
      const groundMat = new THREE.MeshStandardMaterial({
        color: 0x0f172a,
        roughness: 0.85,
        metalness: 0.15,
      });
      const groundMesh = new THREE.Mesh(groundGeo, groundMat);
      groundMesh.rotation.x = -Math.PI / 2;
      groundMesh.receiveShadow = true;
      scene.add(groundMesh);

      // GridHelper (20 x 20)
      const gridHelper = new THREE.GridHelper(20, 20, 0x06b6d4, 0x1e293b);
      gridHelper.position.y = 0.01;
      scene.add(gridHelper);

      // Groups for dynamic elements
      dynamicObjectsGroup = new THREE.Group();
      trailsGroup = new THREE.Group();
      anchorMarkersGroup = new THREE.Group();
      scene.add(dynamicObjectsGroup);
      scene.add(trailsGroup);
      scene.add(anchorMarkersGroup);

      raycaster = new THREE.Raycaster();
      renderer.domElement.addEventListener('pointermove', onPointerMove, false);
      window.addEventListener('resize', onWindowResize, false);

      ensureHudOverlay(containerEl);
    }

    // 2. Configure video slider bounds if video duration is known
    let duration = 0;
    if (session && session.video && session.video.duration) {
      duration = Math.ceil(session.video.duration);
    }
    if (sliderEl) {
      sliderEl.max = String(Math.max(1, duration));
      sliderEl.value = '0';
      if (timeLabelEl) timeLabelEl.textContent = 't = 0.0s';
    }

    // 3. Update scene elements
    currentTime = 0;
    updateSceneBillboard();
    buildMotionTrails();
    buildAnchorMarkers();
    renderFrameAtTime(0);

    // 4. Ensure animation loop is running
    if (!animId) {
      animId = requestAnimationFrame(animate);
    }
  };

  /**
   * Update 3D scene state according to current video timestamp t.
   */
  window.updateScene3D = function(t) {
    if (typeof t !== 'number' || isNaN(t)) return;
    currentTime = t;

    if (sliderEl && Math.abs(parseFloat(sliderEl.value) - t) > 0.5) {
      sliderEl.value = String(Math.round(t));
    }
    if (timeLabelEl) {
      timeLabelEl.textContent = `t = ${t.toFixed(1)}s`;
    }

    renderFrameAtTime(t);

    // Resume loop if paused
    if (!animId) {
      animId = requestAnimationFrame(animate);
    }
  };

})(window);
