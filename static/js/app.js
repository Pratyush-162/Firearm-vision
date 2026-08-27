// Firearm Vision Universal Weapon & CCTV Surveillance Client Logic

let audioContext = null;
let lastAlertFrame = -1;

document.addEventListener('DOMContentLoaded', () => {
    // Setup connect form handler
    const connectForm = document.getElementById('connectForm');
    connectForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const sourceVal = document.getElementById('sourceInput').value.trim();
        if (sourceVal) {
            connectSource(sourceVal);
        }
    });

    // Start periodic polling
    fetchStatus();
    fetchAlerts();
    setInterval(fetchStatus, 1000);
    setInterval(fetchAlerts, 2000);
});

function setPreset(source) {
    document.getElementById('sourceInput').value = source;
    connectSource(source);
}

function updateSliderVal(elementId, val) {
    document.getElementById(elementId).innerText = val;
}

async function connectSource(source) {
    const connectBtn = document.getElementById('connectBtn');
    connectBtn.disabled = true;
    connectBtn.innerHTML = '<span>CONNECTING...</span>';

    try {
        const response = await fetch('/api/connect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source })
        });
        const data = await response.json();
        console.log('[API] Connected to source:', data);
        
        // Refresh stream image src to reset stream connection if needed
        const img = document.getElementById('videoFeed');
        img.src = '/api/stream?t=' + new Date().getTime();
    } catch (err) {
        console.error('[API] Connect error:', err);
    } finally {
        connectBtn.disabled = false;
        connectBtn.innerHTML = '<span>CONNECT STREAM</span>';
    }
}

let currentDetectMode = 'weapons_only';

function setDetectMode(mode) {
    currentDetectMode = mode;
    updateDetectModeButtons(mode);
    applySettings();
}

function updateDetectModeButtons(mode) {
    const btnWeapons = document.getElementById('btnModeWeapons');
    const btnPersons = document.getElementById('btnModePersons');
    const btnAll = document.getElementById('btnModeAll');

    if (btnWeapons) btnWeapons.classList.toggle('active', mode === 'weapons_only');
    if (btnPersons) btnPersons.classList.toggle('active', mode === 'persons_only' || mode === 'faces_only');
    if (btnAll) btnAll.classList.toggle('active', mode === 'all');
}

async function applySettings() {
    const confidence = document.getElementById('confidenceSlider') ? parseFloat(document.getElementById('confidenceSlider').value) : 0.35;
    const frames = document.getElementById('framesSlider') ? parseInt(document.getElementById('framesSlider').value) : 5;
    const required = document.getElementById('requiredSlider') ? parseInt(document.getElementById('requiredSlider').value) : 3;
    const blurFaces = document.getElementById('blurFacesToggle') ? document.getElementById('blurFacesToggle').checked : false;
    const cctvMode = document.getElementById('cctvModeToggle') ? document.getElementById('cctvModeToggle').checked : true;
    const claheEnhance = document.getElementById('claheToggle') ? document.getElementById('claheToggle').checked : false;
    const sharpnessBoost = document.getElementById('sharpnessToggle') ? document.getElementById('sharpnessToggle').checked : true;
    const tileInference = document.getElementById('tileInferenceToggle') ? document.getElementById('tileInferenceToggle').checked : true;
    const imgsz = document.getElementById('imgszSelect') ? parseInt(document.getElementById('imgszSelect').value) : 1280;
    const weaponFilter = document.getElementById('weaponScopeSelect') ? document.getElementById('weaponScopeSelect').value : 'all_weapons';

    try {
        const response = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                confidence: confidence,
                frames: frames,
                required: required,
                faces: false,
                blur_faces: blurFaces,
                cctv_mode: cctvMode,
                clahe_enhance: claheEnhance,
                sharpness_boost: sharpnessBoost,
                tile_inference: tileInference,
                imgsz: imgsz,
                detect_mode: currentDetectMode,
                weapon_filter: weaponFilter
            })
        });
        const data = await response.json();
        console.log('[API] Updated settings:', data);
    } catch (err) {
        console.error('[API] Settings error:', err);
    }
}

async function fetchStatus() {
    try {
        const response = await fetch('/api/status');
        if (!response.ok) return;
        const status = await response.json();

        // Update Connection Badge
        const statusBadge = document.getElementById('connectionStatusBadge');
        const statusText = document.getElementById('statusText');
        if (status.connected) {
            statusBadge.classList.remove('error');
            statusText.innerText = status.status_message.toUpperCase();
        } else {
            statusBadge.classList.add('error');
            statusText.innerText = status.status_message.toUpperCase();
        }

        // Update Statistics & Overlays
        document.getElementById('fpsDisplay').innerText = status.fps.toFixed(1);
        document.getElementById('sourceOverlay').innerText = `SRC: ${status.source}`;
        
        let cctvTag = status.cctv_mode ? `CCTV-HQ (${status.imgsz}p)` : `STD (${status.imgsz}p)`;
        if (status.tile_inference) cctvTag += ' + TILE-4X';
        if (status.sharpness_boost) cctvTag += ' + SHARP';
        if (status.clahe_enhance) cctvTag += ' + CLAHE';
        document.getElementById('cctvOverlay').innerText = `MODE: ${cctvTag}`;
        document.getElementById('frameOverlay').innerText = `FRAME: ${status.frame_count}`;

        // Sync inputs if user isn't actively typing
        if (document.activeElement.id !== 'sourceInput' && !document.getElementById('sourceInput').value) {
            document.getElementById('sourceInput').value = status.source;
        }

        if (status.detect_mode) {
            currentDetectMode = status.detect_mode;
            updateDetectModeButtons(status.detect_mode);
        }

        // Sync toggles and selects if not focused
        const weaponScope = document.getElementById('weaponScopeSelect');
        if (weaponScope && document.activeElement.id !== 'weaponScopeSelect') {
            weaponScope.value = status.weapon_filter || 'all_weapons';
        }
        const imgszSel = document.getElementById('imgszSelect');
        if (imgszSel && document.activeElement.id !== 'imgszSelect') {
            imgszSel.value = status.imgsz || 1280;
        }
        const cctvTgl = document.getElementById('cctvModeToggle');
        if (cctvTgl) cctvTgl.checked = status.cctv_mode;
        
        const claheTgl = document.getElementById('claheToggle');
        if (claheTgl) claheTgl.checked = status.clahe_enhance;
        
        const sharpTgl = document.getElementById('sharpnessToggle');
        if (sharpTgl) sharpTgl.checked = status.sharpness_boost;
        
        const tileTgl = document.getElementById('tileInferenceToggle');
        if (tileTgl) tileTgl.checked = status.tile_inference;

        // Update Alert Banner
        const alertBanner = document.getElementById('alertBanner');
        if (status.is_confirmed_alert) {
            alertBanner.classList.remove('hidden');
            if (status.active_detections && status.active_detections.length > 0) {
                const confirmedDet = status.active_detections.find(d => d.confirmed && d.is_weapon) || status.active_detections.find(d => d.is_weapon);
                if (confirmedDet) {
                    document.getElementById('alertBannerDetail').innerText = 
                        `WEAPON: ${confirmedDet.label.toUpperCase()} (${(confirmedDet.confidence * 100).toFixed(0)}%) — HUMAN REVIEW REQUIRED`;
                }
            }

            // Audio Alert
            const audioTgl = document.getElementById('audioAlertToggle');
            if (audioTgl && audioTgl.checked && status.frame_count !== lastAlertFrame) {
                playAudioBeep();
                lastAlertFrame = status.frame_count;
            }
        } else {
            alertBanner.classList.add('hidden');
        }

        // Update Identities List (Side Panel)
        const identitiesList = document.getElementById('identitiesList');
        const unknownIdentitiesList = document.getElementById('unknownIdentitiesList');
        
        if (identitiesList && unknownIdentitiesList && status.active_detections) {
            const knownIdentities = new Set();
            const unknownIdentities = new Set();
            let unknownCount = 0;
            
            status.active_detections.forEach(det => {
                if (det.label && det.label.startsWith("Person: ")) {
                    knownIdentities.add(det.label.replace("Person: ", "").trim());
                } else if (det.label === "Unknown" || (det.label && det.label.startsWith("Associate "))) {
                    const nameToUse = det.label === "Unknown" ? `Unknown (Track ${det.track_id})` : det.label;
                    unknownIdentities.add(nameToUse);
                }
            });

            // Handle Known Identities
            if (knownIdentities.size === 0) {
                identitiesList.innerHTML = '<span style="color: #888; font-size: 0.9em; text-align: center; margin-top: 15px; grid-column: 1 / -1;">No known persons detected.</span>';
            } else {
                identitiesList.innerHTML = '';
                Array.from(knownIdentities).sort().forEach(ident => {
                    const safeId = ident.toLowerCase().replace(/\s+/g, '_');
                    const color = "#00e676";
                    
                    const item = document.createElement('div');
                    item.style = `display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 6px; background: rgba(0, 230, 118, 0.05); border: 1px solid rgba(0, 230, 118, 0.2); padding: 6px; border-radius: 8px;`;
                    
                    const img = document.createElement('img');
                    img.src = `/faces/${safeId}/${safeId}_1.jpg`;
                    img.style = `width: 100%; aspect-ratio: 1; border-radius: 6px; object-fit: cover; background: #222; border: 2px solid ${color};`;
                    // Fallback avatar
                    img.onerror = () => { img.src = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 24 24" fill="none" stroke="%23888" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>`; };
                    
                    const nameSpan = document.createElement('span');
                    nameSpan.style = 'color: #fff; font-weight: 600; font-size: 0.8em; text-transform: capitalize; text-align: center; word-break: break-word; line-height: 1.2;';
                    nameSpan.innerText = ident;
                    
                    item.appendChild(img);
                    item.appendChild(nameSpan);
                    identitiesList.appendChild(item);
                });
            }

            // Handle Unknown Identities
            if (unknownIdentities.size === 0) {
                unknownIdentitiesList.innerHTML = '<span style="color: #888; font-size: 0.9em; text-align: center; margin-top: 15px; grid-column: 1 / -1;">No unknown persons detected.</span>';
            } else {
                unknownIdentitiesList.innerHTML = '';
                Array.from(unknownIdentities).sort().forEach(ident => {
                    const safeId = ident.toLowerCase().replace(/[()]/g, '').replace(/\s+/g, '_');
                    const color = "#ffc800";
                    
                    const item = document.createElement('div');
                    item.style = `display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 6px; background: rgba(255, 200, 0, 0.05); border: 1px solid rgba(255, 200, 0, 0.2); padding: 6px; border-radius: 8px;`;
                    
                    const img = document.createElement('img');
                    // Add timestamp to bust cache so live snapshot updates
                    img.src = `/faces/${safeId}/${safeId}_1.jpg?t=${Date.now()}`;
                    img.style = `width: 100%; aspect-ratio: 1; border-radius: 6px; object-fit: cover; background: #222; border: 2px solid ${color};`;
                    // Fallback avatar
                    img.onerror = () => { img.src = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 24 24" fill="none" stroke="%23888" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>`; };
                    
                    const nameSpan = document.createElement('span');
                    nameSpan.style = 'color: #fff; font-weight: 600; font-size: 0.8em; text-transform: capitalize; text-align: center; word-break: break-word; line-height: 1.2;';
                    nameSpan.innerText = ident;
                    
                    item.appendChild(img);
                    item.appendChild(nameSpan);
                    unknownIdentitiesList.appendChild(item);
                });
            }
        }

    } catch (err) {
        console.error('[API] Fetch status error:', err);
    }
}

async function fetchAlerts() {
    try {
        const response = await fetch('/api/alerts');
        if (!response.ok) return;
        const data = await response.json();
        const alerts = data.alerts || [];

        document.getElementById('alertCountBadge').innerText = `${alerts.length} ALERTS`;

        const tbody = document.getElementById('alertsTableBody');
        if (alerts.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" class="empty-state">No weapon alerts logged yet</td></tr>';
            return;
        }

        tbody.innerHTML = alerts.map(a => `
            <tr>
                <td><strong>${a.timestamp || '--:--:--'}</strong></td>
                <td style="color: var(--primary-cyan); font-weight: 600;">${a.label}</td>
                <td>${(a.confidence * 100).toFixed(1)}%</td>
                <td><span class="badge-review">HUMAN REVIEW REQUIRED</span></td>
            </tr>
        `).join('');

    } catch (err) {
        console.error('[API] Fetch alerts error:', err);
    }
}

function handleStreamError() {
    console.warn('[Stream] Stream image load error, attempting reconnect...');
    setTimeout(() => {
        const img = document.getElementById('videoFeed');
        img.src = '/api/stream?retry=' + new Date().getTime();
    }, 2000);
}

function playAudioBeep() {
    try {
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (audioContext.state === 'suspended') {
            audioContext.resume();
        }
        const osc = audioContext.createOscillator();
        const gain = audioContext.createGain();
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(880, audioContext.currentTime); // A5 note
        gain.gain.setValueAtTime(0.15, audioContext.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
        osc.connect(gain);
        gain.connect(audioContext.destination);
        osc.start();
        osc.stop(audioContext.currentTime + 0.3);
    } catch (e) {
        console.warn('Audio play failed:', e);
    }
}

async function enrollCurrentFace() {
    const input = document.getElementById('enrollNameInput');
    const name = input.value.trim();
    if (!name) {
        alert("Please enter a person's full name to enroll.");
        return;
    }

    const btn = event ? event.currentTarget : null;
    const hud = document.getElementById('enrollHud');
    const hudDesc = document.getElementById('hudDesc');
    const hudFill = document.getElementById('hudProgressFill');
    const hudIcon = document.getElementById('hudIcon');

    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<span>CAPTURING 30 ANGLES...</span>';
    }

    // Step-by-step Live Dynamic HUD Sequence
    if (hud) hud.style.display = 'block';

    let steps = [
        { time: 0, icon: '👀', text: '1/4: Look STRAIGHT at camera...', pct: '25%' },
        { time: 1000, icon: '👈', text: '2/4: Turn head slowly LEFT...', pct: '50%' },
        { time: 2000, icon: '👉', text: '3/4: Turn head slowly RIGHT...', pct: '75%' },
        { time: 3000, icon: '💻', text: '4/4: Look DOWN at your laptop...', pct: '100%' }
    ];

    let timers = steps.map(s => setTimeout(() => {
        if (hudDesc) hudDesc.textContent = `Step ${s.text}`;
        if (hudIcon) hudIcon.textContent = s.icon;
        if (hudFill) hudFill.style.width = s.pct;
    }, s.time));

    try {
        const response = await fetch('/api/enroll', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });
        const data = await response.json();
        if (response.ok) {
            alert(`✅ ${data.message}`);
            input.value = '';
        } else {
            alert(`⚠️ Error: ${data.detail || 'Could not enroll face'}`);
        }
    } catch (err) {
        console.error('[API] Enroll error:', err);
        alert('Failed to send enrollment request.');
    } finally {
        timers.forEach(t => clearTimeout(t));
        if (hud) hud.style.display = 'none';
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<span>ENROLL FACE</span>';
        }
    }
}
