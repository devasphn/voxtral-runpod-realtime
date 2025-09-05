// UNDERSTANDING-ONLY WEBSOCKET CLIENT - OPTIMIZED FOR 0.3S GAP DETECTION
class VoxtralUnderstandingClient {
    constructor() {
        this.websocket = null;
        this.isConnected = false;
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioStream = null;
        
        // UNDERSTANDING-ONLY configuration
        this.audioConfig = {
            sampleRate: 16000,
            channels: 1,
            mimeType: 'audio/webm;codecs=opus'
        };
        
        // Connection retry configuration
        this.maxRetries = 3;
        this.retryDelay = 2000;
        this.currentRetries = 0;
        
        // Gap detection settings
        this.gapThresholdMs = 300;  // 0.3 second gap detection
        this.lastAudioTime = 0;
        this.audioChunkBuffer = [];
        this.processingResponse = false;
        
        this.initializeUI();
        this.loadSystemInfo();
    }
    
    initializeUI() {
        // Get DOM elements
        this.elements = {
            connectBtn: document.getElementById('connect-btn'),
            recordBtn: document.getElementById('record-btn'),
            disconnectBtn: document.getElementById('disconnect-btn'),
            connectionStatus: document.getElementById('connection-status'),
            modelStatus: document.getElementById('model-status'),
            audioStatus: document.getElementById('audio-status'),
            resultsContainer: document.getElementById('results-container'),
            clearResults: document.getElementById('clear-results'),
            systemInfo: document.getElementById('system-info'),
            logsContainer: document.getElementById('logs-container'),
            gapStatus: document.getElementById('gap-status') || this.createGapStatusElement()
        };
        
        // Hide service selector (UNDERSTANDING-ONLY)
        const serviceSelector = document.querySelector('.service-selector');
        if (serviceSelector) {
            serviceSelector.style.display = 'none';
        }
        
        // Hide query section (not needed for understanding-only)
        const querySection = document.getElementById('query-section');
        if (querySection) {
            querySection.style.display = 'none';
        }
        
        // Bind event listeners
        this.elements.connectBtn.addEventListener('click', () => this.connect());
        this.elements.disconnectBtn.addEventListener('click', () => this.disconnect());
        this.elements.recordBtn.addEventListener('click', () => this.toggleRecording());
        this.elements.clearResults.addEventListener('click', () => this.clearResults());
        
        // Add understanding-only indicators
        this.addUnderstandingOnlyIndicators();
    }
    
    createGapStatusElement() {
        // Create gap detection status element
        const statusPanel = document.querySelector('.status-panel');
        if (statusPanel) {
            const gapItem = document.createElement('div');
            gapItem.className = 'status-item';
            gapItem.innerHTML = `
                <span class="label">Gap Detection:</span>
                <span id="gap-status" class="status">300ms</span>
            `;
            statusPanel.appendChild(gapItem);
            return gapItem.querySelector('#gap-status');
        }
        return null;
    }
    
    addUnderstandingOnlyIndicators() {
        // Update page title and description
        const header = document.querySelector('header h1');
        if (header) {
            header.textContent = '🧠 Voxtral Mini 3B - UNDERSTANDING-ONLY';
        }
        
        const description = document.querySelector('header p');
        if (description) {
            description.textContent = 'Real-time conversational AI with 0.3-second gap detection';
        }
        
        // Add understanding-only badge
        const controlPanel = document.querySelector('.control-panel');
        if (controlPanel) {
            const badge = document.createElement('div');
            badge.className = 'understanding-badge';
            badge.innerHTML = `
                <div style="background: #059669; color: white; padding: 8px 16px; border-radius: 20px; margin-bottom: 20px; text-align: center; font-weight: 600;">
                    ✅ UNDERSTANDING-ONLY MODE | 🎯 0.3s Gap Detection | ⚡ Sub-200ms Response
                </div>
            `;
            controlPanel.insertBefore(badge, controlPanel.firstChild);
        }
    }
    
    async loadSystemInfo() {
        try {
            const response = await fetch('/model/info');
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const info = await response.json();
            
            this.elements.systemInfo.innerHTML = `
                <div><strong>Model:</strong> ${info.model_name}</div>
                <div><strong>Device:</strong> ${info.device}</div>
                <div><strong>Parameters:</strong> ${info.model_size}</div>
                <div><strong>Mode:</strong> UNDERSTANDING-ONLY</div>
                <div><strong>Gap Detection:</strong> ${this.gapThresholdMs}ms</div>
                <div><strong>Target Response:</strong> Sub-200ms</div>
                <div><strong>Languages:</strong> ${info.supported_languages.join(', ')}</div>
            `;
            
            this.updateStatus('model', 'loaded', 'loaded');
        } catch (error) {
            this.log('Failed to load system info: ' + error.message, 'error');
            this.elements.systemInfo.innerHTML = '<div>❌ Failed to load system information</div>';
            this.updateStatus('model', 'error', 'error');
        }
    }
    
    getWebSocketURL() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        return `${protocol}//${host}/ws/understand`;  // UNDERSTANDING-ONLY endpoint
    }
    
    async connect() {
        if (this.isConnected) return;
        
        this.log('Connecting to UNDERSTANDING-ONLY service...', 'info');
        
        try {
            const wsUrl = this.getWebSocketURL();
            this.websocket = new WebSocket(wsUrl);
            
            // Set up event handlers
            this.websocket.onopen = () => {
                this.isConnected = true;
                this.currentRetries = 0;
                this.updateStatus('connection', 'connected', 'connected');
                this.updateButtons();
                this.log('✅ Connected to UNDERSTANDING-ONLY service', 'success');
            };
            
            this.websocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                } catch (error) {
                    this.log('Failed to parse message: ' + error.message, 'error');
                }
            };
            
            this.websocket.onerror = (error) => {
                this.log('WebSocket error occurred', 'error');
                console.error('WebSocket error:', error);
            };
            
            this.websocket.onclose = (event) => {
                this.isConnected = false;
                this.updateStatus('connection', 'disconnected', 'disconnected');
                this.updateButtons();
                
                if (event.wasClean) {
                    this.log('Connection closed cleanly', 'info');
                } else {
                    this.log(`Connection lost (code: ${event.code})`, 'warning');
                    
                    // Auto-reconnect logic
                    if (this.currentRetries < this.maxRetries) {
                        this.currentRetries++;
                        this.log(`Attempting reconnection (${this.currentRetries}/${this.maxRetries})...`, 'info');
                        setTimeout(() => this.connect(), this.retryDelay);
                    } else {
                        this.log('Max reconnection attempts reached', 'error');
                    }
                }
                
                if (this.isRecording) {
                    this.stopRecording();
                }
            };
            
        } catch (error) {
            this.log('Connection failed: ' + error.message, 'error');
            this.updateStatus('connection', 'error', 'error');
        }
    }
    
    disconnect() {
        if (!this.isConnected) return;
        
        this.log('Disconnecting...', 'info');
        this.currentRetries = this.maxRetries; // Prevent auto-reconnect
        
        if (this.isRecording) {
            this.stopRecording();
        }
        
        if (this.websocket) {
            this.websocket.close(1000, 'User disconnected');
            this.websocket = null;
        }
    }
    
    async toggleRecording() {
        if (this.isRecording) {
            this.stopRecording();
        } else {
            await this.startRecording();
        }
    }
    
    async startRecording() {
        if (!this.isConnected) {
            this.log('Not connected to UNDERSTANDING-ONLY service', 'error');
            return;
        }
        
        try {
            this.log('Starting continuous audio recording for UNDERSTANDING-ONLY...', 'info');
            
            // Request microphone access
            this.audioStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: this.audioConfig.sampleRate,
                    channelCount: this.audioConfig.channels,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            
            // Check MIME type support
            let mimeType = this.audioConfig.mimeType;
            if (!MediaRecorder.isTypeSupported(mimeType)) {
                const fallbacks = [
                    'audio/webm;codecs=opus',
                    'audio/webm',
                    'audio/mp4',
                    'audio/ogg;codecs=opus'
                ];
                
                mimeType = fallbacks.find(type => MediaRecorder.isTypeSupported(type)) || '';
                if (mimeType) {
                    this.log(`Using fallback MIME type: ${mimeType}`, 'warning');
                } else {
                    throw new Error('No supported audio MIME type found');
                }
            }
            
            // Create media recorder for continuous streaming
            this.mediaRecorder = new MediaRecorder(this.audioStream, {
                mimeType: mimeType
            });
            
            // Handle audio data - UNDERSTANDING-ONLY processing
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data && event.data.size > 0 && this.isConnected) {
                    this.processAudioChunk(event.data);
                }
            };
            
            this.mediaRecorder.onerror = (error) => {
                this.log('MediaRecorder error: ' + error.error, 'error');
                this.stopRecording();
            };
            
            // Start continuous recording with small intervals for responsiveness
            this.mediaRecorder.start(100); // 100ms chunks for continuous processing
            
            this.isRecording = true;
            this.lastAudioTime = Date.now();
            this.updateStatus('audio', 'recording', 'recording');
            this.updateButtons();
            this.log(`Continuous recording started for UNDERSTANDING-ONLY (${mimeType})`, 'success');
            
        } catch (error) {
            this.log('Failed to start recording: ' + error.message, 'error');
            this.updateStatus('audio', 'error', 'error');
        }
    }
    
    async processAudioChunk(audioBlob) {
        try {
            if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
                return;
            }
            
            this.lastAudioTime = Date.now();
            
            // Convert to ArrayBuffer and send directly as binary
            const arrayBuffer = await audioBlob.arrayBuffer();
            if (arrayBuffer.byteLength > 0) {
                this.websocket.send(arrayBuffer);
            }
            
        } catch (error) {
            this.log('Failed to process audio chunk: ' + error.message, 'error');
        }
    }
    
    stopRecording() {
        this.log('Stopping continuous recording...', 'info');
        
        try {
            if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
                this.mediaRecorder.stop();
            }
            
            if (this.audioStream) {
                this.audioStream.getTracks().forEach(track => {
                    track.stop();
                });
                this.audioStream = null;
            }
            
            this.mediaRecorder = null;
            this.isRecording = false;
            this.processingResponse = false;
            this.audioChunkBuffer = [];
            
            this.updateStatus('audio', 'stopped', 'not recording');
            this.updateButtons();
            this.log('Recording stopped', 'info');
            
        } catch (error) {
            this.log('Error stopping recording: ' + error.message, 'error');
        }
    }
    
    handleMessage(data) {
        if (data.type === 'connection') {
            this.log(data.message || 'Connected to UNDERSTANDING-ONLY service', 'info');
            return;
        }
        
        if (data.error) {
            this.log('Service error: ' + data.error, 'error');
            this.addResult('error', `❌ ${data.error}`, new Date());
            return;
        }
        
        // Handle intermediate feedback
        if (data.type === 'audio_received' || data.audio_received) {
            this.updateAudioFeedback(data);
            return;
        }
        
        // Handle complete understanding response
        if (data.type === 'understanding' && (data.response || data.transcription)) {
            this.processingResponse = false;
            
            const responseTime = data.response_time_ms || 0;
            const gapDetected = data.gap_detected || false;
            const subOptimal = responseTime < 200;
            
            // Format the result
            let resultText = '';
            if (data.transcription) {
                resultText += `🎤 "${data.transcription}"\n\n`;
            }
            if (data.response) {
                resultText += `🧠 ${data.response}`;
            }
            
            // Add timing information
            if (responseTime > 0) {
                resultText += `\n\n⏱️ Response: ${responseTime}ms`;
                if (subOptimal) {
                    resultText += ' ⚡';
                }
            }
            
            if (gapDetected) {
                resultText += ' 🎯 Gap detected';
            }
            
            this.addResult('understanding', resultText, new Date());
            this.log(`✅ UNDERSTANDING response received (${responseTime}ms)${gapDetected ? ' - Gap detected' : ''}`, 'success');
        }
    }
    
    updateAudioFeedback(data) {
        // Update gap detection status
        if (this.elements.gapStatus) {
            const segmentDuration = data.segment_duration_ms || 0;
            const silenceDuration = data.silence_duration_ms || 0;
            const gapTrigger = data.gap_will_trigger_at_ms || this.gapThresholdMs;
            
            let statusText = `${gapTrigger}ms`;
            let statusClass = 'status';
            
            if (segmentDuration > 0) {
                statusText = `Speaking: ${segmentDuration.toFixed(0)}ms`;
                statusClass = 'status recording';
            } else if (silenceDuration > 0) {
                const remaining = gapTrigger - silenceDuration;
                if (remaining > 0) {
                    statusText = `Gap: ${remaining.toFixed(0)}ms remaining`;
                    statusClass = 'status recording';
                } else {
                    statusText = 'Processing...';
                    statusClass = 'status connected';
                }
            }
            
            this.elements.gapStatus.textContent = statusText;
            this.elements.gapStatus.className = statusClass;
        }
    }
    
    addResult(type, content, timestamp) {
        // Remove "no results" message
        const noResults = this.elements.resultsContainer.querySelector('.no-results');
        if (noResults) {
            noResults.remove();
        }
        
        // Create result element
        const resultElement = document.createElement('div');
        resultElement.className = 'result-item';
        
        // Color coding
        let typeClass = 'result-understanding';
        if (type === 'error') typeClass = 'result-error';
        
        resultElement.innerHTML = `
            <div class="result-header">
                <span class="result-type ${typeClass}">${type}</span>
                <span class="result-time">${timestamp.toLocaleTimeString()}</span>
            </div>
            <div class="result-content">${content.replace(/\n/g, '<br>')}</div>
        `;
        
        // Add to container (newest first)
        this.elements.resultsContainer.insertBefore(resultElement, this.elements.resultsContainer.firstChild);
        
        // Keep only last 15 results
        const results = this.elements.resultsContainer.querySelectorAll('.result-item');
        if (results.length > 15) {
            results[results.length - 1].remove();
        }
    }
    
    clearResults() {
        this.elements.resultsContainer.innerHTML = '<p class="no-results">No results yet. Connect and start recording to begin.</p>';
        this.log('Results cleared', 'info');
    }
    
    updateStatus(type, className, text) {
        const element = this.elements[type + 'Status'];
        if (element) {
            element.className = 'status ' + className;
            element.textContent = text;
        }
    }
    
    updateButtons() {
        this.elements.connectBtn.disabled = this.isConnected;
        this.elements.disconnectBtn.disabled = !this.isConnected;
        this.elements.recordBtn.disabled = !this.isConnected;
        
        if (this.isRecording) {
            this.elements.recordBtn.textContent = 'Stop Recording';
            this.elements.recordBtn.className = 'btn btn-danger';
        } else {
            this.elements.recordBtn.textContent = 'Start Continuous Recording';
            this.elements.recordBtn.className = 'btn btn-secondary';
        }
    }
    
    log(message, level = 'info') {
        const timestamp = new Date().toLocaleTimeString();
        const logElement = document.createElement('p');
        logElement.className = `log-entry ${level}`;
        logElement.textContent = `[${timestamp}] ${message}`;
        
        this.elements.logsContainer.insertBefore(logElement, this.elements.logsContainer.firstChild);
        
        // Keep only last 100 log entries
        const logs = this.elements.logsContainer.querySelectorAll('.log-entry');
        if (logs.length > 100) {
            logs[logs.length - 1].remove();
        }
        
        // Auto scroll to latest
        this.elements.logsContainer.scrollTop = 0;
        
        console.log(`[${level.toUpperCase()}] ${message}`);
    }
}

// Initialize UNDERSTANDING-ONLY client when page loads
document.addEventListener('DOMContentLoaded', () => {
    try {
        window.voxtralClient = new VoxtralUnderstandingClient();
    } catch (error) {
        console.error('Failed to initialize Voxtral UNDERSTANDING-ONLY client:', error);
    }
});
