// CONTINUOUS STREAMING WEBSOCKET CLIENT - websocket_client.js - FIXED FOR UNDERSTANDING MODE
class VoxtralClient {
    constructor() {
        this.websocket = null;
        this.isConnected = false;
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioStream = null;
        this.serviceType = 'transcribe';
        
        // CONTINUOUS STREAMING: Enhanced audio configuration
        this.audioConfig = {
            sampleRate: 16000,
            channels: 1,
            mimeType: 'audio/webm;codecs=opus'
        };
        
        // Connection retry configuration  
        this.maxRetries = 3;
        this.retryDelay = 2000;
        this.currentRetries = 0;
        
        this.initializeUI();
        this.loadSystemInfo();
    }
    
    initializeUI() {
        // Get DOM elements
        this.elements = {
            connectBtn: document.getElementById('connect-btn'),
            recordBtn: document.getElementById('record-btn'), 
            disconnectBtn: document.getElementById('disconnect-btn'),
            serviceType: document.getElementById('service-type'),
            textQuery: document.getElementById('text-query'),
            querySection: document.getElementById('query-section'),
            connectionStatus: document.getElementById('connection-status'),
            modelStatus: document.getElementById('model-status'),
            audioStatus: document.getElementById('audio-status'),
            resultsContainer: document.getElementById('results-container'),
            clearResults: document.getElementById('clear-results'),
            systemInfo: document.getElementById('system-info'),
            logsContainer: document.getElementById('logs-container')
        };
        
        // Bind event listeners
        this.elements.connectBtn.addEventListener('click', () => this.connect());
        this.elements.disconnectBtn.addEventListener('click', () => this.disconnect());
        this.elements.recordBtn.addEventListener('click', () => this.toggleRecording());
        this.elements.clearResults.addEventListener('click', () => this.clearResults());
        this.elements.serviceType.addEventListener('change', (e) => {
            this.serviceType = e.target.value;
            this.toggleQuerySection();
            
            // If switching to understanding mode while recording, restart recording
            if (this.isRecording && this.serviceType === 'understand') {
                this.log('Switching to CONTINUOUS STREAMING understanding mode', 'info');
            }
        });
        
        this.toggleQuerySection();
    }
    
    toggleQuerySection() {
        const isUnderstanding = this.serviceType === 'understand';
        this.elements.querySection.style.display = isUnderstanding ? 'block' : 'none';
        
        if (isUnderstanding) {
            this.elements.textQuery.value = 'Please respond naturally to what I said';
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
                <div><strong>Architecture:</strong> ${info.architecture || 'CONTINUOUS_STREAMING_WITH_GAP_DETECTION'}</div>
                <div><strong>Device:</strong> ${info.device}</div>
                <div><strong>Parameters:</strong> ${info.model_size}</div>
                <div><strong>Context Length:</strong> ${info.context_length}</div>
                <div><strong>Gap Threshold:</strong> 300ms</div>
                <div><strong>Target Latency:</strong> &lt;200ms</div>
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
        return `${protocol}//${host}/ws/${this.serviceType}`;
    }
    
    async connect() {
        if (this.isConnected) return;
        
        this.log(`Connecting to CONTINUOUS STREAMING ${this.serviceType} service...`, 'info');
        
        try {
            const wsUrl = this.getWebSocketURL();
            this.websocket = new WebSocket(wsUrl);
            
            // Set up event handlers
            this.websocket.onopen = () => {
                this.isConnected = true;
                this.currentRetries = 0;
                this.updateStatus('connection', 'connected', 'connected');
                this.updateButtons();
                this.log(`✅ Connected to CONTINUOUS STREAMING ${this.serviceType} service`, 'success');
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
                this.log('CONTINUOUS STREAMING WebSocket error occurred', 'error');
                console.error('WebSocket error:', error);
            };
            
            this.websocket.onclose = (event) => {
                this.isConnected = false;
                this.updateStatus('connection', 'disconnected', 'disconnected');
                this.updateButtons();
                
                if (event.wasClean) {
                    this.log('CONTINUOUS STREAMING connection closed cleanly', 'info');
                } else {
                    this.log(`CONTINUOUS STREAMING connection lost (code: ${event.code})`, 'warning');
                    
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
            this.log('CONTINUOUS STREAMING connection failed: ' + error.message, 'error');
            this.updateStatus('connection', 'error', 'error');
        }
    }
    
    disconnect() {
        if (!this.isConnected) return;
        
        this.log('Disconnecting from CONTINUOUS STREAMING...', 'info');
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
            this.log('Not connected to CONTINUOUS STREAMING service', 'error');
            return;
        }
        
        try {
            this.log(`Starting CONTINUOUS STREAMING ${this.serviceType} recording...`, 'info');
            
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
            
            // Create media recorder
            this.mediaRecorder = new MediaRecorder(this.audioStream, {
                mimeType: mimeType
            });
            
            // CONTINUOUS STREAMING: Handle audio data for both modes
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data && event.data.size > 0 && this.isConnected) {
                    this.sendAudioData(event.data);
                }
            };
            
            this.mediaRecorder.onerror = (error) => {
                this.log('MediaRecorder error: ' + error.error, 'error');
                this.stopRecording();
            };
            
            // CONTINUOUS STREAMING: Different intervals for different modes
            const recordingInterval = this.serviceType === 'transcribe' ? 500 : 100; // Faster for understanding
            this.mediaRecorder.start(recordingInterval);
            
            this.isRecording = true;
            this.updateStatus('audio', 'recording', 'recording');
            this.updateButtons();
            this.log(`CONTINUOUS STREAMING ${this.serviceType} recording started (${mimeType})`, 'success');
            
            if (this.serviceType === 'understand') {
                this.log('🧠 CONTINUOUS STREAMING: Speak now, I will respond after 300ms silence gap', 'info');
            }
            
        } catch (error) {
            this.log('Failed to start CONTINUOUS STREAMING recording: ' + error.message, 'error');
            this.updateStatus('audio', 'error', 'error');
        }
    }
    
    stopRecording() {
        this.log('Stopping CONTINUOUS STREAMING recording...', 'info');
        
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
            this.updateStatus('audio', 'stopped', 'not recording');
            this.updateButtons();
            this.log('CONTINUOUS STREAMING recording stopped', 'info');
            
        } catch (error) {
            this.log('Error stopping CONTINUOUS STREAMING recording: ' + error.message, 'error');
        }
    }
    
    async sendAudioData(audioBlob) {
        try {
            if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
                this.log('WebSocket not ready, skipping audio data', 'warning');
                return;
            }
            
            // CONTINUOUS STREAMING: Both modes use binary streaming now
            const arrayBuffer = await audioBlob.arrayBuffer();
            if (arrayBuffer.byteLength > 0) {
                this.websocket.send(arrayBuffer);
            }
            
        } catch (error) {
            this.log('Failed to send CONTINUOUS STREAMING audio data: ' + error.message, 'error');
        }
    }
    
    handleMessage(data) {
        if (data.type === 'connection') {
            this.log(data.message, 'info');
            return;
        }
        
        if (data.error) {
            this.log('CONTINUOUS STREAMING service error: ' + data.error, 'error');
            this.addResult('error', `❌ ${data.error}`, new Date());
            return;
        }
        
        // Handle successful results
        if (data.type === 'transcription' && data.text) {
            this.addResult('transcription', data.text, new Date());
            this.log('✅ CONTINUOUS STREAMING transcription received', 'success');
        } else if (data.type === 'understanding' && data.response) {
            // Show performance metrics if available
            let performanceInfo = '';
            if (data.performance) {
                const totalTime = data.performance.total_time_ms;
                const isUnder200ms = totalTime < 200;
                const performanceIcon = isUnder200ms ? '🚀' : '⏱️';
                performanceInfo = ` ${performanceIcon} ${totalTime.toFixed(0)}ms`;
            }
            
            this.addResult('understanding', data.response + performanceInfo, new Date());
            this.log(`✅ CONTINUOUS STREAMING understanding response received${performanceInfo}`, 'success');
            
            // Show transcription if available
            if (data.transcription) {
                this.log(`📝 Transcribed: "${data.transcription}"`, 'info');
            }
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
        let typeClass = 'result-transcription';
        if (type === 'understanding') typeClass = 'result-understanding';
        if (type === 'error') typeClass = 'result-error';
        
        resultElement.innerHTML = `
            <div class="result-header">
                <span class="result-type ${typeClass}">${type}</span>
                <span class="result-time">${timestamp.toLocaleTimeString()}</span>
            </div>
            <div class="result-content">${content}</div>
        `;
        
        // Add to container (newest first)
        this.elements.resultsContainer.insertBefore(resultElement, this.elements.resultsContainer.firstChild);
        
        // Keep only last 20 results
        const results = this.elements.resultsContainer.querySelectorAll('.result-item');
        if (results.length > 20) {
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
            this.elements.recordBtn.textContent = 'Start Recording';
            this.elements.recordBtn.className = 'btn btn-secondary';
        }
    }
    
    log(message, level = 'info') {
        const timestamp = new Date().toLocaleTimeString();
        const logElement = document.createElement('p');
        logElement.className = `log-entry ${level}`;
        logElement.textContent = `[${timestamp}] ${message}`;
        
        this.elements.logsContainer.insertBefore(logElement, this.elements.logsContainer.firstChild);
        
        // Keep only last 150 log entries
        const logs = this.elements.logsContainer.querySelectorAll('.log-entry');
        if (logs.length > 150) {
            logs[logs.length - 1].remove();
        }
        
        // Auto scroll to latest
        this.elements.logsContainer.scrollTop = 0;
        
        console.log(`[CONTINUOUS STREAMING ${level.toUpperCase()}] ${message}`);
    }
}

// Initialize client when page loads
document.addEventListener('DOMContentLoaded', () => {
    try {
        window.voxtralClient = new VoxtralClient();
        console.log('✅ CONTINUOUS STREAMING Voxtral client initialized');
    } catch (error) {
        console.error('Failed to initialize CONTINUOUS STREAMING Voxtral client:', error);
    }
});
