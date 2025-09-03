// Voxtral Mini 3B WebSocket Client
class VoxtralClient {
    constructor() {
        this.websocket = null;
        this.isConnected = false;
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioStream = null;
        this.serviceType = 'transcribe';
        
        // Audio configuration
        this.audioConfig = {
            sampleRate: 16000,
            channels: 1,
            mimeType: 'audio/webm;codecs=opus'
        };
        
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
        });
        
        this.toggleQuerySection();
    }
    
    toggleQuerySection() {
        const isUnderstanding = this.serviceType === 'understand';
        this.elements.querySection.style.display = isUnderstanding ? 'block' : 'none';
    }
    
    async loadSystemInfo() {
        try {
            const response = await fetch('/model/info');
            const info = await response.json();
            
            this.elements.systemInfo.innerHTML = `
                <div><strong>Model:</strong> ${info.model_name}</div>
                <div><strong>Device:</strong> ${info.device}</div>
                <div><strong>Parameters:</strong> ${info.model_size}</div>
                <div><strong>Context Length:</strong> ${info.context_length}</div>
                <div><strong>Languages:</strong> ${info.supported_languages.join(', ')}</div>
                <div><strong>Capabilities:</strong> ${info.capabilities.join(', ')}</div>
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
        
        this.log('Connecting to Voxtral service...', 'info');
        
        try {
            const wsUrl = this.getWebSocketURL();
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = () => {
                this.isConnected = true;
                this.updateStatus('connection', 'connected', 'connected');
                this.updateButtons();
                this.log(`Connected to ${this.serviceType} service`, 'success');
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
                this.log('WebSocket error: ' + error.message, 'error');
            };
            
            this.websocket.onclose = () => {
                this.isConnected = false;
                this.updateStatus('connection', 'disconnected', 'disconnected');
                this.updateButtons();
                this.log('Disconnected from service', 'warning');
                
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
        
        if (this.isRecording) {
            this.stopRecording();
        }
        
        if (this.websocket) {
            this.websocket.close();
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
            this.log('Not connected to service', 'error');
            return;
        }
        
        try {
            this.log('Starting audio recording...', 'info');
            
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
            
            // Create media recorder
            this.mediaRecorder = new MediaRecorder(this.audioStream, {
                mimeType: this.audioConfig.mimeType
            });
            
            // Handle audio data
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0 && this.isConnected) {
                    this.sendAudioData(event.data);
                }
            };
            
            // Start recording with 500ms intervals
            this.mediaRecorder.start(500);
            
            this.isRecording = true;
            this.updateStatus('audio', 'recording', 'recording');
            this.updateButtons();
            this.log('Recording started', 'success');
            
        } catch (error) {
            this.log('Failed to start recording: ' + error.message, 'error');
            this.updateStatus('audio', 'error', 'error');
        }
    }
    
    stopRecording() {
        this.log('Stopping audio recording...', 'info');
        
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
        }
        
        if (this.audioStream) {
            this.audioStream.getTracks().forEach(track => track.stop());
            this.audioStream = null;
        }
        
        this.isRecording = false;
        this.updateStatus('audio', 'stopped', 'not recording');
        this.updateButtons();
        this.log('Recording stopped', 'info');
    }
    
    async sendAudioData(audioBlob) {
        try {
            if (this.serviceType === 'transcribe') {
                // Send raw audio data for transcription
                const arrayBuffer = await audioBlob.arrayBuffer();
                this.websocket.send(arrayBuffer);
            } else {
                // For understanding, send JSON with audio and query
                const arrayBuffer = await audioBlob.arrayBuffer();
                const audioData = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
                
                const message = {
                    audio: audioData,
                    text: this.elements.textQuery.value || 'What can you hear in this audio?'
                };
                
                this.websocket.send(JSON.stringify(message));
            }
        } catch (error) {
            this.log('Failed to send audio data: ' + error.message, 'error');
        }
    }
    
    handleMessage(data) {
        if (data.type === 'connection') {
            this.log(data.message, 'info');
            return;
        }
        
        if (data.error) {
            this.log('Service error: ' + data.error, 'error');
            this.addResult('error', data.error, new Date());
            return;
        }
        
        // Handle results
        if (data.type === 'transcription' && data.text) {
            this.addResult('transcription', data.text, new Date());
            this.log('Transcription received', 'success');
        } else if (data.type === 'understanding' && data.response) {
            this.addResult('understanding', data.response, new Date());
            this.log('Understanding response received', 'success');
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
        resultElement.innerHTML = `
            <div class="result-header">
                <span class="result-type">${type}</span>
                <span class="result-time">${timestamp.toLocaleTimeString()}</span>
            </div>
            <div class="result-content">${content}</div>
        `;
        
        // Add to container (newest first)
        this.elements.resultsContainer.insertBefore(resultElement, this.elements.resultsContainer.firstChild);
        
        // Keep only last 10 results
        const results = this.elements.resultsContainer.querySelectorAll('.result-item');
        if (results.length > 10) {
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
        
        // Keep only last 50 log entries
        const logs = this.elements.logsContainer.querySelectorAll('.log-entry');
        if (logs.length > 50) {
            logs[logs.length - 1].remove();
        }
        
        // Auto scroll to latest
        this.elements.logsContainer.scrollTop = 0;
        
        console.log(`[${level.toUpperCase()}] ${message}`);
    }
}

// Initialize client when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.voxtralClient = new VoxtralClient();
});
