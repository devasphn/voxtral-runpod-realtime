// ULTIMATE WEBSOCKET CLIENT - FIXED FOR REAL-TIME STREAMING
class UltimateVoxtralClient {
    constructor() {
        this.websocket = null;
        this.isConnected = false;
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioStream = null;
        this.mode = 'realtime';
        
        // Real-time configuration
        this.audioConfig = {
            sampleRate: 16000,
            channels: 1,
            mimeType: 'audio/webm;codecs=opus',
            continuous: true
        };
        
        // Performance tracking
        this.responseCount = 0;
        this.totalLatency = 0;
        this.vadTriggers = 0;
        
        this.initializeUI();
        this.loadSystemInfo();
    }
    
    initializeUI() {
        // Get DOM elements
        this.elements = {
            connectBtn: document.getElementById('connect-btn'),
            recordBtn: document.getElementById('record-btn'),
            disconnectBtn: document.getElementById('disconnect-btn'),
            modeSelect: document.getElementById('mode-select'),
            connectionStatus: document.getElementById('connection-status'),
            audioStatus: document.getElementById('audio-status'),
            performanceStats: document.getElementById('performance-stats'),
            resultsContainer: document.getElementById('results-container'),
            clearResults: document.getElementById('clear-results'),
            logsContainer: document.getElementById('logs-container')
        };
        
        // Bind event listeners
        this.elements.connectBtn.addEventListener('click', () => this.connect());
        this.elements.disconnectBtn.addEventListener('click', () => this.disconnect());
        this.elements.recordBtn.addEventListener('click', () => this.toggleRecording());
        this.elements.clearResults.addEventListener('click', () => this.clearResults());
        
        // Create performance stats display
        if (!this.elements.performanceStats) {
            const statsDiv = document.createElement('div');
            statsDiv.id = 'performance-stats';
            statsDiv.className = 'performance-stats';
            document.querySelector('.status-panel').appendChild(statsDiv);
            this.elements.performanceStats = statsDiv;
        }
    }
    
    async loadSystemInfo() {
        try {
            const response = await fetch('/health');
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const info = await response.json();
            this.updatePerformanceDisplay(info);
            
        } catch (error) {
            this.log('Failed to load system info: ' + error.message, 'error');
        }
    }
    
    getWebSocketURL() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        return `${protocol}//${host}/ws/realtime`;
    }
    
    async connect() {
        if (this.isConnected) return;
        
        this.log('Connecting to ULTIMATE real-time service...', 'info');
        
        try {
            const wsUrl = this.getWebSocketURL();
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = () => {
                this.isConnected = true;
                this.updateStatus('connection', 'connected');
                this.updateButtons();
                this.log('✅ Connected to ULTIMATE real-time service', 'success');
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
                this.updateStatus('connection', 'disconnected');
                this.updateButtons();
                
                if (event.wasClean) {
                    this.log('Connection closed cleanly', 'info');
                } else {
                    this.log(`Connection lost (code: ${event.code})`, 'warning');
                }
                
                if (this.isRecording) {
                    this.stopRecording();
                }
            };
            
        } catch (error) {
            this.log('Connection failed: ' + error.message, 'error');
            this.updateStatus('connection', 'error');
        }
    }
    
    disconnect() {
        if (!this.isConnected) return;
        
        this.log('Disconnecting...', 'info');
        
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
            this.log('Not connected to service', 'error');
            return;
        }
        
        try {
            this.log('Starting ULTIMATE continuous recording...', 'info');
            
            // Request high-quality microphone access
            this.audioStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: this.audioConfig.sampleRate,
                    channelCount: this.audioConfig.channels,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    latency: 0.01 // Request low latency
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
            
            // Create MediaRecorder for continuous streaming
            this.mediaRecorder = new MediaRecorder(this.audioStream, {
                mimeType: mimeType,
                audioBitsPerSecond: 32000 // Higher quality for better recognition
            });
            
            // Handle continuous audio data
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data && event.data.size > 0 && this.isConnected) {
                    this.sendAudioData(event.data);
                }
            };
            
            this.mediaRecorder.onerror = (error) => {
                this.log('MediaRecorder error: ' + error.error, 'error');
                this.stopRecording();
            };
            
            // Start continuous recording with small intervals for real-time processing
            this.mediaRecorder.start(100); // 100ms intervals for smooth streaming
            
            this.isRecording = true;
            this.updateStatus('audio', 'recording');
            this.updateButtons();
            this.log(`✅ ULTIMATE continuous recording started (${mimeType})`, 'success');
            
        } catch (error) {
            this.log('Failed to start recording: ' + error.message, 'error');
            this.updateStatus('audio', 'error');
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
            this.updateStatus('audio', 'stopped');
            this.updateButtons();
            this.log('✅ Recording stopped', 'info');
            
        } catch (error) {
            this.log('Error stopping recording: ' + error.message, 'error');
        }
    }
    
    async sendAudioData(audioBlob) {
        try {
            if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
                return;
            }
            
            // Send binary audio data directly for VAD processing
            const arrayBuffer = await audioBlob.arrayBuffer();
            if (arrayBuffer.byteLength > 0) {
                this.websocket.send(arrayBuffer);
            }
            
        } catch (error) {
            this.log('Failed to send audio data: ' + error.message, 'error');
        }
    }
    
    handleMessage(data) {
        const startTime = performance.now();
        
        if (data.type === 'connection') {
            this.log(data.status + ': ' + (data.continuous_recording ? 'Continuous mode enabled' : ''), 'info');
            return;
        }
        
        if (data.error) {
            this.log('Service error: ' + data.error, 'error');
            this.addResult('error', `❌ ${data.error}`, new Date());
            return;
        }
        
        // Handle real-time responses
        if (data.type === 'transcription' && data.text) {
            const latency = data.processing_time_ms || 0;
            this.responseCount++;
            this.totalLatency += latency;
            
            if (data.vad_triggered) {
                this.vadTriggers++;
            }
            
            // Add result with performance info
            const resultText = `${data.text} (${latency.toFixed(1)}ms${data.vad_triggered ? ', VAD' : ''})`;
            this.addResult('transcription', resultText, new Date());
            
            this.log(`✅ Real-time transcription: ${latency.toFixed(1)}ms - "${data.text}"`, 
                    latency < 200 ? 'success' : 'warning');
            
            this.updatePerformanceStats();
        } 
        else if (data.type === 'understanding' && data.response) {
            const latency = data.processing_time_ms || 0;
            this.addResult('understanding', `${data.response} (${latency.toFixed(1)}ms)`, new Date());
            this.log(`✅ Understanding response: ${latency.toFixed(1)}ms`, 'success');
        }
        else if (data.type === 'status') {
            this.updatePerformanceDisplay(data);
        }
    }
    
    updatePerformanceStats() {
        const avgLatency = this.responseCount > 0 ? this.totalLatency / this.responseCount : 0;
        const vadRate = this.responseCount > 0 ? this.vadTriggers / this.responseCount : 0;
        
        if (this.elements.performanceStats) {
            this.elements.performanceStats.innerHTML = `
                <div><strong>Performance:</strong></div>
                <div>Responses: ${this.responseCount}</div>
                <div>Avg Latency: ${avgLatency.toFixed(1)}ms</div>
                <div>VAD Triggers: ${this.vadTriggers}</div>
                <div>Target Met: ${avgLatency < 200 ? '✅' : '❌'}</div>
            `;
        }
    }
    
    updatePerformanceDisplay(data) {
        if (data.vad_stats) {
            const vadStats = data.vad_stats;
            this.log(`VAD Stats: ${vadStats.active_connections} connections, ${vadStats.total_speech_segments} segments`, 'info');
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
        this.responseCount = 0;
        this.totalLatency = 0;
        this.vadTriggers = 0;
        this.updatePerformanceStats();
        this.log('Results cleared', 'info');
    }
    
    updateStatus(type, className, text = null) {
        const element = this.elements[type + 'Status'];
        if (element) {
            element.className = 'status ' + className;
            if (text) element.textContent = text;
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
        
        console.log(`[${level.toUpperCase()}] ${message}`);
    }
}

// Initialize client when page loads
document.addEventListener('DOMContentLoaded', () => {
    try {
        window.ultimateVoxtralClient = new UltimateVoxtralClient();
    } catch (error) {
        console.error('Failed to initialize ULTIMATE Voxtral client:', error);
    }
});
