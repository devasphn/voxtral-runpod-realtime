// PURE UNDERSTANDING-ONLY WEBSOCKET CLIENT - NO TRANSCRIPTION
class VoxtralPureUnderstandingClient {
    constructor() {
        this.websocket = null;
        this.isConnected = false;
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioStream = null;
        
        // PURE UNDERSTANDING-ONLY: Enhanced audio configuration for gap detection
        this.audioConfig = {
            sampleRate: 16000,
            channels: 1,
            mimeType: 'audio/webm;codecs=opus'
        };
        
        // UNDERSTANDING-ONLY: Gap detection state
        this.gapDetectionState = {
            currentSilenceDuration: 0,
            totalSegmentDuration: 0,
            gapThreshold: 300, // 0.3 seconds
            lastSpeechTime: 0,
            speechDetected: false
        };
        
        // Connection retry configuration
        this.maxRetries = 3;
        this.retryDelay = 2000;
        this.currentRetries = 0;
        
        // Performance tracking
        this.performanceMetrics = {
            responseTimesMs: [],
            averageResponseMs: 0,
            sub200msCount: 0,
            totalResponses: 0,
            contextTurns: 0
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
            connectionStatus: document.getElementById('connection-status'),
            modelStatus: document.getElementById('model-status'),
            audioStatus: document.getElementById('audio-status'),
            gapStatus: document.getElementById('gap-status'),
            resultsContainer: document.getElementById('results-container'),
            clearResults: document.getElementById('clear-results'),
            systemInfo: document.getElementById('system-info'),
            logsContainer: document.getElementById('logs-container'),
            gapMetric: document.getElementById('gap-metric'),
            responseMetric: document.getElementById('response-metric'),
            contextMetric: document.getElementById('context-metric'),
            speedMetric: document.getElementById('speed-metric')
        };
        
        // Bind event listeners
        this.elements.connectBtn.addEventListener('click', () => this.connect());
        this.elements.disconnectBtn.addEventListener('click', () => this.disconnect());
        this.elements.recordBtn.addEventListener('click', () => this.toggleRecording());
        this.elements.clearResults.addEventListener('click', () => this.clearResults());
        
        // Update UI for pure understanding-only mode
        this.updateGapDetectionUI();
        this.updateModeDisplay();
    }
    
    updateModeDisplay() {
        // Update page title and headers to reflect pure understanding-only
        document.title = '🧠 Voxtral PURE UNDERSTANDING-ONLY - Real-Time Conversational AI';
        
        // Update any transcription references to understanding-only
        const pageElements = document.querySelectorAll('[data-mode-update]');
        pageElements.forEach(el => {
            if (el.textContent.includes('transcription')) {
                el.textContent = el.textContent.replace(/transcription/gi, 'understanding');
            }
        });
    }
    
    updateGapDetectionUI() {
        // Update gap detection status
        if (this.elements.gapStatus) {
            this.elements.gapStatus.textContent = `${this.gapDetectionState.gapThreshold}ms`;
            this.elements.gapStatus.className = 'status';
        }
        
        // Update performance metrics
        if (this.elements.gapMetric) {
            this.elements.gapMetric.textContent = `${this.gapDetectionState.gapThreshold}ms`;
        }
        
        if (this.elements.responseMetric) {
            this.elements.responseMetric.textContent = `${this.performanceMetrics.averageResponseMs}ms`;
        }
        
        if (this.elements.contextMetric) {
            this.elements.contextMetric.textContent = `${this.performanceMetrics.contextTurns}`;
        }
        
        if (this.elements.speedMetric) {
            const rate = this.performanceMetrics.totalResponses > 0 
                ? Math.round((this.performanceMetrics.sub200msCount / this.performanceMetrics.totalResponses) * 100)
                : 0;
            this.elements.speedMetric.textContent = `${rate}%`;
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
                <div><strong>Mode:</strong> <span class="understanding-highlight">PURE UNDERSTANDING-ONLY</span></div>
                <div><strong>Transcription:</strong> <span class="danger-color">COMPLETELY DISABLED</span></div>
                <div><strong>Model:</strong> ${info.model_name}</div>
                <div><strong>Device:</strong> ${info.device}</div>
                <div><strong>Parameters:</strong> ${info.model_size}</div>
                <div><strong>Flash Attention:</strong> <span class="warning-color">${info.flash_attention_status}</span></div>
                <div><strong>Gap Detection:</strong> <span class="gap-highlight">${info.gap_detection_ms}ms</span></div>
                <div><strong>Target Response:</strong> <span class="speed-highlight">${info.target_response_ms}ms</span></div>
                <div><strong>Languages:</strong> ${info.supported_languages.join(', ')}</div>
                <div><strong>Features:</strong> Conversational AI, Context Memory, WebRTC VAD</div>
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
        return `${protocol}//${host}/ws/understand`;  // PURE UNDERSTANDING-ONLY endpoint
    }
    
    async connect() {
        if (this.isConnected) return;
        
        this.log('Connecting to PURE UNDERSTANDING-ONLY service...', 'info');
        
        try {
            const wsUrl = this.getWebSocketURL();
            this.websocket = new WebSocket(wsUrl);
            
            // Set up event handlers
            this.websocket.onopen = () => {
                this.isConnected = true;
                this.currentRetries = 0;
                this.updateStatus('connection', 'connected', 'connected');
                this.updateButtons();
                this.log('✅ Connected to PURE UNDERSTANDING-ONLY service', 'success');
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
        
        this.log('Disconnecting from PURE UNDERSTANDING-ONLY service...', 'info');
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
            this.log('Not connected to PURE UNDERSTANDING-ONLY service', 'error');
            return;
        }
        
        try {
            this.log('Starting continuous recording for PURE UNDERSTANDING-ONLY...', 'info');
            
            // Request microphone access with enhanced constraints
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
            
            // PURE UNDERSTANDING-ONLY: Handle continuous audio data
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data && event.data.size > 0 && this.isConnected) {
                    this.sendAudioData(event.data);
                }
            };
            
            this.mediaRecorder.onerror = (error) => {
                this.log('MediaRecorder error: ' + error.error, 'error');
                this.stopRecording();
            };
            
            // PURE UNDERSTANDING-ONLY: Start continuous recording with small intervals
            this.mediaRecorder.start(100); // 100ms chunks for responsive gap detection
            
            this.isRecording = true;
            this.updateStatus('audio', 'recording', 'recording continuously');
            this.updateButtons();
            this.log(`✅ Continuous recording started for PURE UNDERSTANDING-ONLY (${mimeType})`, 'success');
            
        } catch (error) {
            this.log('Failed to start recording: ' + error.message, 'error');
            this.updateStatus('audio', 'error', 'error');
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
            this.updateStatus('audio', 'stopped', 'not recording');
            this.updateButtons();
            this.log('Recording stopped', 'info');
            
        } catch (error) {
            this.log('Error stopping recording: ' + error.message, 'error');
        }
    }
    
    async sendAudioData(audioBlob) {
        try {
            if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
                return;
            }
            
            // PURE UNDERSTANDING-ONLY: Send binary audio data directly
            const arrayBuffer = await audioBlob.arrayBuffer();
            if (arrayBuffer.byteLength > 0) {
                this.websocket.send(arrayBuffer);
            }
        } catch (error) {
            this.log('Failed to send audio data: ' + error.message, 'error');
        }
    }
    
    handleMessage(data) {
        if (data.type === 'connection') {
            this.log(data.message, 'info');
            if (data.mode === 'PURE UNDERSTANDING-ONLY') {
                this.log('✅ Confirmed: PURE UNDERSTANDING-ONLY mode active', 'success');
            }
            return;
        }
        
        if (data.error) {
            this.log('Service error: ' + data.error, 'error');
            this.addResult('error', `❌ ${data.error}`, new Date());
            return;
        }
        
        // PURE UNDERSTANDING-ONLY: Handle audio feedback (gap detection status)
        if (data.type === 'audio_feedback') {
            this.updateGapDetectionFeedback(data);
            return;
        }
        
        // PURE UNDERSTANDING-ONLY: Handle complete understanding results
        if (data.type === 'understanding' && data.response) {
            this.handleUnderstandingResponse(data);
            return;
        }
        
        // Handle other message types
        this.log(`Received message: ${data.type}`, 'info');
    }
    
    updateGapDetectionFeedback(data) {
        // Update gap detection state
        this.gapDetectionState.currentSilenceDuration = data.silence_duration_ms || 0;
        this.gapDetectionState.totalSegmentDuration = data.total_duration_ms || 0;
        this.gapDetectionState.speechDetected = data.speech_detected || false;
        
        // Update UI with real-time feedback
        const remaining = Math.max(0, data.remaining_to_gap_ms || 0);
        
        if (this.elements.gapStatus) {
            if (remaining > 0) {
                this.elements.gapStatus.textContent = `${remaining.toFixed(0)}ms to gap`;
                this.elements.gapStatus.className = 'status recording';
            } else {
                this.elements.gapStatus.textContent = 'Processing...';
                this.elements.gapStatus.className = 'status processing';
            }
        }
        
        // Log gap detection progress
        if (data.speech_detected) {
            this.log(`🎤 Speech: ${data.segment_duration_ms?.toFixed(0)}ms, total: ${data.total_duration_ms?.toFixed(0)}ms`, 'info');
        } else {
            this.log(`🔇 Silence: ${data.silence_duration_ms?.toFixed(0)}ms (${remaining.toFixed(0)}ms to gap)`, 'info');
        }
    }
    
    handleUnderstandingResponse(data) {
        // Update performance metrics
        const responseTime = data.response_time_ms || 0;
        this.performanceMetrics.responseTimesMs.push(responseTime);
        this.performanceMetrics.totalResponses++;
        this.performanceMetrics.contextTurns = data.conversation?.turns || 0;
        
        if (responseTime < 200) {
            this.performanceMetrics.sub200msCount++;
        }
        
        // Calculate average response time
        if (this.performanceMetrics.responseTimesMs.length > 0) {
            const sum = this.performanceMetrics.responseTimesMs.reduce((a, b) => a + b, 0);
            this.performanceMetrics.averageResponseMs = Math.round(sum / this.performanceMetrics.responseTimesMs.length);
        }
        
        // Update UI metrics
        this.updateGapDetectionUI();
        
        if (this.elements.responseMetric) {
            this.elements.responseMetric.textContent = `${responseTime.toFixed(0)}ms`;
            this.elements.responseMetric.className = responseTime < 200 ? 'metric-value speed-highlight' : 'metric-value';
        }
        
        // Reset gap status
        if (this.elements.gapStatus) {
            this.elements.gapStatus.textContent = '300ms';
            this.elements.gapStatus.className = 'status';
        }
        
        // Add result to UI
        const resultText = `🎤: "${data.transcription}"\n\n🧠: ${data.response}`;
        this.addResult('understanding', resultText, new Date(), {
            responseTime: responseTime,
            sub200ms: data.sub_200ms,
            audioDuration: data.audio_duration_ms,
            speechQuality: data.speech_quality,
            gapDetected: data.gap_detected,
            flashAttentionDisabled: data.flash_attention_disabled
        });
        
        this.log(`✅ PURE UNDERSTANDING response: ${responseTime.toFixed(0)}ms ${data.sub_200ms ? '⚡' : ''} ${data.flash_attention_disabled ? '🚫⚡' : ''}`, 'success');
    }
    
    addResult(type, content, timestamp, metadata = null) {
        // Remove "no results" message
        const noResults = this.elements.resultsContainer.querySelector('.no-results');
        if (noResults) {
            noResults.remove();
        }
        
        // Create result element
        const resultElement = document.createElement('div');
        resultElement.className = 'result-item';
        
        // Color coding based on type
        let typeClass = 'result-understanding';
        if (type === 'error') typeClass = 'result-error';
        
        let metadataHtml = '';
        if (metadata) {
            metadataHtml = `
                <div class="result-metadata">
                    ${metadata.responseTime ? `Response: ${metadata.responseTime.toFixed(0)}ms ${metadata.sub200ms ? '⚡' : ''}` : ''}
                    ${metadata.audioDuration ? `Audio: ${metadata.audioDuration.toFixed(0)}ms` : ''}
                    ${metadata.speechQuality ? `Quality: ${(metadata.speechQuality * 100).toFixed(0)}%` : ''}
                    ${metadata.gapDetected ? '🎯 Gap Detected' : ''}
                    ${metadata.flashAttentionDisabled ? '🚫⚡ Flash Attention Disabled' : ''}
                </div>
            `;
        }
        
        resultElement.innerHTML = `
            <div class="result-header">
                <span class="result-type ${typeClass}">PURE UNDERSTANDING</span>
                <span class="result-time">${timestamp.toLocaleTimeString()}</span>
            </div>
            <div class="result-content">${content}</div>
            ${metadataHtml}
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
        this.elements.resultsContainer.innerHTML = '<p class="no-results">No results yet. Connect and start recording for PURE UNDERSTANDING-ONLY conversational AI.</p>';
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
            this.elements.recordBtn.textContent = 'Stop Continuous Recording';
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

// Initialize PURE UNDERSTANDING-ONLY client when page loads
document.addEventListener('DOMContentLoaded', () => {
    try {
        window.voxtralClient = new VoxtralPureUnderstandingClient();
        console.log('✅ PURE UNDERSTANDING-ONLY Voxtral client initialized');
    } catch (error) {
        console.error('Failed to initialize PURE UNDERSTANDING-ONLY Voxtral client:', error);
    }
});
