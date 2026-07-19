// static/js/app.js - FIXED VERSION

class FaceRecognitionApp {
    constructor() {
        this.apiUrl = '';
        this.stream = null;
        this.capturedImages = [];
        this.captureMode = 'manual';
        this.isCapturing = false;
        this.searchStream = null;
        this.searchImage = null;
        this.currentMode = 'register'; // ADDED: Initialize currentMode
        
        this.init();
    }
    
    init() {
        this.cacheElements();
        this.bindEvents();
        this.loadStats();
        
        // Debug log
        console.log('FaceRecognitionApp initialized');
    }
    
    cacheElements() {
        // Mode toggle
        this.modeBtns = document.querySelectorAll('.mode-btn');
        this.sections = document.querySelectorAll('.section');
        
        // Register mode elements
        this.userIdInput = document.getElementById('userId');
        this.cameraPermission = document.getElementById('cameraPermission');
        this.cameraPreview = document.getElementById('cameraPreview');
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.startCameraBtn = document.getElementById('startCameraBtn');
        this.captureBtn = document.getElementById('captureBtn');
        this.capturedCount = document.getElementById('capturedCount');
        this.capturedStrip = document.getElementById('capturedStrip');
        this.stripImages = document.getElementById('stripImages');
        this.clearCapturedBtn = document.getElementById('clearCaptured');
        this.uploadBtn = document.getElementById('uploadBtn');
        this.retakeBtn = document.getElementById('retakeBtn');
        this.uploadProgress = document.getElementById('uploadProgress');
        this.uploadFill = document.getElementById('uploadFill');
        this.uploadPercent = document.getElementById('uploadPercent');
        this.resultsSection = document.getElementById('resultsSection');
        this.resultsSummary = document.getElementById('resultsSummary');
        this.burstProgress = document.getElementById('burstProgress');
        this.burstFill = document.getElementById('burstFill');
        this.burstCount = document.getElementById('burstCount');
        this.modeSwitches = document.querySelectorAll('.mode-switch');
        
        // Search mode elements - FIXED: Better error handling for missing elements
        this.searchCameraPermission = document.getElementById('searchCameraPermission');
        this.searchPreview = document.getElementById('searchPreview');
        this.searchVideo = document.getElementById('searchVideo');
        this.startSearchCameraBtn = document.getElementById('startSearchCamera');
        this.searchCaptureBtn = document.getElementById('searchCaptureBtn');
        this.searchResultPreview = document.getElementById('searchResultPreview');
        this.searchResultImg = document.getElementById('searchResultImg');
        this.retakeSearchBtn = document.getElementById('retakeSearch');
        this.submitSearchBtn = document.getElementById('submitSearch');
        this.searchResults = document.getElementById('searchResults');
        this.matchesList = document.getElementById('matchesList');
        
        // Delete mode elements
        this.deleteUserIdInput = document.getElementById('deleteUserId');
        this.deleteUserBtn = document.getElementById('deleteUserBtn');

        // Verify critical elements exist
        if (!this.searchVideo) console.error('searchVideo element not found!');
        if (!this.startSearchCameraBtn) console.error('startSearchCamera button not found!');
        if (!this.submitSearchBtn) console.error('submitSearch button not found!');
        
        // Stats
        this.vectorCount = document.getElementById('vectorCount');
        
        // Toast
        this.toastContainer = document.getElementById('toastContainer');
    }
    
    bindEvents() {
        // Mode switching - FIXED: Proper binding
        this.modeBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const mode = e.currentTarget.dataset.mode;
                console.log('Switching to mode:', mode);
                this.switchMode(mode);
            });
        });
        
        // Register mode events
        if (this.startCameraBtn) {
            this.startCameraBtn.addEventListener('click', () => this.startCamera());
        }
        if (this.captureBtn) {
            this.captureBtn.addEventListener('click', () => this.handleCapture());
        }
        if (this.clearCapturedBtn) {
            this.clearCapturedBtn.addEventListener('click', () => this.clearCaptured());
        }
        if (this.uploadBtn) {
            this.uploadBtn.addEventListener('click', () => this.uploadImages());
        }
        if (this.retakeBtn) {
            this.retakeBtn.addEventListener('click', () => this.retake());
        }
        
        // Capture mode switching
        this.modeSwitches.forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchCaptureMode(e.currentTarget.dataset.capture);
            });
        });
        
        // Search mode events - FIXED: Added null checks and proper binding
        if (this.startSearchCameraBtn) {
            this.startSearchCameraBtn.addEventListener('click', () => {
                console.log('Start search camera clicked');
                this.startSearchCamera();
            });
        }
        
        if (this.searchCaptureBtn) {
            this.searchCaptureBtn.addEventListener('click', () => {
                console.log('Search capture clicked');
                this.captureSearchImage();
            });
        }
        
        if (this.retakeSearchBtn) {
            this.retakeSearchBtn.addEventListener('click', () => {
                console.log('Retake search clicked');
                this.retakeSearch();
            });
        }
        
        if (this.submitSearchBtn) {
            this.submitSearchBtn.addEventListener('click', () => {
                console.log('Submit search clicked');
                this.submitSearchQuery();
            });
        }
        
        // User ID validation
        if (this.userIdInput) {
            this.userIdInput.addEventListener('input', () => this.validateForm());
        }

        // Delete mode events
        if (this.deleteUserBtn) {
            this.deleteUserBtn.addEventListener('click', () => this.deleteUser());
        }
    }
    
    switchMode(mode) {
        console.log('Switching mode from', this.currentMode, 'to', mode);
        
        // Stop current mode's camera
        if (this.currentMode === 'register' && this.stream) {
            this.stopCamera();
        }
        if (this.currentMode === 'search' && this.searchStream) {
            this.stopSearchCamera();
        }
        
        this.currentMode = mode;
        
        // Update UI
        this.modeBtns.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });
        
        this.sections.forEach(section => {
            const shouldShow = (mode === 'register' && section.id === 'registerSection') ||
                              (mode === 'search' && section.id === 'searchSection') ||
                              (mode === 'delete' && section.id === 'deleteSection');
            section.classList.toggle('active', shouldShow);
        });
        
        // Reset search UI when entering search mode
        if (mode === 'search') {
            this.resetSearchUI();
        }
    }
    
    resetSearchUI() {
        console.log('Resetting search UI');
        this.searchImage = null;
        if (this.submitSearchBtn) this.submitSearchBtn.disabled = true;
        if (this.searchResults) this.searchResults.style.display = 'none';
        
        // Reset to initial state
        if (this.searchCameraPermission) this.searchCameraPermission.style.display = 'block';
        if (this.searchPreview) this.searchPreview.style.display = 'none';
        if (this.searchResultPreview) this.searchResultPreview.style.display = 'none';
    }
    
    // ==================== SEARCH FUNCTIONS (FIXED) ====================
    
    async startSearchCamera() {
        console.log('Starting search camera...');
        
        try {
            // Stop any existing stream first
            if (this.searchStream) {
                this.stopSearchCamera();
            }
            
            const constraints = {
                video: {
                    facingMode: 'user',
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            };
            
            console.log('Requesting media with constraints:', constraints);
            this.searchStream = await navigator.mediaDevices.getUserMedia(constraints);
            
            console.log('Got stream:', this.searchStream);
            console.log('Tracks:', this.searchStream.getTracks());
            
            if (!this.searchVideo) {
                throw new Error('Search video element not found!');
            }
            
            this.searchVideo.srcObject = this.searchStream;
            
            // Wait for video to be ready
            await new Promise((resolve) => {
                this.searchVideo.onloadedmetadata = () => {
                    console.log('Search video metadata loaded');
                    resolve();
                };
            });
            
            await this.searchVideo.play();
            console.log('Search video playing');
            
            // Update UI
            if (this.searchCameraPermission) this.searchCameraPermission.style.display = 'none';
            if (this.searchPreview) this.searchPreview.style.display = 'block';
            if (this.searchResultPreview) this.searchResultPreview.style.display = 'none';
            
            this.showToast('Camera started - Position your face and click capture', 'success');
            
        } catch (err) {
            console.error('Search camera error:', err);
            this.showToast('Camera error: ' + err.message, 'error');
            
            // Show helpful message for permission denied
            if (err.name === 'NotAllowedError') {
                this.showToast('Please allow camera access in your browser settings', 'error');
            }
        }
    }
    
    stopSearchCamera() {
        console.log('Stopping search camera');
        if (this.searchStream) {
            this.searchStream.getTracks().forEach(track => {
                console.log('Stopping track:', track.label);
                track.stop();
            });
            this.searchStream = null;
        }
        if (this.searchVideo) {
            this.searchVideo.srcObject = null;
        }
    }
    
    captureSearchImage() {
        console.log('Capturing search image...');
        
        if (!this.searchVideo || !this.searchVideo.videoWidth) {
            this.showToast('Camera not ready yet', 'error');
            return;
        }
        
        try {
            this.searchImage = this.captureFrame(this.searchVideo);
            console.log('Search image captured, length:', this.searchImage.length);
            
            // Update UI
            if (this.searchResultImg) {
                this.searchResultImg.src = this.searchImage;
            }
            
            this.stopSearchCamera();
            
            if (this.searchPreview) this.searchPreview.style.display = 'none';
            if (this.searchResultPreview) this.searchResultPreview.style.display = 'block';
            if (this.submitSearchBtn) this.submitSearchBtn.disabled = false;
            
            this.showToast('Image captured! Click Search Database to find matches', 'success');
            
        } catch (err) {
            console.error('Capture error:', err);
            this.showToast('Failed to capture image: ' + err.message, 'error');
        }
    }
    
    retakeSearch() {
        console.log('Retaking search image');
        this.searchImage = null;
        if (this.submitSearchBtn) this.submitSearchBtn.disabled = true;
        if (this.searchResults) this.searchResults.style.display = 'none';
        
        // Restart camera
        this.startSearchCamera();
    }
    
    async submitSearchQuery() {
        console.log('Submitting search query...');
        
        if (!this.searchImage) {
            this.showToast('No image captured', 'error');
            return;
        }
        
        if (this.submitSearchBtn) {
            this.submitSearchBtn.disabled = true;
            this.submitSearchBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Searching...';
        }
        
        try {
            // Convert base64 to blob
            console.log('Converting image to blob...');
            const response = await fetch(this.searchImage);
            const blob = await response.blob();
            const file = new File([blob], 'search.jpg', { type: 'image/jpeg' });
            
            console.log('File created:', file.size, 'bytes');
            
            const formData = new FormData();
            formData.append('file', file);
            
            console.log('Sending search request...');
            const searchResponse = await fetch(`${this.apiUrl}/search_face?top_k=5`, {
                method: 'POST',
                body: formData
            });
            
            console.log('Search response status:', searchResponse.status);
            
            if (!searchResponse.ok) {
                const errorText = await searchResponse.text();
                throw new Error(`Search failed: ${searchResponse.status} - ${errorText}`);
            }
            
            const data = await searchResponse.json();
            console.log('Search results:', data);
            
            this.displaySearchResults(data);
            
        } catch (error) {
            console.error('Search error:', error);
            this.showToast(error.message, 'error');
        } finally {
            if (this.submitSearchBtn) {
                this.submitSearchBtn.disabled = false;
                this.submitSearchBtn.innerHTML = '<i class="fas fa-search"></i> Search Database';
            }
        }
    }
    
    displaySearchResults(data) {
        console.log('Displaying search results:', data);
        
        if (!this.searchResults || !this.matchesList) {
            console.error('Search results elements not found!');
            return;
        }
        
        this.searchResults.style.display = 'block';
        
        if (!data.matches || data.matches.length === 0) {
            this.matchesList.innerHTML = `
                <div style="text-align: center; color: var(--text-muted); padding: 2rem;">
                    <i class="fas fa-user-slash" style="font-size: 2rem; margin-bottom: 0.5rem; display: block;"></i>
                    No matching faces found in database
                </div>
            `;
            return;
        }
        
        this.matchesList.innerHTML = data.matches.map((match, index) => `
            <div class="match-item" style="animation: slideIn 0.3s ease ${index * 0.1}s both;">
                <div class="match-info">
                    <span class="match-user">
                        <i class="fas fa-user" style="margin-right: 0.5rem; color: var(--primary);"></i>
                        ${match.user_id}
                    </span>
                    <span class="match-confidence">
                        ${new Date(match.timestamp).toLocaleString()}
                    </span>
                </div>
                <div class="match-score" style="
                    background: ${match.confidence > 0.8 ? 'var(--success)' : match.confidence > 0.6 ? 'var(--warning)' : 'var(--error)'};
                    color: white;
                    padding: 0.25rem 0.75rem;
                    border-radius: 9999px;
                    font-size: 0.875rem;
                ">
                    ${(match.confidence * 100).toFixed(1)}%
                </div>
            </div>
        `).join('');
        
        // Scroll to results
        this.searchResults.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    
    // ==================== REGISTER FUNCTIONS (Unchanged) ====================
    
    async startCamera() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'user',
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            });
            
            this.video.srcObject = this.stream;
            this.cameraPermission.style.display = 'none';
            this.cameraPreview.style.display = 'block';
            
            this.showToast('Camera started', 'success');
        } catch (err) {
            console.error('Camera error:', err);
            this.showToast('Could not access camera: ' + err.message, 'error');
        }
    }
    
    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
    }
    
    switchCaptureMode(mode) {
        this.captureMode = mode;
        this.modeSwitches.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.capture === mode);
        });
        
        if (mode === 'burst') {
            this.showToast('Auto burst mode: Will capture 30 images automatically', 'info');
        }
    }
    
    async handleCapture() {
        if (this.isCapturing) return;
        
        if (this.captureMode === 'burst') {
            await this.captureBurst();
        } else {
            this.captureSingle();
        }
    }
    
    captureSingle() {
        if (this.capturedImages.length >= 30) {
            this.showToast('Maximum 30 images reached', 'warning');
            return;
        }
        
        const imageData = this.captureFrame(this.video);
        this.addCapturedImage(imageData);
    }
    
    async captureBurst() {
        this.isCapturing = true;
        this.captureBtn.disabled = true;
        this.burstProgress.style.display = 'flex';
        
        const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
        
        for (let i = 0; i < 30; i++) {
            const imageData = this.captureFrame(this.video);
            this.addCapturedImage(imageData);
            
            const progress = ((i + 1) / 30) * 100;
            this.burstFill.style.width = `${progress}%`;
            this.burstCount.textContent = i + 1;
            
            await delay(100);
        }
        
        this.isCapturing = false;
        this.captureBtn.disabled = false;
        this.burstProgress.style.display = 'none';
        this.showToast('Burst capture complete!', 'success');
    }
    
    captureFrame(videoElement) {
        const canvas = this.canvas;
        canvas.width = videoElement.videoWidth || 640;
        canvas.height = videoElement.videoHeight || 480;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
        
        return canvas.toDataURL('image/jpeg', 0.9);
    }
    
    addCapturedImage(imageData) {
        this.capturedImages.push(imageData);
        this.updateCapturedUI();
        this.validateForm();
        this.playShutterSound();
    }
    
    playShutterSound() {
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();
            
            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);
            
            oscillator.frequency.value = 800;
            oscillator.type = 'sine';
            
            gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1);
            
            oscillator.start(audioContext.currentTime);
            oscillator.stop(audioContext.currentTime + 0.1);
        } catch (e) {
            // Ignore audio errors
        }
    }
    
    updateCapturedUI() {
        this.capturedCount.textContent = this.capturedImages.length;
        
        this.stripImages.innerHTML = this.capturedImages.map((img, idx) => `
            <img src="${img}" class="captured-thumb" alt="Capture ${idx + 1}" title="Image ${idx + 1}">
        `).join('');
        
        this.stripImages.scrollLeft = this.stripImages.scrollWidth;
        
        if (this.capturedImages.length > 0) {
            this.capturedStrip.style.display = 'block';
        }
    }
    
    clearCaptured() {
        this.capturedImages = [];
        this.updateCapturedUI();
        this.capturedStrip.style.display = 'none';
        this.validateForm();
        this.resultsSection.style.display = 'none';
    }
    
    validateForm() {
        const hasUserId = this.userIdInput && this.userIdInput.value.trim().length > 0;
        const hasImages = this.capturedImages.length > 0;
        if (this.uploadBtn) {
            this.uploadBtn.disabled = !(hasUserId && hasImages);
        }
    }
    
    async uploadImages() {
        const userId = this.userIdInput.value.trim();
        if (!userId || this.capturedImages.length === 0) return;
        
        this.uploadBtn.disabled = true;
        this.uploadProgress.style.display = 'block';
        this.retakeBtn.style.display = 'none';
        
        const blobs = await Promise.all(
            this.capturedImages.map(async (dataUrl, idx) => {
                const response = await fetch(dataUrl);
                const blob = await response.blob();
                return new File([blob], `capture_${idx + 1}.jpg`, { type: 'image/jpeg' });
            })
        );
        
        const formData = new FormData();
        formData.append('user_id', userId);
        blobs.forEach(file => formData.append('files', file));
        
        const xhr = new XMLHttpRequest();
        
        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                const percent = Math.round((e.loaded / e.total) * 100);
                this.uploadFill.style.width = `${percent}%`;
                this.uploadPercent.textContent = `${percent}%`;
            }
        });
        
        xhr.addEventListener('load', () => {
            if (xhr.status === 200) {
                const response = JSON.parse(xhr.responseText);
                this.showResults(response);
                this.loadStats();
                this.showToast(`Registered ${response.successful_uploads} faces!`, 'success');
                this.retakeBtn.style.display = 'inline-flex';
            } else {
                this.showToast('Upload failed', 'error');
                this.uploadBtn.disabled = false;
            }
            this.uploadProgress.style.display = 'none';
        });
        
        xhr.addEventListener('error', () => {
            this.showToast('Network error', 'error');
            this.uploadBtn.disabled = false;
            this.uploadProgress.style.display = 'none';
        });
        
        xhr.open('POST', `${this.apiUrl}/register_faces`);
        xhr.send(formData);
    }
    
    showResults(data) {
        this.resultsSection.style.display = 'block';
        this.resultsSummary.innerHTML = `
            <div class="stat-box success">
                <span class="stat-number">${data.successful_uploads}</span>
                <span class="stat-label">Successful</span>
            </div>
            <div class="stat-box error">
                <span class="stat-number">${data.failed_uploads}</span>
                <span class="stat-label">Failed</span>
            </div>
        `;
    }
    
    retake() {
        this.clearCaptured();
        this.resultsSection.style.display = 'none';
        this.retakeBtn.style.display = 'none';
        this.uploadBtn.disabled = true;
    }
    
    // ==================== DELETE FUNCTIONS ====================

    async deleteUser() {
        const userId = this.deleteUserIdInput.value.trim();
        if (!userId) {
            this.showToast('Please enter a User ID', 'warning');
            return;
        }

        if (!confirm(`Are you sure you want to delete user "${userId}"? This cannot be undone.`)) {
            return;
        }

        this.deleteUserBtn.disabled = true;
        this.deleteUserBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Deleting...';

        try {
            const response = await fetch(`${this.apiUrl}/user/${userId}`, {
                method: 'DELETE'
            });

            const data = await response.json();

            if (response.ok) {
                this.showToast(`Deleted user ${userId} (${data.vectors_removed} faces removed)`, 'success');
                this.deleteUserIdInput.value = '';
                this.loadStats();
            } else {
                throw new Error(data.detail || 'Delete failed');
            }
        } catch (error) {
            console.error('Delete error:', error);
            this.showToast(error.message, 'error');
        } finally {
            this.deleteUserBtn.disabled = false;
            this.deleteUserBtn.innerHTML = '<i class="fas fa-trash-alt"></i> Delete User';
        }
    }

    // ==================== UTILITIES ====================
    
    async loadStats() {
        try {
            const response = await fetch(`${this.apiUrl}/stats`);
            const data = await response.json();
            if (this.vectorCount) {
                this.vectorCount.textContent = data.total_vectors.toLocaleString();
            }
        } catch (error) {
            console.error('Failed to load stats:', error);
        }
    }
    
    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icons = {
            success: 'check-circle',
            error: 'times-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };
        
        toast.innerHTML = `
            <i class="fas fa-${icons[type]}"></i>
            <span>${message}</span>
        `;
        
        this.toastContainer.appendChild(toast);
        
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(100%)';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
}

// Initialize
const app = new FaceRecognitionApp();