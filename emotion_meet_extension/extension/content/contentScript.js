let isDetectionActive = false;
let overlayCanvas = null;
let processingInterval = null;

// Create overlay canvas
function createOverlay() {
  overlayCanvas = document.createElement('canvas');
  overlayCanvas.id = 'emotionOverlay';
  overlayCanvas.style.position = 'absolute';
  overlayCanvas.style.top = '0';
  overlayCanvas.style.left = '0';
  overlayCanvas.style.zIndex = '100000';
  overlayCanvas.style.pointerEvents = 'none';
  document.body.appendChild(overlayCanvas);
}

// Process video frame
async function processFrame(videoElement) {
  if (!isDetectionActive) return;

  const tempCanvas = document.createElement('canvas');
  const tempCtx = tempCanvas.getContext('2d');
  
  tempCanvas.width = videoElement.videoWidth;
  tempCanvas.height = videoElement.videoHeight;
  tempCtx.drawImage(videoElement, 0, 0);

  try {
    const blob = await new Promise(resolve => 
      tempCanvas.toBlob(resolve, 'image/jpeg', 0.8)
    );
    
    const formData = new FormData();
    formData.append('image', blob, 'frame.jpg');

    const response = await fetch('http://localhost:5000/detect', {
      method: 'POST',
      body: formData
    });
    
    const results = await response.json();
    drawResults(results);
  } catch (error) {
    console.error('Detection error:', error);
  }
}

// Draw detection results
function drawResults(results) {
  const ctx = overlayCanvas.getContext('2d');
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  results.forEach(result => {
    // Draw bounding box
    ctx.strokeStyle = '#00FF00';
    ctx.lineWidth = 2;
    ctx.strokeRect(result.x, result.y, result.width, result.height);

    // Draw emotion label
    ctx.fillStyle = '#00FF00';
    ctx.font = '16px Arial';
    ctx.fillText(
      `${result.emotion} (${Math.round(result.confidence * 100)}%)`,
      result.x,
      result.y - 5
    );
  });
}

// Main detection loop
function startDetection(videoElement) {
  overlayCanvas.width = videoElement.offsetWidth;
  overlayCanvas.height = videoElement.offsetHeight;

  processingInterval = setInterval(() => {
    if (videoElement.readyState === HTMLMediaElement.HAVE_ENOUGH_DATA) {
      processFrame(videoElement);
    }
  }, 1000); // Process 1 frame/second
}

// Find Meet video element
function findVideoElement() {
  const videoElement = document.querySelector('video');
  if (videoElement && !overlayCanvas) {
    createOverlay();
    startDetection(videoElement);
  }
}

// Message listener
chrome.runtime.onMessage.addListener((message) => {
  if (message.action === 'toggleDetection') {
    isDetectionActive = message.isActive;
    if (!isDetectionActive) {
      clearInterval(processingInterval);
    }
  }
});

// Initialize
setInterval(findVideoElement, 1000);