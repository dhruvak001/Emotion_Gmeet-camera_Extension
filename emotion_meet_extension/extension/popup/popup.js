document.addEventListener('DOMContentLoaded', () => {
    const toggleButton = document.getElementById('toggleDetection');
    const statusText = document.getElementById('statusText');
    
    let isActive = false;
  
    toggleButton.addEventListener('click', () => {
      isActive = !isActive;
      statusText.textContent = isActive ? 'Active' : 'Inactive';
      chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
        chrome.tabs.sendMessage(tabs[0].id, {
          action: 'toggleDetection',
          isActive: isActive
        });
      });
    });
  });