// Tab switching functionality
document.addEventListener('DOMContentLoaded', function() {
    const methodTabs = document.querySelectorAll('.method-tab');
    const inputMethods = document.querySelectorAll('.input-method');
    
    methodTabs.forEach(tab => {
        tab.addEventListener('click', function() {
            const method = this.getAttribute('data-method');
            
            // Update active tab
            methodTabs.forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            
            // Update active input method
            inputMethods.forEach(im => im.classList.remove('active'));
            document.getElementById(method + '-method').classList.add('active');
        });
    });
    
    // File upload functionality
    setupFileUpload();
    
    // Text area character counter
    setupTextCounter();
});

function setupFileUpload() {
    const fileInput = document.getElementById('policy-file');
    const dropZone = document.getElementById('file-drop-zone');
    const analyzeBtn = document.getElementById('file-analyze-btn');
    
    // File input change
    fileInput.addEventListener('change', function(e) {
        handleFileSelect(e.target.files[0]);
    });
    
    // Drag and drop
    dropZone.addEventListener('dragover', function(e) {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    
    dropZone.addEventListener('dragleave', function(e) {
        e.preventDefault();
        dropZone.classList.remove('dragover');
    });
    
    dropZone.addEventListener('drop', function(e) {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
    
    function handleFileSelect(file) {
        if (file) {
            dropZone.querySelector('p').textContent = `Selected: ${file.name}`;
            analyzeBtn.disabled = false;
        }
    }
}

function setupTextCounter() {
    const textArea = document.getElementById('policy-text');
    const charCount = document.getElementById('char-count');
    const wordCount = document.getElementById('word-count');
    
    textArea.addEventListener('input', function() {
        const text = this.value;
        const chars = text.length;
        const words = text.trim() ? text.trim().split(/\s+/).length : 0;
        
        charCount.textContent = `${chars} characters`;
        wordCount.textContent = `${words} words`;
    });
}

// Analysis functions
function analyzeFromURL() {
    const url = document.getElementById('policy-url').value;
    if (!url) {
        alert('Please enter a valid URL');
        return;
    }
    startAnalysis('url', url);
}

function analyzeFromFile() {
    const file = document.getElementById('policy-file').files[0];
    if (!file) {
        alert('Please select a file');
        return;
    }
    startAnalysis('file', file);
}

function analyzeFromText() {
    const text = document.getElementById('policy-text').value;
    if (!text.trim()) {
        alert('Please enter some text');
        return;
    }
    startAnalysis('text', text);
}

function startAnalysis(method, data) {
    // Show loading state
    const submitBtn = document.querySelector('.submit-compute-btn');
    const originalText = submitBtn.innerHTML;
    
    submitBtn.innerHTML = '<span>🔄 Analyzing...</span>';
    submitBtn.disabled = true;
    
    // Simulate analysis process
    setTimeout(() => {
        alert(`Analysis started for ${method} input!\n\nIn a real implementation, this would:\n1. Process your ${method} input\n2. Run GDPR compliance analysis\n3. Generate comprehensive report\n4. Display results with actionable recommendations`);
        
        // Reset button
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
    }, 2000);
}
// Smooth cursor follower effect
const cursor = document.querySelector('.cursor-follower');
document.addEventListener('mousemove', (e) => {
    cursor.style.transform = `translate3d(${e.clientX-16}px,${e.clientY-16}px,0)`;
});
