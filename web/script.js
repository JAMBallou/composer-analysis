// script.js - Composer Classification Web App

let selectedModel = null;
let audioFile = null;
let midiFile = null;

const DEFAULT_MODEL_PATH = 'outputs/models/trial3b_20260307_055047/trial3b_fold5_20260308_120855.keras';

function normalizePath(path) {
    return (path || '').replace(/\\/g, '/').toLowerCase();
}

function getComposerList(model) {
    return Array.isArray(model?.composers) ? model.composers : [];
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    loadModels();
    setupFileInputs();
    setupPredictButton();
});

// Load available models from API
async function loadModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        
        if (!data.success) {
            showError('Failed to load models: ' + data.error);
            return;
        }
        
        displayModels(data.models);
    } catch (error) {
        showError('Failed to connect to server: ' + error.message);
    }
}

// Display models in the UI
function displayModels(models) {
    const container = document.getElementById('modelSelection');
    
    if (models.length === 0) {
        container.innerHTML = '<p class="loading">No trained models found. Please train a model first.</p>';
        return;
    }
    
    container.innerHTML = `
        <div id="selectedModelSummary" class="selected-model-summary"></div>
        <button id="toggleModelListBtn" class="toggle-model-btn" type="button">Change model</button>
        <div id="modelList" class="model-list hidden"></div>
    `;

    const modelList = document.getElementById('modelList');
    const toggleButton = document.getElementById('toggleModelListBtn');

    toggleButton.addEventListener('click', () => {
        const isHidden = modelList.classList.contains('hidden');
        setModelListVisibility(isHidden);
    });

    let defaultSelection = null;
    const normalizedDefaultPath = normalizePath(DEFAULT_MODEL_PATH);
    
    // Group models by trial
    const grouped = {};
    models.forEach(model => {
        if (!grouped[model.trial]) {
            grouped[model.trial] = [];
        }
        grouped[model.trial].push(model);
    });
    
    // Display models grouped by trial
    Object.keys(grouped).forEach(trial => {
        const trialModels = grouped[trial];
        
        // Create a card for the first model of each trial (or let user pick fold)
        trialModels.forEach((model, index) => {
            const card = document.createElement('div');
            card.className = 'model-card';
            card.dataset.modelPath = model.path;
            
            const composers = getComposerList(model);
            const composerCount = composers.length;
            const composerPreview = composers.slice(0, 3).join(', ') + 
                                   (composerCount > 3 ? ` + ${composerCount - 3} more` : '');
            
            card.innerHTML = `
                <h3>${model.description}</h3>
                <p><strong>Model:</strong> ${model.name}</p>
                <p class="composers-list"><strong>Composers (${composerCount}):</strong> ${composerPreview}</p>
            `;
            
            card.addEventListener('click', () => selectModel(card, model));
            modelList.appendChild(card);

            const normalizedModelPath = normalizePath(model.path);
            if (!defaultSelection && normalizedModelPath.endsWith(normalizedDefaultPath)) {
                defaultSelection = { card, model };
            }
        });
    });

    if (!modelList.children.length) {
        modelList.innerHTML = '<p class="loading">No selectable models found.</p>';
        toggleButton.disabled = true;
    }

    if (defaultSelection) {
        selectModel(defaultSelection.card, defaultSelection.model);
    } else {
        const firstCard = modelList.querySelector('.model-card');
        if (firstCard) {
            const firstModel = models.find(m => m.path === firstCard.dataset.modelPath);
            if (firstModel) {
                selectModel(firstCard, firstModel);
            }
        }
    }

}

function setModelListVisibility(visible) {
    const modelList = document.getElementById('modelList');
    const toggleButton = document.getElementById('toggleModelListBtn');
    if (!modelList || !toggleButton) {
        return;
    }

    modelList.classList.toggle('hidden', !visible);
    toggleButton.textContent = visible ? 'Hide models' : 'Change model';
    toggleButton.setAttribute('aria-expanded', visible ? 'true' : 'false');
}

function renderSelectedModelSummary(model) {
    const summary = document.getElementById('selectedModelSummary');
    if (!summary || !model) {
        return;
    }

    const composers = getComposerList(model);
    const composerCount = composers.length;
    const composerPreview = composers.slice(0, 3).join(', ') +
        (composerCount > 3 ? ` + ${composerCount - 3} more` : '');

    summary.innerHTML = `
        <h3>Selected Model</h3>
        <p><strong>${model.description}</strong></p>
        <p>Model: ${model.name}</p>
        <p class="composers-list">Composers (${composerCount}): ${composerPreview}</p>
    `;
}

// Select a model
function selectModel(card, model) {
    // Remove previous selection
    document.querySelectorAll('.model-card').forEach(c => c.classList.remove('selected'));
    
    // Select this card
    card.classList.add('selected');
    selectedModel = model;
    renderSelectedModelSummary(model);
    setModelListVisibility(false);
    
    // Enable predict button if file is uploaded
    updatePredictButton();
}

// Setup file input handlers
function setupFileInputs() {
    const audioInput = document.getElementById('audioFile');
    const midiInput = document.getElementById('midiFile');
    const audioLabel = document.getElementById('audioFileName');
    const midiLabel = document.getElementById('midiFileName');
    
    audioInput.addEventListener('change', function(e) {
        audioFile = e.target.files[0];
        if (audioFile) {
            audioLabel.textContent = audioFile.name;
            audioInput.parentElement.classList.add('has-file');
        } else {
            audioLabel.textContent = 'Choose Audio File';
            audioInput.parentElement.classList.remove('has-file');
        }
        updatePredictButton();
    });
    
    midiInput.addEventListener('change', function(e) {
        midiFile = e.target.files[0];
        if (midiFile) {
            midiLabel.textContent = midiFile.name;
            midiInput.parentElement.classList.add('has-file');
        } else {
            midiLabel.textContent = 'Choose MIDI File';
            midiInput.parentElement.classList.remove('has-file');
        }
        updatePredictButton();
    });
}

// Setup predict button
function setupPredictButton() {
    const button = document.getElementById('predictBtn');
    button.addEventListener('click', handlePredict);
}

// Update predict button state
function updatePredictButton() {
    const button = document.getElementById('predictBtn');
    const hasFile = audioFile || midiFile;
    const hasModel = selectedModel !== null;
    
    button.disabled = !(hasFile && hasModel);
}

// Handle prediction
async function handlePredict() {
    const button = document.getElementById('predictBtn');
    
    // Validate inputs
    if (!selectedModel) {
        showStatus('Please select a model first.', 'error');
        return;
    }
    
    if (!audioFile && !midiFile) {
        showStatus('Please upload at least one file (audio or MIDI).', 'error');
        return;
    }
    
    // Prepare form data
    const formData = new FormData();
    formData.append('model_path', selectedModel.path);
    
    if (audioFile) {
        formData.append('audio_file', audioFile);
    }
    
    if (midiFile) {
        formData.append('midi_file', midiFile);
    }
    
    // Show loading state
    button.disabled = true;
    button.classList.add('loading');
    button.innerHTML = '<span class="loading-spinner"></span> Analyzing...';
    
    // Show helpful message on first prediction
    const firstPrediction = !window._tfLoaded;
    if (firstPrediction) {
        showStatus('Preparing ML model (this may take a few minutes)...', 'loading');
        window._tfLoaded = true;
    } else {
        showStatus('Processing file and making prediction...', 'loading');
    }
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!data.success) {
            showStatus(data.error, 'error');
            resetButton(button);
            return;
        }
        
        // Display results
        displayResults(data.prediction);
        showStatus('Prediction complete!', 'success');
        resetButton(button);
        
    } catch (error) {
        showStatus('Error: ' + error.message, 'error');
        resetButton(button);
    }
}

// Reset button to normal state
function resetButton(button) {
    button.classList.remove('loading');
    button.innerHTML = 'Predict Composer';
    updatePredictButton();
}

// Display prediction results
function displayResults(prediction) {
    const section = document.getElementById('resultsSection');
    const container = document.getElementById('results');
    
    const composer = prediction.composer;
    const confidence = (prediction.confidence * 100).toFixed(1);
    const probabilities = prediction.probabilities;
    const modelInfo = prediction.model_info;
    
    // Sort probabilities by value
    const sortedProbs = Object.entries(probabilities)
        .sort((a, b) => b[1] - a[1]);
    
    // Build probability bars
    let probBars = '';
    sortedProbs.forEach(([name, value], index) => {
        const percentage = (value * 100).toFixed(1);
        const isTop = index === 0;
        probBars += `
            <div class="probability-bar">
                <div class="probability-label">
                    <span class="name">${name}</span>
                    <span class="value">${percentage}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill ${isTop ? 'top-prediction' : ''}" 
                         style="width: ${percentage}%"></div>
                </div>
            </div>
        `;
    });
    
    container.innerHTML = `
        <div class="prediction-result">
            <div class="composer-name">${composer}</div>
            <div class="confidence">Confidence: ${confidence}%</div>
        </div>
        
        <div class="probabilities">
            <h3>All Predictions</h3>
            ${probBars}
        </div>
        
        <div class="model-info">
            <h4>Model Information</h4>
            <p><strong>Trial:</strong> ${modelInfo.trial}</p>
            <p><strong>Description:</strong> ${modelInfo.description}</p>
            <p><strong>Composers:</strong> ${modelInfo.composers.join(', ')}</p>
        </div>
    `;
    
    // Show results section
    section.style.display = 'block';
    
    // Scroll to results
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Show status message
function showStatus(message, type) {
    const status = document.getElementById('status');
    status.textContent = message;
    status.className = 'status visible ' + type;
    
    // Auto-hide success messages after 5 seconds
    if (type === 'success') {
        setTimeout(() => {
            status.classList.remove('visible');
        }, 5000);
    }
}

// Show error in model loading area
function showError(message) {
    const container = document.getElementById('modelSelection');
    container.innerHTML = `
        <p class="loading" style="color: #e74c3c;">
            ⚠️ ${message}
        </p>
    `;
}
