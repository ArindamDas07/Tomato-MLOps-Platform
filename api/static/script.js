const imageInput = document.getElementById("image-input");
const previewImage = document.getElementById("preview-image");
const placeholder = document.getElementById("placeholder-text");
const statusDiv = document.getElementById("status");
const resultBox = document.getElementById("result-box");
const errorBox = document.getElementById("error-box");
const predictBtn = document.getElementById("predict-btn");
const uploadBtn = document.getElementById("upload-btn");

let currentUserId = null;

// STEP 1: RESET & CHOOSE FILE
uploadBtn.onclick = () => {
    // Reset UI for fresh start
    currentUserId = null;
    imageInput.value = "";
    previewImage.style.display = "none";
    placeholder.style.display = "block";
    resultBox.style.display = "none";
    errorBox.style.display = "none";
    predictBtn.disabled = true;
    statusDiv.innerText = "Waiting for file...";
    
    imageInput.click();
};

// STEP 2: AUTO-TRIGGER GATEKEEPER ON SELECTION
imageInput.onchange = async () => {
    const file = imageInput.files[0];
    if (!file) return;

    // Show preview immediately
    previewImage.src = URL.createObjectURL(file);
    previewImage.style.display = "block";
    placeholder.style.display = "none";

    // Start background validation
    statusDiv.innerText = "🔍 Validating image quality...";
    
    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("/upload", { method: "POST", body: formData });
        const data = await response.json();
        
        currentUserId = data.user_id;
        pollGatekeeper(data.user_id, data.task_id);
        
    } catch (err) {
        showError("Backend connection failed.");
    }
};

// STEP 3: POLL GATEKEEPER (Leaf Validation)
async function pollGatekeeper(userId, taskId) {
    const interval = setInterval(async () => {
        try {
            const res = await fetch(`/leaf_checker/${userId}/${taskId}`);
            const data = await res.json();

            if (data.status === "processing") return;

            clearInterval(interval);
            if (data.valid) {
                statusDiv.innerText = "✅ Tomato leaf detected! You can now predict.";
                predictBtn.disabled = false; // UNFREEZE BUTTON
            } else {
                showError("❌ Error: " + data.message);
            }
        } catch (e) {
            clearInterval(interval);
            showError("Validation service interrupted.");
        }
    }, 1500);
}

// STEP 4: TRIGGER PREDICTION
predictBtn.onclick = async () => {
    if (!currentUserId) return;

    predictBtn.disabled = true;
    statusDiv.innerText = "🧠 Analyzing for diseases (A/B Test running)...";
    errorBox.style.display = "none";

    try {
        const response = await fetch(`/predict/${currentUserId}`, { method: "POST" });
        const data = await response.json();
        pollFinalResult(currentUserId, data.task_id);
    } catch (err) {
        showError("Prediction request failed.");
    }
};

// STEP 5: POLL FINAL RESULT
async function pollFinalResult(userId, taskId) {
    const interval = setInterval(async () => {
        try {
            const res = await fetch(`/result/${userId}/${taskId}`);
            const data = await res.json();

            if (data.status === "processing") return;

            clearInterval(interval);
            if (data.status === "done") {
                statusDiv.innerText = "Classification Complete";
                // Data comes from our JSON.dumps in worker.py
                const prediction = JSON.parse(data.prediction); 
                
                document.getElementById("disease").innerText = prediction.disease;
                document.getElementById("confidence").innerText = prediction.confidence;
                // If you logged model_used in worker, you can show it:
                // document.getElementById("model-used").innerText = prediction.model;
                
                resultBox.style.display = "block";
            } else {
                showError("Final inference failed.");
            }
        } catch (e) {
            clearInterval(interval);
            showError("Result retrieval failed.");
        }
    }, 1500);
}

function showError(msg) {
    statusDiv.innerText = "Ready";
    errorBox.innerText = msg;
    errorBox.style.display = "block";
    predictBtn.disabled = true;
}