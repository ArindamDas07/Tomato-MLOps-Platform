const imageInput = document.getElementById("image-input");
const previewImage = document.getElementById("preview-image");
const placeholder = document.getElementById("placeholder-text");
const statusDiv = document.getElementById("status");
const resultBox = document.getElementById("result-box");
const errorBox = document.getElementById("error-box");
const predictBtn = document.getElementById("predict-btn");
const uploadBtn = document.getElementById("upload-btn");

let currentUserId = null;

const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// STEP 1: RESET & CHOOSE FILE
uploadBtn.onclick = () => {
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

// STEP 2: AUTO-TRIGGER GATEKEEPER
imageInput.onchange = async () => {
    const file = imageInput.files[0];
    if (!file) return;

    previewImage.src = URL.createObjectURL(file);
    previewImage.style.display = "block";
    placeholder.style.display = "none";

    statusDiv.innerText = "🔍 Validating image quality...";
    
    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("/upload", { method: "POST", body: formData });
        if (!response.ok) throw new Error("Upload failed");
        
        const data = await response.json();
        currentUserId = data.user_id;
        
        await pollService(`/leaf_checker/${data.user_id}/${data.task_id}`, handleGatekeeperResult);
        
    } catch (err) {
        showError("Backend connection failed. Please check your internet.");
    }
};

// STEP 3: SMART POLLING ENGINE
async function pollService(endpoint, successCallback, maxRetries = 40) {
    let retries = 0;
    while (retries < maxRetries) {
        try {
            const res = await fetch(endpoint);
            if (!res.ok) throw new Error("Server communication error");
            
            const data = await res.json();

            if (data.status === "done" || data.status === "error") {
                successCallback(data);
                return; 
            }

            retries++;
            // Linear Backoff
            await delay(1500 + (retries * 50)); 
            
        } catch (e) {
            console.error("Polling error caught:", e);
            showError("Service connection lost.");
            return;
        }
    }
    showError("Request timed out. The server is taking too long.");
}

// CALLBACKS FOR POLLING
function handleGatekeeperResult(data) {
    if (data.valid) {
        statusDiv.innerText = "✅ Tomato leaf detected! You can now predict.";
        predictBtn.disabled = false;
    } else {
        showError(`❌ Reject: ${data.message || "Not a tomato leaf"}`);
    }
}

function handleFinalResult(data) {
    if (data.status === "done" && data.prediction) {
        statusDiv.innerText = "Classification Complete";
        
        //  'data.prediction' is already a JSON object from FastAPI.
        // We removed JSON.parse() to prevent the code from crashing.
        const prediction = data.prediction; 
        
        document.getElementById("disease").innerText = prediction.disease;
        document.getElementById("confidence").innerText = prediction.confidence;
        document.getElementById("model-used").innerText = prediction.model.toUpperCase();
        
        resultBox.style.display = "block";
    } else {
        showError("Final inference failed on the worker side.");
    }
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
        
        await pollService(`/result/${currentUserId}/${data.task_id}`, handleFinalResult);
    } catch (err) {
        showError("Prediction request failed.");
    }
};

function showError(msg) {
    statusDiv.innerText = "Ready";
    errorBox.innerText = msg;
    errorBox.style.display = "block";
    predictBtn.disabled = true;
}