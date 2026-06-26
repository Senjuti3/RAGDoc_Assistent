document.addEventListener('DOMContentLoaded', () => {
    // ----------------------------------------------------
    // DOM Elements
    // ----------------------------------------------------
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const progressContainer = document.getElementById('progressContainer');
    const progressFileName = document.getElementById('progressFileName');
    const progressPercent = document.getElementById('progressPercent');
    const progressBar = document.getElementById('progressBar');
    const uploadStatusText = document.getElementById('uploadStatusText');
    
    const documentList = document.getElementById('documentList');
    const docCount = document.getElementById('docCount');
    const clearBtn = document.getElementById('clearBtn');
    
    const referenceToggle = document.getElementById('referenceToggle');
    const chatFeed = document.getElementById('chatFeed');
    const chatForm = document.getElementById('chatForm');
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');
    
    const toast = document.getElementById('toast');
    const welcomeOverlay = document.getElementById('welcomeOverlay');
    const letsGoBtn = document.getElementById('letsGoBtn');

    // ----------------------------------------------------
    // App Session Management (Clean DB per connection)
    // ----------------------------------------------------
    // SessionStorage is automatically cleared when the user closes the tab/browser.
    let sessionId = sessionStorage.getItem('rag_session_id');
    if (!sessionId) {
        sessionId = uuidv4();
        sessionStorage.setItem('rag_session_id', sessionId);
    }

    // ----------------------------------------------------
    // App State
    // ----------------------------------------------------
    let isUploading = false;
    let isQuerying = false;

    // ----------------------------------------------------
    // Startup initialization
    // ----------------------------------------------------
    fetchIndexedFiles();

    // Dismiss welcome screen
    if (letsGoBtn && welcomeOverlay) {
        letsGoBtn.addEventListener('click', () => {
            welcomeOverlay.style.opacity = '0';
            setTimeout(() => {
                welcomeOverlay.style.display = 'none';
            }, 400);
        });
    }

    // Auto-expand textarea as user types
    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = (chatInput.scrollHeight) + 'px';
    });

    // Handle Shift+Enter for new line, Enter for submit
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });

    // Toggle showing references globally and dynamically
    if (referenceToggle) {
        referenceToggle.addEventListener('change', () => {
            const showReferences = referenceToggle.checked;
            const containers = document.querySelectorAll('.sources-container');
            containers.forEach(container => {
                container.style.display = showReferences ? 'block' : 'none';
            });
        });
    }

    // ----------------------------------------------------
    // Ingestion & File Upload Logic
    // ----------------------------------------------------
    
    // Trigger file dialog
    dropzone.addEventListener('click', () => {
        if (isUploading) return;
        fileInput.click();
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFileUpload(fileInput.files[0]);
        }
    });

    // Drag & drop events
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (isUploading) return;
            dropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        if (isUploading) return;
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    // XHR Upload (to support progress bar and custom headers)
    function handleFileUpload(file) {
        const allowedExtensions = ['pdf', 'docx', 'txt'];
        const fileExtension = file.name.split('.').pop().toLowerCase();
        
        if (!allowedExtensions.includes(fileExtension)) {
            showToast('Invalid file format. Please upload PDF, DOCX or TXT files.', 'error');
            return;
        }

        isUploading = true;
        progressContainer.style.display = 'block';
        progressFileName.textContent = file.name;
        progressPercent.textContent = '0%';
        progressBar.style.width = '0%';
        uploadStatusText.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Uploading file...';

        const formData = new FormData();
        formData.append('file', file);

        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/upload', true);
        
        // Add Session ID Header for database isolation
        xhr.setRequestHeader('X-Session-ID', sessionId);

        // Upload progress listener
        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                const percentage = Math.round((e.loaded / e.total) * 100);
                progressBar.style.width = percentage + '%';
                progressPercent.textContent = percentage + '%';
                if (percentage === 100) {
                    uploadStatusText.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Processing & indexing document chunks...';
                }
            }
        });

        // Upload finish listener
        xhr.onreadystatechange = () => {
            if (xhr.readyState === XMLHttpRequest.DONE) {
                isUploading = false;
                fileInput.value = ''; // Reset file input

                try {
                    const response = JSON.parse(xhr.responseText);
                    if (xhr.status === 200 && response.success) {
                        progressBar.style.width = '100%';
                        progressPercent.textContent = '100%';
                        uploadStatusText.innerHTML = '<span style="color: var(--color-success);"><i class="fa-solid fa-circle-check"></i> Indexed successfully!</span>';
                        showToast(`Successfully indexed: ${file.name}`, 'success');
                        
                        // Refresh files list
                        fetchIndexedFiles();
                        
                        // Auto-hide progress card after 3 seconds
                        setTimeout(() => {
                            if (!isUploading) {
                                progressContainer.style.display = 'none';
                            }
                        }, 3000);
                    } else {
                        handleUploadFailure(response.error || 'Unknown server error');
                    }
                } catch (e) {
                    handleUploadFailure('Server returned an invalid JSON response.');
                }
            }
        };

        xhr.onerror = () => {
            isUploading = false;
            handleUploadFailure('Network connection error.');
        };

        xhr.send(formData);
    }

    function handleUploadFailure(errorMsg) {
        progressBar.style.width = '0%';
        progressPercent.textContent = 'Error';
        uploadStatusText.innerHTML = `<span style="color: var(--color-danger);"><i class="fa-solid fa-circle-exclamation"></i> ${errorMsg}</span>`;
        showToast(errorMsg, 'error');
    }

    // ----------------------------------------------------
    // Fetch Indexed Files
    // ----------------------------------------------------
    async function fetchIndexedFiles() {
        try {
            const res = await fetch('/files', {
                headers: { 'X-Session-ID': sessionId }
            });
            const data = await res.json();
            if (data.success) {
                renderFileList(data.files);
            }
        } catch (e) {
            console.error('Error fetching indexed files:', e);
        }
    }

    function renderFileList(files) {
        documentList.innerHTML = '';
        docCount.textContent = files.length;

        if (files.length === 0) {
            documentList.innerHTML = '<li class="empty-list-message">No documents indexed yet.</li>';
            return;
        }

        files.forEach(filename => {
            const ext = filename.split('.').pop().toLowerCase();
            const li = document.createElement('li');
            li.classList.add(`${ext}-item`);
            
            let iconClass = 'fa-file-lines';
            if (ext === 'pdf') iconClass = 'fa-file-pdf';
            else if (ext === 'docx') iconClass = 'fa-file-word';

            li.innerHTML = `
                <i class="fa-solid ${iconClass}"></i>
                <span title="${filename}">${filename}</span>
            `;
            documentList.appendChild(li);
        });
    }

    // ----------------------------------------------------
    // Clear Database Logic
    // ----------------------------------------------------
    clearBtn.addEventListener('click', async () => {
        if (confirm('Are you sure you want to clear all indexed documents? This will permanently delete uploaded files and empty the vector store.')) {
            try {
                const res = await fetch('/clear', { 
                    method: 'POST',
                    headers: { 'X-Session-ID': sessionId }
                });
                const data = await res.json();
                if (data.success) {
                    showToast('Database cleared successfully.', 'success');
                    fetchIndexedFiles();
                    // Clear chat feed
                    chatFeed.innerHTML = `
                        <div class="message assistant-message welcome-message">
                            <div class="message-avatar">
                                <i class="fa-solid fa-robot"></i>
                            </div>
                            <div class="message-content">
                                <p>Database cleared! Upload new files in the sidebar to ask questions.</p>
                            </div>
                        </div>
                    `;
                } else {
                    showToast(data.error || 'Failed to clear database.', 'error');
                }
            } catch (e) {
                showToast('Failed to clear database due to network error.', 'error');
            }
        }
    });

    // ----------------------------------------------------
    // Chat & RAG Query Logic
    // ----------------------------------------------------
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const question = chatInput.value.trim();
        if (!question || isQuerying) return;

        // Reset input immediately
        chatInput.value = '';
        chatInput.style.height = 'auto';

        // Add user message to UI
        appendUserMessage(question);

        // Add typing indicator
        const typingEl = appendTypingIndicator();
        chatFeed.scrollTop = chatFeed.scrollHeight;

        isQuerying = true;
        setFormDisabled(true);

        try {
            const showReferences = referenceToggle.checked;
            const response = await fetch('/ask', {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'X-Session-ID': sessionId
                },
                body: JSON.stringify({ question, show_references: showReferences })
            });

            const data = await response.json();
            
            // Remove typing indicator
            typingEl.remove();

            if (data.success) {
                appendAssistantMessage(data.answer, data.sources);
            } else {
                appendAssistantMessage(`<span style="color: var(--color-danger);"><i class="fa-solid fa-triangle-exclamation"></i> Error: ${data.error}</span>`);
            }
        } catch (err) {
            typingEl.remove();
            appendAssistantMessage(`<span style="color: var(--color-danger);"><i class="fa-solid fa-triangle-exclamation"></i> Network error: Failed to connect to server.</span>`);
        } finally {
            isQuerying = false;
            setFormDisabled(false);
            chatFeed.scrollTop = chatFeed.scrollHeight;
        }
    });

    function setFormDisabled(disabled) {
        chatInput.disabled = disabled;
        sendBtn.disabled = disabled;
        if (disabled) {
            sendBtn.style.opacity = '0.5';
        } else {
            sendBtn.style.opacity = '1';
            chatInput.focus();
        }
    }

    // ----------------------------------------------------
    // Message Rendering Helpers
    // ----------------------------------------------------
    function appendUserMessage(text) {
        const msgDiv = document.createElement('div');
        msgDiv.classList.add('message', 'user-message');
        
        msgDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fa-solid fa-user"></i>
            </div>
            <div class="message-content">
                <p>${escapeHTML(text)}</p>
            </div>
        `;
        chatFeed.appendChild(msgDiv);
    }

    function appendTypingIndicator() {
        const msgDiv = document.createElement('div');
        msgDiv.classList.add('message', 'assistant-message', 'typing-message');
        
        msgDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fa-solid fa-robot"></i>
            </div>
            <div class="message-content">
                <div class="typing-indicator">
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                </div>
            </div>
        `;
        chatFeed.appendChild(msgDiv);
        return msgDiv;
    }

    function appendAssistantMessage(answerText, references = []) {
        const msgDiv = document.createElement('div');
        msgDiv.classList.add('message', 'assistant-message');

        // Formats newlines in response text
        const formattedAnswer = escapeHTML(answerText).replace(/\n/g, '<br>');

        let html = `
            <div class="message-avatar">
                <i class="fa-solid fa-robot"></i>
            </div>
            <div class="message-content">
                <p>${formattedAnswer}</p>
        `;

        // Render sources if available and requested
        if (references && references.length > 0) {
            const uniqueId = 'collapse-' + uuidv4();
            const showReferences = referenceToggle ? referenceToggle.checked : true;
            html += `
                <div class="sources-container" style="display: ${showReferences ? 'block' : 'none'};">
                    <div class="sources-header" onclick="toggleSources('${uniqueId}')">
                        <span><i class="fa-solid fa-circle-info"></i> View Source References (${references.length})</span>
                        <i class="fa-solid fa-chevron-down" id="arrow-${uniqueId}"></i>
                    </div>
                    <div class="sources-list" id="${uniqueId}">
            `;

            references.forEach((ref, index) => {
                let fileIcon = 'fa-file-lines';
                const fileExt = ref.source.split('.').pop().toLowerCase();
                if (fileExt === 'pdf') fileIcon = 'fa-file-pdf';
                else if (fileExt === 'docx') fileIcon = 'fa-file-word';

                html += `
                    <div class="source-item">
                        <div class="source-meta">
                            <span class="source-file"><i class="fa-solid ${fileIcon}"></i> ${escapeHTML(ref.source)} (Page ${ref.page})</span>
                            <span class="source-score">Match: ${Math.round(ref.similarity * 100)}%</span>
                        </div>
                        <div class="source-text">
                            "${escapeHTML(ref.text)}"
                        </div>
                    </div>
                `;
            });

            html += `
                    </div>
                </div>
            `;
        }

        html += `
            </div>
        `;

        msgDiv.innerHTML = html;
        chatFeed.appendChild(msgDiv);
    }

    // ----------------------------------------------------
    // Utility UI Helpers
    // ----------------------------------------------------
    
    // Toggle Source Accordions
    window.toggleSources = (id) => {
        const list = document.getElementById(id);
        const arrow = document.getElementById('arrow-' + id);
        if (list && arrow) {
            const isCurrentlyActive = list.classList.contains('active');
            if (isCurrentlyActive) {
                list.classList.remove('active');
                arrow.style.transform = 'rotate(0deg)';
            } else {
                list.classList.add('active');
                arrow.style.transform = 'rotate(180deg)';
            }
        }
    };

    // Escape raw HTML strings to avoid XSS injections
    function escapeHTML(str) {
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    // Generate random UUID for unique DOM element IDs and Session IDs
    function uuidv4() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }

    // Display simple visual popups
    function showToast(message, type = 'success') {
        toast.className = 'notification-toast';
        toast.textContent = message;
        toast.classList.add('active', type);
        
        setTimeout(() => {
            toast.classList.remove('active');
        }, 4000);
    }

    // ----------------------------------------------------
    // Auto-clear backend session data when tab/window closes
    // ----------------------------------------------------
    window.addEventListener('beforeunload', () => {
        if (!sessionId) return;

        const data = new FormData();
        data.append('session_id', sessionId);

       navigator.sendBeacon('/cleanup-session');
    });
});
