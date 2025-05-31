/**
 * Main JavaScript for AI-Powered Classroom Attendance System
 */

// Upload video file with student metadata
function uploadVideo() {
    const form = document.getElementById('uploadForm');
    const formData = new FormData(form);
    const uploadStatus = document.getElementById('uploadStatus');
    const uploadProgress = document.getElementById('uploadProgress');
    const uploadMessage = document.getElementById('uploadMessage');
    
    // Display upload status
    uploadStatus.classList.remove('d-none');
    uploadProgress.style.width = '0%';
    uploadMessage.textContent = 'Uploading video...';
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || 'Upload failed');
            });
        }
        return response.json();
    })
    .then(data => {
        uploadProgress.style.width = '100%';
        uploadProgress.classList.remove('bg-danger');
        uploadProgress.classList.add('bg-success');
        uploadMessage.textContent = 'Video uploaded successfully!';
        
        // Reset form after successful upload
        setTimeout(() => {
            form.reset();
            location.reload();
        }, 1000);
    })
    .catch(error => {
        uploadProgress.style.width = '100%';
        uploadProgress.classList.remove('bg-success');
        uploadProgress.classList.add('bg-danger');
        uploadMessage.textContent = `Error: ${error.message}`;
    });
}

// Create a dataset for a specific department and section
function createDataset() {
    const form = document.getElementById('datasetForm');
    const formData = new FormData(form);
    const datasetStatus = document.getElementById('datasetStatus');
    const datasetJobId = document.getElementById('datasetJobId');
    const datasetJobStatus = document.getElementById('datasetJobStatus');
    const statusLink = document.getElementById('statusLink');
    
    // Get department and section from form
    const department = formData.get('department');
    const section = formData.get('section');
    
    fetch('/create-dataset', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || 'Dataset creation failed');
            });
        }
        return response.json();
    })
    .then(data => {
        datasetStatus.classList.remove('d-none');
        datasetJobId.textContent = data.job_id;
        datasetJobStatus.textContent = 'Pending';
        statusLink.href = data.status_url;
        
        // Set up interval to check job status and enable train button when completed
        const checkInterval = setInterval(() => {
            fetch(`/status/${data.job_id}/json`)
                .then(response => response.json())
                .then(statusData => {
                    if (statusData.status === 'completed') {
                        clearInterval(checkInterval);
                        console.log('Dataset creation completed, updating UI');
                        
                        // Enable the corresponding train button
                        const trainButton = document.getElementById(`train-btn-${department}-${section}`);
                        if (trainButton) {
                            trainButton.disabled = false;
                            trainButton.title = "Train model for this section";
                        } else {
                            // Button may not be on the same page, reload department list
                            loadDepartmentList();
                        }
                    } else if (statusData.status === 'failed') {
                        clearInterval(checkInterval);
                        console.error('Dataset creation failed:', statusData.error_message);
                    }
                })
                .catch(error => {
                    console.error('Error checking job status:', error);
                    clearInterval(checkInterval);
                });
        }, 5000);
        
        // Reset form after successful submission
        form.reset();
    })
    .catch(error => {
        alert(`Error: ${error.message}`);
    });
}

// Load list of departments and sections
function loadDepartmentList() {
    const departmentList = document.getElementById('departmentList');
    const departmentDropdown = document.getElementById('datasetDepartment');
    const sectionDropdown = document.getElementById('datasetSection');
    
    fetch('/departments')
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to load departments');
        }
        return response.json();
    })
    .then(data => {
        // Clear loading indicator
        departmentList.innerHTML = '';
        
        if (data.length === 0) {
            departmentList.innerHTML = `
                <div class="alert alert-info">
                    <i data-feather="info" class="me-1"></i> No departments or students found. 
                    Upload videos and create datasets first.
                </div>
            `;
            feather.replace();
            return;
        }
        
        // Populate department dropdown
        if (departmentDropdown) {
            // Clear existing options (except the first one)
            const firstOption = departmentDropdown.options[0];
            departmentDropdown.innerHTML = '';
            departmentDropdown.appendChild(firstOption);
            
            // Add departments to dropdown
            data.forEach(dept => {
                const option = document.createElement('option');
                option.value = dept.department;
                option.textContent = dept.department;
                departmentDropdown.appendChild(option);
            });
            
            // When department changes, update sections dropdown
            departmentDropdown.addEventListener('change', function() {
                const selectedDept = this.value;
                const deptData = data.find(d => d.department === selectedDept);
                
                // Clear section dropdown (except the first option)
                const firstSectionOption = sectionDropdown.options[0];
                sectionDropdown.innerHTML = '';
                sectionDropdown.appendChild(firstSectionOption);
                
                // Add sections to dropdown
                if (deptData) {
                    deptData.sections.forEach(section => {
                        const option = document.createElement('option');
                        option.value = section.section;
                        option.textContent = `Section ${section.section} (${section.student_count} students)`;
                        sectionDropdown.appendChild(option);
                    });
                }
            });
        }
        
        // Create department cards for training section
        data.forEach(dept => {
            const deptCard = document.createElement('div');
            deptCard.className = 'card mb-3';
            
            const deptHeader = document.createElement('div');
            deptHeader.className = 'card-header';
            deptHeader.innerHTML = `<h5>${dept.department}</h5>`;
            deptCard.appendChild(deptHeader);
            
            const deptBody = document.createElement('div');
            deptBody.className = 'card-body';
            
            // Create section buttons
            dept.sections.forEach(section => {
                const sectionRow = document.createElement('div');
                sectionRow.className = 'd-flex justify-content-between align-items-center mb-3';
                
                // Create inner div for content
                const sectionContent = document.createElement('div');
                sectionContent.className = 'w-100';
                sectionContent.dataset.department = dept.department;
                sectionContent.dataset.section = section.section;
                
                // Add section info
                const sectionInfo = document.createElement('div');
                sectionInfo.className = 'd-flex justify-content-between align-items-center mb-2';
                
                // First create the train button with disabled state by default
                const trainButton = document.createElement('button');
                trainButton.className = 'btn btn-warning train-model-btn';
                trainButton.dataset.department = dept.department;
                trainButton.dataset.section = section.section;
                trainButton.innerHTML = '<i data-feather="cpu" class="me-1"></i> Train Model';
                trainButton.disabled = true; // Disabled by default
                trainButton.id = `train-btn-${dept.department}-${section.section}`;
                
                // Add HTML content to sectionInfo
                sectionInfo.innerHTML = `
                    <div>
                        <span class="fw-bold">Section ${section.section}</span>
                        <span class="badge bg-info ms-2">${section.student_count} students</span>
                    </div>
                `;
                
                // Append the button to sectionInfo
                sectionInfo.appendChild(trainButton);
                sectionContent.appendChild(sectionInfo);
                
                // Add model buttons container - initially hidden
                const modelActions = document.createElement('div');
                modelActions.className = 'model-actions d-none mt-2 d-flex gap-2 justify-content-end';
                modelActions.id = `model-actions-${dept.department}-${section.section}`;
                sectionContent.appendChild(modelActions);
                
                // Check if dataset exists for this department/section
                checkDatasetExists(dept.department, section.section, trainButton);
                
                // Fetch model info for this department/section
                fetch(`/model-info/${dept.department}/${section.section}`)
                    .then(response => {
                        if (!response.ok) {
                            if (response.status === 404) {
                                console.log(`No model found for ${dept.department}/${section.section}`);
                                return null;
                            }
                            throw new Error(`Error fetching model info: ${response.statusText}`);
                        }
                        return response.json();
                    })
                    .then(modelData => {
                        if (modelData && modelData.exists) {
                            // Show model actions if model exists
                            modelActions.classList.remove('d-none');
                            
                            // Add .pt download button if available
                            if (modelData.has_model) {
                                const ptButton = document.createElement('a');
                                ptButton.href = modelData.download_model_url;
                                ptButton.className = 'btn btn-sm btn-primary';
                                ptButton.innerHTML = '<i data-feather="download" class="me-1"></i> Download .pt';
                                modelActions.appendChild(ptButton);
                            }
                            
                            // Add TFLite buttons - either download or export
                            if (modelData.has_tflite) {
                                const tfliteButton = document.createElement('a');
                                tfliteButton.href = modelData.download_tflite_url;
                                tfliteButton.className = 'btn btn-sm btn-success';
                                tfliteButton.innerHTML = '<i data-feather="download" class="me-1"></i> Download TFLite';
                                modelActions.appendChild(tfliteButton);
                            } else {
                                const exportButton = document.createElement('button');
                                exportButton.className = 'btn btn-sm btn-info export-model-btn';
                                exportButton.dataset.modelId = modelData.model_id;
                                exportButton.innerHTML = '<i data-feather="package" class="me-1"></i> Export to TFLite';
                                exportButton.addEventListener('click', function() {
                                    exportButton.disabled = true;
                                    exportModel(modelData.model_id);
                                    
                                });
                                modelActions.appendChild(exportButton);
                            }
                            
                            // Initialize feather icons
                            feather.replace();
                        }
                    })
                    .catch(error => {
                        console.error(`Error fetching model info for ${dept.department}/${section.section}:`, error);
                    });
                
                sectionRow.appendChild(sectionContent);
                deptBody.appendChild(sectionRow);
            });
            
            deptCard.appendChild(deptBody);
            departmentList.appendChild(deptCard);
        });
        
        // Initialize feather icons
        feather.replace();
        
        // Add event listeners to train model buttons
        document.querySelectorAll('.train-model-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const department = this.dataset.department;
                const section = this.dataset.section;
                document.getElementById(`train-btn-${department}-${section}`).disabled =true;
                trainModel(department, section);
            });
        });
    })
    .catch(error => {
        departmentList.innerHTML = `
            <div class="alert alert-danger">
                <i data-feather="alert-triangle" class="me-1"></i> ${error.message}
            </div>
        `;
        feather.replace();
    });
}
function checkDatasetExists(department, section, buttonElement) {
    console.log(`Checking dataset for ${department}_${section}`);
    
    fetch(`/check-dataset/${department}/${section}`)
        .then(response => {
            console.log('Response status:', response.status);
            if (!response.ok) {
                if (response.status === 404) {
                    console.log('Dataset not found (404)');
                    buttonElement.disabled = true;
                    buttonElement.title = "No dataset available. Create a dataset first.";
                    return false;
                }
                throw new Error(`Error checking dataset: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Dataset check response:', data);
            
            if (data && data.exists) {
                console.log('Dataset exists, enabling button');
                buttonElement.disabled = false;
                buttonElement.title = "Train model for this section";
            } else {
                console.log('Dataset does not exist according to response');
                buttonElement.disabled = true;
                buttonElement.title = "No dataset available. Create a dataset first.";
            }
        })
        .catch(error => {
            console.error(`Error checking dataset for ${department}/${section}:`, error);
            buttonElement.disabled = true;
            buttonElement.title = "Error checking dataset availability";
        });
}

// Train a model for a specific classroom
function trainModel(department, section) {
    console.log(`Training model for ${department}/${section}`);
    const trainingStatus = document.getElementById('trainingStatus');
    const trainingJobId = document.getElementById('trainingJobId');
    const trainingJobStatus = document.getElementById('trainingJobStatus');
    const trainingStatusLink = document.getElementById('trainingStatusLink');
    
    fetch(`/train/${department}/${section}`, {
        method: 'POST'
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || 'Model training failed');
            });
        }
        return response.json();
    })
    .then(data => {
        console.log('Training job started:', data);
        trainingStatus.classList.remove('d-none');
        trainingJobId.textContent = data.job_id;
        trainingJobStatus.textContent = 'Pending';
        trainingStatusLink.href = data.status_url;
        
        // Store department and section in data attributes for later refresh
        trainingStatus.dataset.department = department;
        trainingStatus.dataset.section = section;
        
        // Create or update model actions container if it doesn't exist
        let modelActionsContainer = document.getElementById(`model-actions-${department}-${section}`);
        if (!modelActionsContainer) {
            const sectionContainer = document.querySelector(`[data-department="${department}"][data-section="${section}"]`);
            if (sectionContainer) {
                modelActionsContainer = document.createElement('div');
                modelActionsContainer.className = 'model-actions d-none mt-2 d-flex gap-2 justify-content-end';
                modelActionsContainer.id = `model-actions-${department}-${section}`;
                sectionContainer.appendChild(modelActionsContainer);
            }
        }
        
        // Set an interval to check if the job is complete and update model actions
        const checkInterval = setInterval(() => {
            fetch(`/status/${data.job_id}/json`)
                .then(response => response.json())
                .then(statusData => {
                    // Update status badge
                    trainingJobStatus.textContent = statusData.status;
                    
                    if (statusData.status === 'completed') {
                        clearInterval(checkInterval);
                        console.log('Training completed, updating UI');
                        
                        // Refresh the model actions for this department/section
                        updateModelActions(department, section);
                        
                        // Also reload models list if on models page
                        if (document.getElementById('modelsContainer')) {
                            loadModels();
                        }
                        
                        // Update department list to show proper model actions
                        loadDepartmentList();
                    } else if (statusData.status === 'failed') {
                        clearInterval(checkInterval);
                        console.error('Training failed:', statusData.error_message);
                    }
                })
                .catch(error => {
                    console.error('Error checking job status:', error);
                    clearInterval(checkInterval);
                });
        }, 5000);
    })
    .catch(error => {
        alert(`Error: ${error.message}`);
    });
}

// Load list of available models
function loadModels() {
    const modelsContainer = document.getElementById('modelsContainer');
    
    fetch('/models/json')
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to load models');
        }
        return response.json();
    })
    .then(data => {
        // Clear loading indicator
        modelsContainer.innerHTML = '';
        
        if (data.length === 0) {
            modelsContainer.innerHTML = `
                <div class="alert alert-info">
                    <i data-feather="info" class="me-1"></i> No models available yet. 
                    Train models from the home page first.
                </div>
            `;
            feather.replace();
            return;
        }
        
        // Create model table
        const table = document.createElement('table');
        table.className = 'table table-striped';
        table.innerHTML = `
            <thead>
                <tr>
                    <th>Name</th>
                    <th>Department</th>
                    <th>Section</th>
                    <th>Created</th>
                    <th>TFLite Status</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody id="modelTableBody"></tbody>
        `;
        
        const tableBody = table.querySelector('tbody');
        
        // Add model rows
        data.forEach(model => {
            const row = document.createElement('tr');
            
            const createdDate = new Date(model.created_at);
            const formattedDate = createdDate.toLocaleDateString() + ' ' + 
                                  createdDate.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            
            row.innerHTML = `
                <td>${model.name}</td>
                <td>${model.department}</td>
                <td>${model.section}</td>
                <td>${formattedDate}</td>
                <td>
                    ${model.has_tflite 
                        ? '<span class="badge bg-success">Available</span>' 
                        : '<span class="badge bg-secondary">Not exported</span>'}
                </td>
                <td class="d-flex gap-2">
                    ${model.has_model 
                        ? `<a href="${model.download_model_url}" class="btn btn-sm btn-primary">
                             <i data-feather="download" class="me-1"></i> Download .pt
                           </a>` 
                        : ''}
                    
                    ${model.has_tflite 
                        ? `<a href="${model.download_tflite_url}" class="btn btn-sm btn-success">
                             <i data-feather="download" class="me-1"></i> Download TFLite
                           </a>` 
                        : `<button class="btn btn-sm btn-info export-model-btn" data-model-id="${model.id}">
                             <i data-feather="package" class="me-1"></i> Export to TFLite
                           </button>`}
                </td>
            `;
            
            tableBody.appendChild(row);
        });
        
        modelsContainer.appendChild(table);
        
        // Initialize feather icons
        feather.replace();
        
        // Add event listeners to export buttons
        document.querySelectorAll('.export-model-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const modelId = this.dataset.modelId;
                exportModel(modelId);
            });
        });
    })
    .catch(error => {
        modelsContainer.innerHTML = `
            <div class="alert alert-danger">
                <i data-feather="alert-triangle" class="me-1"></i> ${error.message}
            </div>
        `;
        feather.replace();
    });
}

// Export a model to TFLite format
// 
// Export a model to TFLite format
function exportModel(modelId) {
    const exportStatus = document.getElementById('exportStatus');
    const exportJobId = document.getElementById('exportJobId');
    const exportJobStatus = document.getElementById('exportJobStatus');
    const exportStatusLink = document.getElementById('exportStatusLink');
   
    // Find and disable the clicked button
    const clickedButton = document.querySelector(`.export-model-btn[data-model-id="${modelId}"]`);
    if (clickedButton) {
        clickedButton.disabled = true;
        clickedButton.innerHTML = '<i data-feather="loader" class="me-1"></i> Exporting...';
        if (window.feather) {
            feather.replace();
        }
    }
    
    // First get model information to know department and section
    fetch(`/models/json`)
        .then(response => response.json())
        .then(models => {
            const model = models.find(m => m.id === parseInt(modelId));

            if (!model) {
                throw new Error('Model not found');
            }
            
            // Store department and section for later refresh
            const department = model.department;
            const section = model.section;
            
            // Now proceed with export
            return fetch(`/export/${modelId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(data => {
                        throw new Error(data.error || 'Model export failed');
                    });
                }
                return response.json();
            })
            .then(data => {
                // Show export status panel if it exists
                if (exportStatus) {
                    exportStatus.classList.remove('d-none');
                    exportJobId.textContent = data.job_id;
                    exportJobStatus.textContent = 'Pending';
                    exportStatusLink.href = data.status_url;
                    // Set department and section for later reference
                    exportStatus.dataset.department = department;
                    exportStatus.dataset.section = section;
                }
                
                // Set an interval to check if the job is complete and update model actions
                const checkInterval = setInterval(() => {
                    fetch(`/status/${data.job_id}/json`)
                        .then(response => response.json())
                        .then(statusData => {
                            // Update status panel if it exists
                            if (exportJobStatus) {
                                exportJobStatus.textContent = statusData.status;
                            }
                            
                            if (statusData.status === 'completed') {
                                clearInterval(checkInterval);
                                
                                // Reload the page to show updated buttons
                                location.reload();
                                
                                // Refresh the model actions for this department/section
                                updateModelActions(department, section);
                                
                                // Also reload models list if on models page
                                if (document.getElementById('modelsContainer')) {
                                    loadModels();
                                }
                            } else if (statusData.status === 'failed') {
                                clearInterval(checkInterval);
                                
                                // Re-enable the button if the export failed
                                if (clickedButton) {
                                    clickedButton.disabled = false;
                                    clickedButton.innerHTML = '<i data-feather="package" class="me-1"></i> Export to TFLite';
                                    if (window.feather) {
                                        feather.replace();
                                    }
                                }
                                
                                // Show error message
                                alert('Export failed: ' + (statusData.error_message || 'Unknown error'));
                            }
                        })
                        .catch(error => {
                            console.error('Error checking job status:', error);
                            clearInterval(checkInterval);
                            
                            // Re-enable the button if there was an error
                            if (clickedButton) {
                                clickedButton.disabled = false;
                                clickedButton.innerHTML = '<i data-feather="package" class="me-1"></i> Export to TFLite';
                                if (window.feather) {
                                    feather.replace();
                                }
                            }
                        });
                }, 5000);
                return data;
            });
        })
        .catch(error => {
            console.error('Error exporting model:', error);
            alert(`Error: ${error.message}`);
            
            // Re-enable the button if there was an error
            if (clickedButton) {
                clickedButton.disabled = false;
                clickedButton.innerHTML = '<i data-feather="package" class="me-1"></i> Export to TFLite';
                if (window.feather) {
                    feather.replace();
                }
            }
        });
}

// Update model actions for a specific department and section
function updateModelActions(department, section) {
    console.log(`Updating model actions for ${department}/${section}`);
    const modelActionsContainer = document.getElementById(`model-actions-${department}-${section}`);
    
    if (!modelActionsContainer) {
        console.error(`Model actions container for ${department}/${section} not found`);
        return;
    }
    
    // Clear existing content
    modelActionsContainer.innerHTML = '';
    modelActionsContainer.classList.remove('d-none');
    
    // Fetch latest model info
    fetch(`/model-info/${department}/${section}`)
        .then(response => {
            if (!response.ok) {
                if (response.status === 404) {
                    console.log(`No model found for ${department}/${section}`);
                    // Model doesn't exist yet, hide the container
                    modelActionsContainer.classList.add('d-none');
                    return null;
                }
                throw new Error('Failed to get model info');
            }
            return response.json();
        })
        .then(modelData => {
            if (!modelData) return;
            
            console.log('Model data:', modelData);
            
            // Add .pt download button if available
            if (modelData.has_model) {
                const ptButton = document.createElement('a');
                ptButton.href = modelData.download_model_url;
                ptButton.className = 'btn btn-sm btn-primary';
                ptButton.innerHTML = '<i data-feather="download" class="me-1"></i> Download .pt';
                modelActionsContainer.appendChild(ptButton);
                
                // Always add space between buttons
                const spacer = document.createElement('span');
                spacer.className = 'ms-2';
                modelActionsContainer.appendChild(spacer);
            }
            
            // Add TFLite buttons - either download or export
            if (modelData.has_tflite) {
                const tfliteButton = document.createElement('a');
                tfliteButton.href = modelData.download_tflite_url;
                tfliteButton.className = 'btn btn-sm btn-success';
                tfliteButton.innerHTML = '<i data-feather="download" class="me-1"></i> Download TFLite';
                modelActionsContainer.appendChild(tfliteButton);
            } else {
                const exportButton = document.createElement('button');
                exportButton.className = 'btn btn-sm btn-info export-model-btn';
                exportButton.dataset.modelId = modelData.model_id;
                exportButton.innerHTML = '<i data-feather="package" class="me-1"></i> Export to TFLite';
                exportButton.addEventListener('click', function() {
                    exportButton.disabled = true;
                    exportModel(modelData.model_id);
                });
                modelActionsContainer.appendChild(exportButton);
            }
            
            // Initialize feather icons
            feather.replace();
        })
        .catch(error => {
            console.error(`Error updating model actions for ${department}/${section}:`, error);
            modelActionsContainer.innerHTML = `<div class="alert alert-danger small">Error: ${error.message}</div>`;
        });
}

// Refresh job status information
function refreshJobStatus(jobId) {
    const jobStatus = document.getElementById('jobStatus');
    const resultContainer = document.getElementById('result-container');
    const resultContent = document.getElementById('result-content');
    
    fetch(`/status/${jobId}/json`)
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to get job status');
        }
        return response.json();
    })
    .then(data => {
        // Update status badge
        jobStatus.textContent = data.status;
        jobStatus.className = 'badge';
        
        if (data.status === 'pending') {
            jobStatus.classList.add('bg-secondary');
        } else if (data.status === 'processing') {
            jobStatus.classList.add('bg-primary');
        } else if (data.status === 'completed') {
            jobStatus.classList.add('bg-success');
            // Show result information if available
            if (data.result) {
                resultContainer.classList.remove('d-none');
                
                let resultHTML = '<div class="table-responsive"><table class="table">';
                
                if (data.job_type === 'dataset_creation') {
                    resultHTML += `
                        <tr><th>Dataset ID</th><td>${data.result.dataset_id}</td></tr>
                        <tr><th>Name</th><td>${data.result.name}</td></tr>
                        <tr><th>Department</th><td>${data.result.department}</td></tr>
                        <tr><th>Section</th><td>${data.result.section}</td></tr>
                        <tr><th>Students</th><td>${data.result.num_students}</td></tr>
                    `;
                } else if (data.job_type === 'model_training') {
                    resultHTML += `
                        <tr><th>Model ID</th><td>${data.result.model_id}</td></tr>
                        <tr><th>Name</th><td>${data.result.name}</td></tr>
                        <tr><th>Department</th><td>${data.result.department}</td></tr>
                        <tr><th>Section</th><td>${data.result.section}</td></tr>
                        <tr><th>Actions</th><td class="d-flex gap-2">
                            ${data.result.has_model 
                                ? `<a href="${data.result.download_model_url}" class="btn btn-sm btn-primary">
                                     <i data-feather="download" class="me-1"></i> Download .pt
                                   </a>` 
                                : ''}
                            
                            ${data.result.has_tflite 
                                ? `<a href="${data.result.download_tflite_url}" class="btn btn-sm btn-success">
                                     <i data-feather="download" class="me-1"></i> Download TFLite
                                   </a>` 
                                : `<a href="${data.result.export_url}" class="btn btn-sm btn-info">
                                     <i data-feather="package" class="me-1"></i> Export to TFLite
                                   </a>`}
                        </td></tr>
                    `;
                }
                
                resultHTML += '</table></div>';
                resultContent.innerHTML = resultHTML;
                
                // Initialize feather icons
                feather.replace();
                
            }
            
        } else {
            jobStatus.classList.add('bg-danger');
        }
    })
    .catch(error => {
        console.error('Error refreshing job status:', error);
    });
}
