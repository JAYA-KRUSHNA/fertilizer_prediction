// Main JavaScript file for AgriFert Predict application

document.addEventListener('DOMContentLoaded', function () {
    // Initialize all components
    initializeNavigation();
    initializeForms();
    initializeAnimations();

    // Check if we are on the visualization page and load data
    if (document.body.querySelector('.analytics-section')) {
        loadAnalyticsData();
        // Initialize Chart.js placeholder logic once Chart.js is loaded in visualization.html
    }
});

// Navigation functionality
function initializeNavigation() {
    // Mobile menu toggle
    const hamburger = document.querySelector('.nav-hamburger');
    const navMenu = document.querySelector('.nav-menu');

    if (hamburger) {
        hamburger.addEventListener('click', function () {
            hamburger.classList.toggle('active');
            navMenu.classList.toggle('active');
        });
    }

    // Close mobile menu when clicking a link
    const navLinks = document.querySelectorAll('.nav-menu a');
    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            if (hamburger) hamburger.classList.remove('active');
            if (navMenu) navMenu.classList.remove('active');
        });
    });

    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });

    // Update active nav link based on current page
    updateActiveNavLink();
}

function updateActiveNavLink() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-menu a');

    navLinks.forEach(link => {
        link.classList.remove('active');
        const linkPath = new URL(link.href).pathname;
        if (currentPath === linkPath || (currentPath === '/' && linkPath === '/')) {
            link.classList.add('active');
        }
    });
}

// Form functionality
function initializeForms() {
    const predictForm = document.getElementById('predict-form');
    if (predictForm) {
        predictForm.addEventListener('submit', handlePrediction);
    }

    // Real-time form validation
    const inputs = document.querySelectorAll('input, select');
    inputs.forEach(input => {
        input.addEventListener('blur', validateField);
        input.addEventListener('input', validateField);
    });

    // Auto-fill demo data button
    const demoButton = document.getElementById('demo-data');
    if (demoButton) {
        demoButton.addEventListener('click', fillDemoData);
    }
}

function validateField(e) {
    // ... (Original Validation code) ...
    // Note: The backend has more robust validation, but client-side remains useful
    const field = e.target;
    const value = field.value.trim();
    const fieldName = field.name;

    // Remove existing error messages
    const existingError = field.parentNode.querySelector('.error-message');
    if (existingError) {
        existingError.remove();
    }

    let isValid = true;
    let errorMessage = '';

    // Validation rules (Same as original but condensed for snippet)
    switch (fieldName) {
        case 'soil_ph':
            const ph = parseFloat(value);
            if (isNaN(ph) || ph < 4.0 || ph > 9.0) {
                isValid = false;
                errorMessage = 'Soil pH must be between 4.0 and 9.0';
            }
            break;
        case 'nitrogen':
        case 'phosphorus':
        case 'potassium':
            const nutrient = parseFloat(value);
            if (isNaN(nutrient) || nutrient < 0 || nutrient > 200) {
                isValid = false;
                errorMessage = 'Nutrient levels must be between 0 and 200 kg/ha';
            }
            break;
        case 'organic_matter':
            const om = parseFloat(value);
            if (isNaN(om) || om < 0 || om > 10) {
                isValid = false;
                errorMessage = 'Organic matter must be between 0 and 10%';
            }
            break;
        case 'temperature':
            const temp = parseFloat(value);
            if (isNaN(temp) || temp < 10 || temp > 40) {
                isValid = false;
                errorMessage = 'Temperature must be between 10°C and 40°C';
            }
            break;
        case 'humidity':
            const hum = parseFloat(value);
            if (isNaN(hum) || hum < 20 || hum > 90) {
                isValid = false;
                errorMessage = 'Humidity must be between 20% and 90%';
            }
            break;
        case 'rainfall':
            const rain = parseFloat(value);
            if (isNaN(rain) || rain < 200 || rain > 2000) {
                isValid = false;
                errorMessage = 'Rainfall must be between 200 and 2000 mm';
            }
            break;
    }

    // Visual feedback
    if (!isValid && value !== '') {
        field.style.borderColor = '#e74c3c';
        field.classList.add('error');

        // Add error message below field
        const errorSpan = document.createElement('span');
        errorSpan.className = 'error-message';
        errorSpan.textContent = errorMessage;
        errorSpan.style.color = '#e74c3c';
        errorSpan.style.fontSize = '0.85rem';
        errorSpan.style.marginTop = '0.25rem';
        errorSpan.style.display = 'block';
        field.parentNode.appendChild(errorSpan);
    } else if (value !== '') {
        field.style.borderColor = '#4caf50';
        field.classList.remove('error');
        field.classList.add('valid');
    }

    return isValid;
}


function fillDemoData() {
    // Demo data with realistic agricultural values
    const demoValues = {
        soil_ph: 6.5,
        soil_texture: 'Loamy',
        nitrogen: 45.0,
        phosphorus: 30.0,
        potassium: 40.0,
        organic_matter: 3.5,
        crop_type: 'Wheat',
        irrigation_type: 'Drip',
        temperature: 25.0,
        humidity: 65.0,
        rainfall: 850
    };

    // Fill each form field
    Object.keys(demoValues).forEach(key => {
        const field = document.getElementById(key) || document.querySelector(`[name="${key}"]`);
        if (field) {
            field.value = demoValues[key];
            // Trigger validation visual feedback
            field.style.borderColor = '#4caf50';
            field.classList.add('valid');

            // Add animation effect
            field.style.transition = 'all 0.3s ease';
            field.style.transform = 'scale(1.02)';
            setTimeout(() => {
                field.style.transform = 'scale(1)';
            }, 200);
        }
    });

    // Show success notification
    showToast('Demo data filled successfully!', 'success');
}


/**
 * @function handlePrediction
 * Handles form submission, calls the Flask API, and redirects to results page.
 * IMPORTANT: Corrected redirection to pass the entire result object.
 */
async function handlePrediction(e) {
    e.preventDefault();

    const form = e.target;
    const submitButton = form.querySelector('button[type="submit"]');
    const originalText = submitButton.innerHTML; // Get HTML content to preserve icon

    // Validate all fields
    const inputs = form.querySelectorAll('input, select');
    let isFormValid = true;

    inputs.forEach(input => {
        // Only validate if it has a value, otherwise browser handles 'required'
        if (input.value && !validateField({ target: input })) {
            isFormValid = false;
        }
    });

    if (!isFormValid) {
        alert('Please correct the errors in the form before submitting.');
        return;
    }

    // Show loading state
    submitButton.disabled = true;
    submitButton.innerHTML = '<span class="loading"></span> Processing...';

    try {
        // Collect form data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        // Send prediction request
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        if (response.ok) {
            const result = await response.json();
            // Redirect to results page, passing the full result object as a URL parameter
            const resultUrl = `/results?data=${encodeURIComponent(JSON.stringify(result))}`;
            window.location.href = resultUrl;
        } else {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Prediction failed with an unknown server error.');
        }

    } catch (error) {
        console.error('Error:', error);
        alert(`An error occurred: ${error.message}. Please check console for details.`);
    } finally {
        // Reset button state
        submitButton.disabled = false;
        submitButton.innerHTML = originalText;
    }
}


// Animation functionality
function initializeAnimations() {
    // Add fade-in animation on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in-visible');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe elements for animation
    const animatedElements = document.querySelectorAll(
        '.feature-card, .stat-card, .step, .info-card, .recommendation-card, .tech-card, .team-member, .insight-card'
    );

    animatedElements.forEach(el => {
        el.classList.add('fade-in');
        observer.observe(el);
    });

    // Add hover effects to buttons
    document.querySelectorAll('.btn-primary, .btn-secondary, .cta-button').forEach(btn => {
        btn.addEventListener('mouseenter', function () {
            this.style.transform = 'translateY(-3px) scale(1.02)';
        });
        btn.addEventListener('mouseleave', function () {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });
}

// Toast notification function
function showToast(message, type = 'info') {
    // Remove existing toast
    const existingToast = document.querySelector('.toast-notification');
    if (existingToast) existingToast.remove();

    const toast = document.createElement('div');
    toast.className = `toast-notification toast-${type}`;
    toast.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
        <span>${message}</span>
    `;

    document.body.appendChild(toast);

    // Animate in
    setTimeout(() => toast.classList.add('show'), 10);

    // Remove after 3 seconds
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}


// --- ANALYTICS AND VISUALIZATION FUNCTIONS ---

// Chart storage for updating/destroying
let charts = {};

async function loadAnalyticsData() {
    const spinner = document.getElementById('predictions-tbody');
    if (spinner) {
        spinner.innerHTML = '<tr><td colspan="7" class="loading-cell"><span class="loading"></span> Loading analytics data...</td></tr>';
    }

    try {
        const response = await fetch('/api/analytics');
        if (!response.ok) {
            throw new Error('Failed to fetch analytics data from server.');
        }
        const data = await response.json();

        // 1. Render Model Performance Charts and Table
        renderModelCharts(data.model_performance);
        renderModelTable(data.model_performance);

        // 2. Render Recent Predictions Table
        renderPredictionHistory(data.recent_predictions);

        // 3. Render General Overview Charts (using data.statistics and hardcoded distributions)
        renderOverviewCharts(data.statistics, data.model_performance);

    } catch (error) {
        console.error('Error loading analytics:', error);
        if (spinner) {
            spinner.innerHTML = '<tr><td colspan="7" class="error-cell">Failed to load data. Please check the backend console.</td></tr>';
        }
    }
}

function renderOverviewCharts(stats, performanceData, recentPredictions) {
    const cropStats = stats.crop_statistics;
    const labels = cropStats.map(c => c.crop_type);

    // --- Crop Distribution Chart ---
    const cropCtx = document.getElementById('crop-distribution-chart');
    if (cropCtx) {
        if (charts.cropDistribution) charts.cropDistribution.destroy();
        charts.cropDistribution = new Chart(cropCtx.getContext('2d'), {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: cropStats.map(c => c.count),
                    backgroundColor: ['#4caf50', '#2196f3', '#ff9800', '#e91e63', '#9c27b0']
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { position: 'bottom' } }
            }
        });
    }

    // --- Fertilizer by Crop Chart ---
    const fertilizerCtx = document.getElementById('fertilizer-by-crop-chart');
    if (fertilizerCtx) {
        if (charts.fertilizerByCrop) charts.fertilizerByCrop.destroy();
        charts.fertilizerByCrop = new Chart(fertilizerCtx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Avg Fertilizer (kg/ha)',
                    data: cropStats.map(c => c.avg_fertilizer),
                    backgroundColor: '#4caf50'
                }]
            },
            options: {
                responsive: true,
                scales: { y: { beginAtZero: true } }
            }
        });
    }

    // --- Update Key Stats using correct IDs ---
    const overall = stats.overall_statistics;
    const totalSamplesEl = document.getElementById('total-samples');
    const cropTypesEl = document.getElementById('crop-types');
    const avgFertilizerStatEl = document.getElementById('avg-fertilizer-stat');
    const avgTempEl = document.getElementById('avg-temperature');

    if (totalSamplesEl) totalSamplesEl.textContent = (overall.total_samples || 0).toLocaleString();
    if (cropTypesEl) cropTypesEl.textContent = cropStats.length;
    if (avgFertilizerStatEl) avgFertilizerStatEl.textContent = `${(overall.avg_fertilizer_req || 0).toFixed(1)}`;
    if (avgTempEl) avgTempEl.textContent = `${(overall.avg_temperature || 0).toFixed(1)}°C`;

    // --- Soil pH Distribution Chart (placeholder with sample data) ---
    const soilPhCtx = document.getElementById('soil-ph-distribution-chart');
    if (soilPhCtx) {
        if (charts.soilPh) charts.soilPh.destroy();
        charts.soilPh = new Chart(soilPhCtx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: ['< 5.5', '5.5-6.0', '6.0-6.5', '6.5-7.0', '7.0-7.5', '> 7.5'],
                datasets: [{
                    label: 'Sample Count',
                    data: [50, 120, 280, 350, 150, 50],
                    backgroundColor: '#2196f3'
                }]
            },
            options: {
                responsive: true,
                scales: { y: { beginAtZero: true } },
                plugins: { legend: { display: false } }
            }
        });
    }

    // --- Climate Factors Radar Chart ---
    const climateCtx = document.getElementById('climate-factors-chart');
    if (climateCtx) {
        if (charts.climate) charts.climate.destroy();
        charts.climate = new Chart(climateCtx.getContext('2d'), {
            type: 'radar',
            data: {
                labels: ['Temperature', 'Humidity', 'Rainfall', 'Soil pH', 'Organic Matter'],
                datasets: [{
                    label: 'Normalized Averages',
                    data: [
                        overall.avg_temperature ? overall.avg_temperature / 40 * 100 : 60,
                        65, // humidity placeholder
                        overall.avg_rainfall ? overall.avg_rainfall / 2000 * 100 : 55,
                        overall.avg_soil_ph ? overall.avg_soil_ph / 14 * 100 : 46,
                        45 // organic matter placeholder
                    ],
                    backgroundColor: 'rgba(76, 175, 80, 0.2)',
                    borderColor: '#4caf50',
                    pointBackgroundColor: '#4caf50'
                }]
            },
            options: {
                responsive: true,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    }

    // --- Feature Importance Chart (from recent predictions) ---
    renderFeatureImportanceChart(recentPredictions);
}

function renderFeatureImportanceChart(recentPredictions) {
    const featureCtx = document.getElementById('feature-importance-chart');
    if (!featureCtx) return;

    // Try to get feature importance from the most recent prediction
    let featureImportance = null;
    if (recentPredictions && recentPredictions.length > 0) {
        featureImportance = recentPredictions[0].feature_importance;
    }

    // Fallback data if no predictions available
    if (!featureImportance || featureImportance.length === 0) {
        featureImportance = [
            { name: 'Crop Type', importance: 23.3 },
            { name: 'Soil pH', importance: 22.7 },
            { name: 'Humidity', importance: 11.4 },
            { name: 'Organic Matter', importance: 10.4 },
            { name: 'Temperature', importance: 7.3 },
            { name: 'Nitrogen', importance: 6.2 },
            { name: 'Potassium', importance: 5.6 },
            { name: 'Phosphorus', importance: 5.3 }
        ];
    }

    // Convert to percentage if needed (values > 1 are already percentages)
    const data = featureImportance.map(f => {
        const val = f.importance < 1 ? f.importance * 100 : f.importance;
        return { name: f.name, value: val };
    }).sort((a, b) => b.value - a.value);

    if (charts.featureImportance) charts.featureImportance.destroy();

    charts.featureImportance = new Chart(featureCtx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: data.map(f => f.name),
            datasets: [{
                label: 'Importance (%)',
                data: data.map(f => f.value.toFixed(1)),
                backgroundColor: [
                    '#4caf50', '#2196f3', '#ff9800', '#e91e63',
                    '#9c27b0', '#00bcd4', '#ff5722', '#795548',
                    '#607d8b', '#3f51b5', '#cddc39'
                ]
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { beginAtZero: true, max: 30 }
            }
        }
    });

    // --- Correlation Matrix Chart (placeholder) ---
    const correlationCtx = document.getElementById('correlation-matrix-chart');
    if (correlationCtx) {
        if (charts.correlation) charts.correlation.destroy();
        charts.correlation = new Chart(correlationCtx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: ['N-P', 'N-K', 'P-K', 'pH-Fert', 'Temp-Fert', 'Rain-Fert'],
                datasets: [{
                    label: 'Correlation Strength',
                    data: [0.72, 0.65, 0.58, 0.45, 0.32, 0.28],
                    backgroundColor: '#9c27b0'
                }]
            },
            options: {
                responsive: true,
                scales: { y: { beginAtZero: true, max: 1 } }
            }
        });
    }
}

function renderModelCharts(performanceData) {
    const labels = performanceData.map(m => m.model_name);

    // --- R2 Score Chart ---
    const r2Ctx = document.getElementById('model-r2-chart');
    if (charts.r2Chart) charts.r2Chart.destroy();

    charts.r2Chart = new Chart(r2Ctx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'R² Score',
                data: performanceData.map(m => m.r2.toFixed(3)),
                backgroundColor: ['#4caf50', '#2196f3', '#ff9800']
            }]
        },
        options: {
            responsive: true,
            scales: { y: { beginAtZero: true, max: 1 } }
        }
    });

    // --- RMSE Chart ---
    const rmseCtx = document.getElementById('model-rmse-chart');
    if (charts.rmseChart) charts.rmseChart.destroy();

    charts.rmseChart = new Chart(rmseCtx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'RMSE',
                data: performanceData.map(m => m.rmse.toFixed(2)),
                backgroundColor: ['#4caf50', '#2196f3', '#ff9800']
            }]
        },
        options: {
            responsive: true,
            scales: { y: { beginAtZero: true } }
        }
    });
}

function renderModelTable(performanceData) {
    const tbody = document.querySelector('#models .model-table tbody');
    if (!tbody) return;
    tbody.innerHTML = ''; // Clear existing placeholders

    performanceData.forEach(m => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${m.model_name}</td>
            <td>${m.r2.toFixed(3)}</td>
            <td>${m.rmse.toFixed(2)}</td>
            <td>${m.mae.toFixed(2)}</td>
            <td>${m.cv_rmse.toFixed(2)}</td>
        `;
        tbody.appendChild(row);
    });
}


function renderPredictionHistory(predictions) {
    const tbody = document.getElementById('predictions-tbody');
    if (!tbody) return;
    tbody.innerHTML = ''; // Clear loading state

    if (predictions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="no-data-cell">No predictions yet. Make one!</td></tr>';
        return;
    }

    let totalFertilizer = 0;
    const cropCounts = {};

    predictions.forEach(p => {
        const tr = document.createElement('tr');
        const date = new Date(p.prediction_date).toLocaleDateString();

        tr.innerHTML = `
            <td>${date}</td>
            <td>${p.crop_type}</td>
            <td>${p.soil_ph.toFixed(1)}</td>
            <td>${p.temperature.toFixed(1)}°C</td>
            <td>${p.rainfall.toFixed(0)} mm</td>
            <td><strong>${p.predicted_fertilizer.toFixed(1)} kg/ha</strong></td>
            <td>${p.model_used}</td>
        `;
        tbody.appendChild(tr);

        // Calculate stats
        totalFertilizer += p.predicted_fertilizer;
        cropCounts[p.crop_type] = (cropCounts[p.crop_type] || 0) + 1;
    });

    // Update stats panel
    const avgFertilizer = totalFertilizer / predictions.length;
    let popularCrop = '-';
    if (Object.keys(cropCounts).length > 0) {
        popularCrop = Object.keys(cropCounts).reduce((a, b) => cropCounts[a] > cropCounts[b] ? a : b);
    }

    document.getElementById('total-predictions').textContent = predictions.length;
    document.getElementById('avg-fertilizer').textContent = `${avgFertilizer.toFixed(1)} kg/ha`;
    document.getElementById('popular-crop').textContent = popularCrop;

    // Re-initialize filter/refresh handlers after loading data
    document.getElementById('refresh-history').addEventListener('click', loadAnalyticsData);
}


// Utility functions
function formatNumber(num, decimals = 1) {
    if (num === null || num === undefined || isNaN(num)) return '0';
    return Number(num).toLocaleString('en-IN', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });
}

function debounce(func, wait = 300) {
    let timeout;
    return function (...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

window.AgriFert = {
    validateField,
    formatNumber,
    debounce
};