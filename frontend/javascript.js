const API_BASE_URL = window.COMPASS_API_BASE_URL || 'http://localhost:8000';
const STATUS_ORDER = ['non_compliant', 'ambiguous', 'compliant'];
const CATEGORY_COLORS = {
    non_compliant: '#C53030',
    ambiguous: '#D69E2E',
    compliant: '#2F855A',
};
const STATUS_LABELS = {
    non_compliant: 'Non-Compliant',
    ambiguous: 'Ambiguous',
    compliant: 'Compliant',
};
const STATUS_EMOJI = {
    non_compliant: '⚠️',
    ambiguous: '🟡',
    compliant: '✅',
};
const STATUS_RISK = {
    non_compliant: 'High',
    ambiguous: 'Medium',
    compliant: 'Low',
};
const CLAUSE_LABELS = {
    data_collection: 'Data Collection',
    data_sharing: 'Data Sharing',
    user_consent: 'User Consent',
    data_retention: 'Data Retention',
    rights_of_user: 'Rights of Users',
    security_measures: 'Security Measures',
    non_compliance_warning: 'Potential Non-Compliance',
    cookies_tracking: 'Cookies & Tracking',
};

const RULE_LABEL_STATUS = {
    non_compliance_warning: 'non_compliant',
    data_sharing: 'non_compliant',
    data_collection: 'ambiguous',
    data_retention: 'ambiguous',
    cookies_tracking: 'ambiguous',
    user_consent: 'compliant',
    rights_of_user: 'compliant',
    security_measures: 'compliant',
};

const RISK_LEVEL_TO_STATUS = {
    High: 'non_compliant',
    Medium: 'ambiguous',
    Low: 'compliant',
};

const RISK_COLOR_TO_STATUS = {
    red: 'non_compliant',
    yellow: 'ambiguous',
    green: 'compliant',
};

const LABEL_GDPR_ARTICLES = {
    data_collection: ['Art. 5(1)(a)', 'Art. 6'],
    data_sharing: ['Art. 13', 'Art. 14', 'Art. 28'],
    user_consent: ['Art. 6(1)(a)', 'Art. 7'],
    data_retention: ['Art. 5(1)(e)', 'Art. 30'],
    rights_of_user: ['Art. 12', 'Art. 15-18', 'Art. 21'],
    security_measures: ['Art. 32'],
    non_compliance_warning: ['Art. 5', 'Art. 83'],
    cookies_tracking: ['Art. 6(1)(a)', 'Art. 7', 'Art. 5(3) ePrivacy'],
};

const LABEL_RECOMMENDATIONS = {
    data_collection: [
        'Document lawful bases for all data collection activities.',
        'Update the privacy notice to describe what data is collected and why.',
    ],
    data_sharing: [
        'List all third parties receiving data and specify their roles.',
        'Obtain explicit consent before sharing personal data for secondary purposes.',
        'Execute controller-processor contracts with each processor (Art. 28).',
    ],
    user_consent: [
        'Ensure consent is freely given, specific, informed, and unambiguous.',
        'Provide clear opt-in mechanisms with the ability to withdraw consent easily.',
    ],
    data_retention: [
        'Define concrete retention periods or criteria for each data category.',
        'Implement deletion or anonymisation routines after retention periods expire.',
    ],
    rights_of_user: [
        'Document procedures to respond to data subject requests within one month.',
        'Provide self-service tools or contact channels for exercising data subject rights.',
    ],
    security_measures: [
        'Review technical and organisational measures regularly for effectiveness.',
        'Implement encryption, access controls, and incident response plans.',
    ],
    non_compliance_warning: [
        'Perform a DPIA to identify and mitigate high-risk processing activities.',
        'Engage legal counsel to align processing activities with GDPR principles.',
    ],
    cookies_tracking: [
        'Deploy a consent banner that records affirmative choices before setting non-essential cookies.',
        'Offer granular controls for analytics, advertising, and functional cookies.',
    ],
};

const STATUS_DEFAULT_ARTICLES = {
    non_compliant: ['Art. 5', 'Art. 6'],
    ambiguous: ['Art. 5(1)(a)', 'Art. 24'],
    compliant: ['Art. 24', 'Art. 32'],
};

const STATUS_DEFAULT_RECOMMENDATIONS = {
    non_compliant: [
        'Address identified legal gaps before continuing the processing activity.',
        'Update governance documentation to evidence compliance with GDPR principles.',
    ],
    ambiguous: [
        'Clarify ambiguous language in the privacy policy with specific legal bases.',
        'Add detail about user rights, retention periods, or safeguards to remove uncertainty.',
    ],
    compliant: [
        'Maintain records of processing and monitor changes to ensure continued compliance.',
    ],
};

let lastAnalysisReport = null;
let controlsInitialized = false;
let fullReportData = [];
const currentFilters = {
    search: '',
    risk: 'all',
    sort: 'default',
};

document.addEventListener('DOMContentLoaded', () => {
    setupTabs();
    setupFileUpload();
    setupTextCounter();
    setupReportControls();

    const submitBtn = document.getElementById('submit-analyze-btn');
    if (submitBtn) {
        submitBtn.addEventListener('click', handleSubmit);
    }

    const downloadBtn = document.getElementById('download-report-btn');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', downloadReport);
    }

    const cursor = document.querySelector('.cursor-follower');
    if (cursor) {
        document.addEventListener('mousemove', (event) => {
            cursor.style.transform = `translate3d(${event.clientX - 16}px, ${event.clientY - 16}px, 0)`;
        });
    }
});

function setupTabs() {
    const methodTabs = document.querySelectorAll('.method-tab');
    const inputMethods = document.querySelectorAll('.input-method');

    methodTabs.forEach((tab) => {
        tab.addEventListener('click', function handleTabClick() {
            const method = this.getAttribute('data-method');

            methodTabs.forEach((t) => t.classList.remove('active'));
            this.classList.add('active');

            inputMethods.forEach((methodPane) => methodPane.classList.remove('active'));
            document.getElementById(`${method}-method`).classList.add('active');
        });
    });
}

function setupFileUpload() {
    const fileInput = document.getElementById('policy-file');
    const dropZone = document.getElementById('file-drop-zone');
    if (!fileInput || !dropZone) return;

    const fileLabel = dropZone.querySelector('p');

    dropZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (event) => {
        const file = event.target.files?.[0];
        if (file && fileLabel) {
            fileLabel.textContent = `Selected: ${file.name}`;
        }
    });

    dropZone.addEventListener('dragover', (event) => {
        event.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', (event) => {
        event.preventDefault();
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (event) => {
        event.preventDefault();
        dropZone.classList.remove('dragover');
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            if (fileLabel) {
                fileLabel.textContent = `Selected: ${files[0].name}`;
            }
        }
    });
}

function setupTextCounter() {
    const textArea = document.getElementById('policy-text');
    const charCount = document.getElementById('char-count');
    const wordCount = document.getElementById('word-count');
    if (!textArea || !charCount || !wordCount) return;

    textArea.addEventListener('input', function handleTextInput() {
        const text = this.value;
        const chars = text.length;
        const words = text.trim() ? text.trim().split(/\s+/).length : 0;

        charCount.textContent = `${chars} characters`;
        wordCount.textContent = `${words} words`;
    });
}

function setupReportControls() {
    if (controlsInitialized) return;
    const searchInput = document.getElementById('search-bar');
    const sortSelect = document.getElementById('sort-select');
    const riskButtons = Array.from(document.querySelectorAll('.risk-filter-btn'));

    searchInput?.addEventListener('input', (event) => {
        currentFilters.search = event.target.value.toLowerCase();
        renderReportGrid();
    });

    riskButtons.forEach((button) => {
        button.addEventListener('click', () => {
            riskButtons.forEach((btn) => btn.classList.remove('active'));
            button.classList.add('active');
            currentFilters.risk = button.dataset.value;
            renderReportGrid();
        });
    });

    sortSelect?.addEventListener('change', (event) => {
        currentFilters.sort = event.target.value;
        renderReportGrid();
    });

    controlsInitialized = true;
}

function handleSubmit() {
    const activeTab = document.querySelector('.method-tab.active');
    if (!activeTab) {
        showError('Select an input method to begin the analysis.');
        return;
    }

    const method = activeTab.getAttribute('data-method');
    if (method === 'url') {
        const url = document.getElementById('policy-url')?.value.trim();
        if (!url) {
            showError('Please enter a valid URL to analyze.');
            return;
        }
        startAnalysis('url', url);
    } else if (method === 'file') {
        const file = document.getElementById('policy-file')?.files?.[0];
        if (!file) {
            showError('Please select a privacy policy document to upload.');
            return;
        }
        startAnalysis('file', file);
    } else {
        const text = document.getElementById('policy-text')?.value;
        if (!text || !text.trim()) {
            showError('Please paste the privacy policy text before running the analysis.');
            return;
        }
        startAnalysis('text', text);
    }
}

async function startAnalysis(method, payload) {
    setLoadingState(true);
    resetResults();

    try {
        let response;

        if (method === 'file') {
            const formData = new FormData();
            formData.append('file', payload);
            response = await fetch(`${API_BASE_URL}/analyze/file`, {
                method: 'POST',
                body: formData,
            });
        } else if (method === 'url') {
            response = await fetch(`${API_BASE_URL}/analyze/url`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: payload }),
            });
        } else {
            response = await fetch(`${API_BASE_URL}/analyze/text`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content: payload }),
            });
        }

        if (!response.ok) {
            let detail = 'Analysis failed. Please try again.';
            try {
                const errorPayload = await response.json();
                if (errorPayload?.detail) {
                    detail = errorPayload.detail;
                }
            } catch (jsonError) {
                const fallbackText = await response.text();
                if (fallbackText) {
                    detail = fallbackText;
                }
            }
            throw new Error(detail);
        }

        const report = await response.json();
    const normalizedReport = normalizeReport(report);
    lastAnalysisReport = normalizedReport;
    hydrateReportData(normalizedReport);
    updateSummary(normalizedReport.summary, normalizedReport.metadata);
    renderFlaggedKeywords(normalizedReport.summary?.flagged_keywords || []);
        renderReportGrid();
        showReportShell();
    } catch (error) {
        showError(error.message || 'Unexpected error while running the analysis.');
        hideReportShell();
    } finally {
        setLoadingState(false);
    }
}

function normalizeReport(rawReport) {
    if (rawReport?.chunks && rawReport?.summary) {
        return rawReport;
    }

    const analysisResults = Array.isArray(rawReport?.analysis_results)
        ? rawReport.analysis_results
        : [];

    const statusCounts = {
        non_compliant: 0,
        ambiguous: 0,
        compliant: 0,
    };

    const flaggedMap = new Map();

    const chunks = analysisResults.map((entry, index) => {
        const text = entry.text || '';
        const ruleLabel = entry.rule_label || null;
        const ruleColor = (entry.rule_risk || '').toLowerCase();
        const explicitRiskLevel = entry.risk_level || null;

        let status = null;
        if (explicitRiskLevel && RISK_LEVEL_TO_STATUS[explicitRiskLevel]) {
            status = RISK_LEVEL_TO_STATUS[explicitRiskLevel];
        } else if (ruleColor && RISK_COLOR_TO_STATUS[ruleColor]) {
            status = RISK_COLOR_TO_STATUS[ruleColor];
        } else if (ruleLabel && RULE_LABEL_STATUS[ruleLabel]) {
            status = RULE_LABEL_STATUS[ruleLabel];
        }

        if (!status) {
            const aiLabel = entry.ai_label || '';
            const aiScore = typeof entry.ai_score === 'number' ? entry.ai_score : 0.5;
            if (aiScore >= 0.6) {
                status = aiLabel === 'LABEL_0' ? 'compliant' : 'non_compliant';
            } else if (aiScore <= 0.4) {
                status = aiLabel === 'LABEL_0' ? 'non_compliant' : 'compliant';
            } else {
                status = aiScore >= 0.5 ? 'ambiguous' : 'compliant';
            }
        }

        if (!STATUS_LABELS[status]) {
            status = 'ambiguous';
        }

        statusCounts[status] += 1;

        const articlesSet = new Set();
        const recommendationsSet = new Set();

        if (ruleLabel) {
            (LABEL_GDPR_ARTICLES[ruleLabel] || []).forEach((article) => articlesSet.add(article));
            (LABEL_RECOMMENDATIONS[ruleLabel] || []).forEach((rec) => recommendationsSet.add(rec));

            const flagged = flaggedMap.get(ruleLabel) || {
                label: ruleLabel,
                count: 0,
                keywords: new Set(),
                status: RULE_LABEL_STATUS[ruleLabel] || status,
            };
            flagged.count += 1;
            flagged.keywords.add(CLAUSE_LABELS[ruleLabel] || ruleLabel);
            flaggedMap.set(ruleLabel, flagged);
        }

        if (articlesSet.size === 0) {
            (STATUS_DEFAULT_ARTICLES[status] || []).forEach((article) => articlesSet.add(article));
        }

        (STATUS_DEFAULT_RECOMMENDATIONS[status] || []).forEach((rec) => recommendationsSet.add(rec));

        const ruleMatches = [];
        if (ruleLabel) {
            ruleMatches.push({
                label: ruleLabel,
                keyword: CLAUSE_LABELS[ruleLabel] || ruleLabel,
                risk: ruleColor || 'green',
                status: RULE_LABEL_STATUS[ruleLabel] || status,
            });
        }

        return {
            id: index + 1,
            text,
            status,
            color: CATEGORY_COLORS[status] || '#3182ce',
            risk_level: STATUS_RISK[status] || 'Medium',
            status_source: ruleLabel ? `rule:${ruleLabel}` : `ai:${entry.ai_label || 'prototype'}`,
            scores: {
                non_compliant: status === 'non_compliant' ? 100 : 0,
                ambiguous: status === 'ambiguous' ? 100 : 0,
                compliant: status === 'compliant' ? 100 : 0,
            },
            probability: Math.round((entry.ai_score || 0) * 10000) / 100,
            ai_status: entry.ai_label || 'unknown',
            ai_probability: Math.round((entry.ai_score || 0) * 10000) / 100,
            raw_similarity: {},
            calibrated_similarity: {},
            rule_matches: ruleMatches,
            gdpr_articles: Array.from(articlesSet),
            recommendations: Array.from(recommendationsSet),
            top_prototype: null,
        };
    });

    const totalChunks = chunks.length;
    const complianceRatio = totalChunks ? statusCounts.compliant / totalChunks : 0;

    const flagged_keywords = Array.from(flaggedMap.values()).map((item) => ({
        label: item.label,
        count: item.count,
        keywords: Array.from(item.keywords),
        risk: item.status,
        status: item.status,
    }));
    flagged_keywords.sort((a, b) => b.count - a.count);

    const sortedForTop = [...chunks].sort(
        (a, b) => STATUS_ORDER.indexOf(a.status) - STATUS_ORDER.indexOf(b.status)
    );

    const summary = {
        status_counts: statusCounts,
        compliance_ratio: Math.round(complianceRatio * 1000) / 1000,
        flagged_keywords,
        top_sections: sortedForTop.slice(0, 5).map((entry) => ({
            id: entry.id,
            status: entry.status,
            color: entry.color,
            risk_level: entry.risk_level,
            articles: entry.gdpr_articles,
            recommendations: entry.recommendations.slice(0, 2),
            snippet: entry.text.slice(0, 200),
        })),
    };

    const combinedText = analysisResults.map((entry) => entry.text || '').join(' ');
    const wordCount = rawReport?.word_count
        ?? (combinedText.trim() ? combinedText.trim().split(/\s+/).length : 0);
    const characterCount = rawReport?.character_count ?? combinedText.length;

    const metadata = {
        source: rawReport?.filename || rawReport?.document_name || 'uploaded document',
        source_type: rawReport?.content_type || 'text/plain',
        chunk_count: rawReport?.chunk_count || totalChunks,
        generated_at: new Date().toISOString(),
        device: 'Backend',
        character_count: characterCount,
        word_count: wordCount,
    };

    return {
        metadata,
        summary,
        chunks,
    };
}

function hydrateReportData(report) {
    const chunks = report?.chunks || [];
    fullReportData = chunks.map((chunk, index) => ({
        id: chunk.id ?? index + 1,
        text: chunk.text || '',
        status: chunk.status,
        risk_level: chunk.risk_level || STATUS_RISK[chunk.status] || 'Medium',
        color: chunk.color,
        status_source: chunk.status_source,
        rule_matches: chunk.rule_matches || [],
        articles: chunk.gdpr_articles || [],
        recommendations: chunk.recommendations || [],
        top_prototype: chunk.top_prototype || null,
    }));

    currentFilters.search = '';
    currentFilters.risk = 'all';
    currentFilters.sort = 'default';

    const searchInput = document.getElementById('search-bar');
    const sortSelect = document.getElementById('sort-select');
    const riskButtons = document.querySelectorAll('.risk-filter-btn');

    if (searchInput) searchInput.value = '';
    if (sortSelect) sortSelect.value = 'default';
    riskButtons.forEach((btn) => {
        if (btn.dataset.value === 'all') {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
}

function setLoadingState(isLoading) {
    const submitBtn = document.getElementById('submit-analyze-btn');
    const loader = document.getElementById('report-loader');
    const section = document.getElementById('compliance-results-section');

    if (isLoading) {
        if (submitBtn) {
            if (!submitBtn.dataset.originalLabel) {
                submitBtn.dataset.originalLabel = submitBtn.innerHTML;
            }
            submitBtn.innerHTML = '<span class="btn-text">Analyzing...</span><span class="btn-icon">🔄</span>';
            submitBtn.disabled = true;
        }
        if (section) {
            section.style.display = 'block';
            section.classList.add('visible');
        }
        if (loader) {
            loader.style.display = 'block';
        }
    } else {
        if (submitBtn) {
            submitBtn.innerHTML = submitBtn.dataset.originalLabel || submitBtn.innerHTML;
            submitBtn.disabled = false;
        }
        if (loader) {
            loader.style.display = 'none';
        }
    }
}

function resetResults() {
    const section = document.getElementById('compliance-results-section');
    const grid = document.getElementById('results-grid');
    const flaggedList = document.getElementById('flagged-keywords-list');
    const meta = document.getElementById('analysis-meta');
    const downloadBtn = document.getElementById('download-report-btn');

    fullReportData = [];

    if (section) {
        section.style.display = 'none';
        section.classList.remove('visible');
    }
    if (grid) {
        grid.innerHTML = '';
    }
    if (flaggedList) {
        flaggedList.innerHTML = '<p class="placeholder">No GDPR risk keywords detected in this document.</p>';
    }
    if (meta) {
        meta.textContent = '';
    }
    if (downloadBtn) {
        downloadBtn.disabled = true;
    }
}

function showReportShell() {
    const section = document.getElementById('compliance-results-section');
    const downloadBtn = document.getElementById('download-report-btn');
    if (section) {
        section.style.display = 'block';
        section.classList.add('visible');
    }
    if (downloadBtn) {
        downloadBtn.disabled = !lastAnalysisReport;
    }
}

function hideReportShell() {
    const section = document.getElementById('compliance-results-section');
    if (section) {
        section.style.display = 'none';
        section.classList.remove('visible');
    }
}

function updateSummary(summary, metadata) {
    const providedCounts = summary?.status_counts || {};
    const counts = {
        non_compliant: Number(providedCounts.non_compliant) || 0,
        ambiguous: Number(providedCounts.ambiguous) || 0,
        compliant: Number(providedCounts.compliant) || 0,
    };

    const totalProvided = counts.non_compliant + counts.ambiguous + counts.compliant;
    if (totalProvided === 0 && fullReportData.length) {
        counts.non_compliant = fullReportData.filter((item) => item.status === 'non_compliant').length;
        counts.ambiguous = fullReportData.filter((item) => item.status === 'ambiguous').length;
        counts.compliant = fullReportData.filter((item) => item.status === 'compliant').length;
    }

    const totalChunks = counts.non_compliant + counts.ambiguous + counts.compliant;
    let ratioValue = summary?.compliance_ratio;
    if (typeof ratioValue !== 'number' || Number.isNaN(ratioValue)) {
        ratioValue = totalChunks ? counts.compliant / totalChunks : 0;
    }

    const nonCompliantEl = document.getElementById('summary-non-compliant');
    const ambiguousEl = document.getElementById('summary-ambiguous');
    const compliantEl = document.getElementById('summary-compliant');
    const ratioEl = document.getElementById('summary-ratio');
    const meta = document.getElementById('analysis-meta');

    if (nonCompliantEl) nonCompliantEl.textContent = counts.non_compliant ?? 0;
    if (ambiguousEl) ambiguousEl.textContent = counts.ambiguous ?? 0;
    if (compliantEl) compliantEl.textContent = counts.compliant ?? 0;
    if (ratioEl) ratioEl.textContent = `${Math.round(ratioValue * 1000) / 10}%`;

    if (meta && metadata) {
        const generated = metadata.generated_at ? new Date(metadata.generated_at) : new Date();
        const formattedDate = generated.toLocaleString(undefined, {
            dateStyle: 'medium',
            timeStyle: 'short',
        });
        meta.textContent = `Analyzed ${metadata.chunk_count} sections • Source: ${metadata.source} • Generated ${formattedDate} • Inference on ${metadata.device}`;
    }
}

function renderFlaggedKeywords(flaggedKeywords) {
    const container = document.getElementById('flagged-keywords-list');
    if (!container) return;

    container.innerHTML = '';
    if (!flaggedKeywords.length) {
        container.innerHTML = '<p class="placeholder">No GDPR risk keywords detected in this document.</p>';
        return;
    }

    flaggedKeywords.forEach((item) => {
        const keywordChip = document.createElement('div');
        keywordChip.className = `keyword-chip status-${item.status}`;
        const label = CLAUSE_LABELS[item.label] || item.label;
        keywordChip.innerHTML = `<strong>${label}</strong> · ${item.count} hit${item.count > 1 ? 's' : ''}<br><span>${item.keywords.join(', ')}</span>`;
        container.appendChild(keywordChip);
    });
}

function renderReportGrid() {
    const grid = document.getElementById('results-grid');
    const downloadBtn = document.getElementById('download-report-btn');
    if (!grid) return;

    const riskOrder = { High: 3, Medium: 2, Low: 1 };

    let dataset = fullReportData.filter((item) => {
        const riskMatch = currentFilters.risk === 'all' || item.risk_level === currentFilters.risk;
        const searchTarget = `${item.text} ${item.recommendations.join(' ')} ${item.articles.join(' ')} ${formatStatusSource(item.status_source)}`.toLowerCase();
        const searchMatch = !currentFilters.search || searchTarget.includes(currentFilters.search);
        return riskMatch && searchMatch;
    });

    if (currentFilters.sort === 'risk-desc') {
        dataset.sort((a, b) => (riskOrder[b.risk_level] || 0) - (riskOrder[a.risk_level] || 0) || STATUS_ORDER.indexOf(a.status) - STATUS_ORDER.indexOf(b.status));
    } else if (currentFilters.sort === 'risk-asc') {
        dataset.sort((a, b) => (riskOrder[a.risk_level] || 0) - (riskOrder[b.risk_level] || 0) || STATUS_ORDER.indexOf(a.status) - STATUS_ORDER.indexOf(b.status));
    } else {
        dataset.sort((a, b) => STATUS_ORDER.indexOf(a.status) - STATUS_ORDER.indexOf(b.status));
    }

    grid.innerHTML = '';

    if (!dataset.length) {
        grid.innerHTML = '<p class="placeholder" style="text-align:center;">No sections match the current filters.</p>';
        if (downloadBtn) downloadBtn.disabled = !lastAnalysisReport;
        return;
    }

    dataset.forEach((item, index) => {
        grid.insertAdjacentHTML('beforeend', createClauseCard(item, index));
    });

    if (downloadBtn) downloadBtn.disabled = !lastAnalysisReport;
}

function createClauseCard(item, index) {
    const articles = item.articles.length
        ? item.articles.map((article) => `<span class="article-chip">${escapeHtml(article)}</span>`).join(' ')
        : '<span class="placeholder">No specific article identified</span>';

    const recommendations = item.recommendations.length
        ? `<ul class="recommendations-list">${item.recommendations.map((rec) => `<li>${escapeHtml(rec)}</li>`).join('')}</ul>`
        : '<p class="placeholder">No additional actions required.</p>';

    const ruleMatches = item.rule_matches?.length
        ? `<div class="rule-matches">${item.rule_matches
              .map((match) => {
                  const label = CLAUSE_LABELS[match.label] || match.label;
                  return `<span class="rule-match">${escapeHtml(label)}: “${escapeHtml(match.keyword)}”</span>`;
              })
              .join('')}</div>`
        : '<div class="rule-matches none">No GDPR risk keywords detected in this section.</div>';

    const statusPill = `${STATUS_EMOJI[item.status] || '📌'} ${STATUS_LABELS[item.status] || item.status}`;

    return `
        <article class="clause-card risk-${item.risk_level}" style="border-left-color: ${item.color || '#3182ce'}; animation-delay: ${index * 0.05}s;">
            <header class="card-header">
                <div class="header-left">
                    <span class="risk-badge badge-${item.risk_level}">${item.risk_level} Risk</span>
                    <span class="status-pill">${statusPill}</span>
                </div>
                <span class="reference">${item.articles[0] ? escapeHtml(item.articles[0]) : 'GDPR Ref'}</span>
            </header>
            <div class="clause-content">
                <h4>Section ${item.id}</h4>
                <p class="text">${escapeHtml(item.text)}</p>
                <div class="details-grid">
                    <div class="info-row"><strong>Decision Basis:</strong> ${escapeHtml(formatStatusSource(item.status_source))}</div>
                    <div class="info-row"><strong>GDPR Articles:</strong> ${articles}</div>
                    <div class="info-row"><strong>Recommendations:</strong> ${recommendations}</div>
                </div>
            </div>
            ${ruleMatches}
        </article>
    `;
}

function formatStatusSource(source) {
    if (!source) {
        return 'AI-driven classification';
    }
    if (source.startsWith('rule:')) {
        const [, label, keyword] = source.split(':');
        const readableLabel = CLAUSE_LABELS[label] || label;
        return `Rule match · ${readableLabel} · “${keyword}”`;
    }
    return 'AI-driven classification';
}

async function downloadReport() {
    if (!lastAnalysisReport) {
        showError('Run an analysis before downloading the report.');
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/report/pdf`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(lastAnalysisReport),
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(errorText || 'Failed to generate PDF report.');
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `compass-gdpr-report-${Date.now()}.pdf`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    } catch (error) {
        showError(error.message);
    }
}

function escapeHtml(text = '') {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function showError(message) {
    alert(message);
}
