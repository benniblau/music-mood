document.addEventListener('DOMContentLoaded', function () {

    // Initialize Bootstrap tooltips
    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(function (el) {
        new bootstrap.Tooltip(el);
    });

    // Character counter for textarea fields
    document.querySelectorAll('[data-char-limit]').forEach(function (field) {
        var limit = parseInt(field.dataset.charLimit, 10);
        var countEl = document.getElementById(field.id + '-count');
        if (!countEl) return;
        var update = function () {
            countEl.textContent = field.value.length;
            countEl.classList.toggle('text-danger', field.value.length > limit);
        };
        field.addEventListener('input', update);
        update();
    });

    // File input validation: ensure .md extension
    var uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function (e) {
            var fileInputs = uploadForm.querySelectorAll('input[type="file"]');
            for (var i = 0; i < fileInputs.length; i++) {
                var input = fileInputs[i];
                if (input.files.length > 0) {
                    var fileName = input.files[0].name.toLowerCase();
                    if (!fileName.endsWith('.md')) {
                        e.preventDefault();
                        input.classList.add('is-invalid');
                        var feedback = input.nextElementSibling;
                        if (!feedback || !feedback.classList.contains('invalid-feedback')) {
                            feedback = document.createElement('div');
                            feedback.classList.add('invalid-feedback');
                            input.parentNode.insertBefore(feedback, input.nextSibling);
                        }
                        feedback.textContent = 'Please upload a Markdown (.md) file.';
                        return;
                    }
                }
            }
        });
    }

    // Auto-dismiss flash alerts after 5 seconds
    document.querySelectorAll('.alert-dismissible').forEach(function (alert) {
        setTimeout(function () {
            var bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
            bsAlert.close();
        }, 5000);
    });

    // Word rotator animation
    var rotator = document.querySelector('.word-rotator');
    var rotatorItems = document.querySelectorAll('.word-rotator-item');
    if (rotator && rotatorItems.length > 1) {
        var maxWidth = 0;
        rotatorItems.forEach(function (item) {
            item.style.position = 'relative';
            item.style.visibility = 'hidden';
            item.style.display = 'block';
            var w = item.offsetWidth;
            if (w > maxWidth) maxWidth = w;
            item.style.position = '';
            item.style.visibility = '';
            item.style.display = '';
        });
        rotator.style.width = maxWidth + 'px';

        var currentIndex = 0;
        var paused = false;

        document.addEventListener('visibilitychange', function () {
            paused = document.hidden;
        });

        setInterval(function () {
            if (paused) return;
            var current = rotatorItems[currentIndex];
            current.classList.remove('active');
            current.classList.add('slide-out');
            currentIndex = (currentIndex + 1) % rotatorItems.length;
            var next = rotatorItems[currentIndex];
            next.classList.add('active');
            setTimeout(function () { current.classList.remove('slide-out'); }, 500);
        }, 2500);
    }

    // Vote widget (UI only — no backend)
    document.querySelectorAll('.vote-widget').forEach(function (widget) {
        var scoreEl = widget.querySelector('.vote-score');
        var upBtn = widget.querySelector('.btn-vote-up');
        var downBtn = widget.querySelector('.btn-vote-down');
        var score = parseInt(scoreEl.textContent, 10) || 0;

        function setVote(value) {
            if (value === 1) {
                upBtn.classList.add('active');
                upBtn.querySelector('i').className = 'bi bi-hand-thumbs-up-fill';
                downBtn.classList.remove('active');
                downBtn.querySelector('i').className = 'bi bi-hand-thumbs-down';
            } else if (value === -1) {
                downBtn.classList.add('active');
                downBtn.querySelector('i').className = 'bi bi-hand-thumbs-down-fill';
                upBtn.classList.remove('active');
                upBtn.querySelector('i').className = 'bi bi-hand-thumbs-up';
            } else {
                upBtn.classList.remove('active');
                upBtn.querySelector('i').className = 'bi bi-hand-thumbs-up';
                downBtn.classList.remove('active');
                downBtn.querySelector('i').className = 'bi bi-hand-thumbs-down';
            }
        }

        upBtn.addEventListener('click', function () {
            if (upBtn.classList.contains('active')) {
                score--;
                setVote(0);
            } else {
                score += downBtn.classList.contains('active') ? 2 : 1;
                setVote(1);
            }
            scoreEl.textContent = score;
        });

        downBtn.addEventListener('click', function () {
            if (downBtn.classList.contains('active')) {
                score++;
                setVote(0);
            } else {
                score -= upBtn.classList.contains('active') ? 2 : 1;
                setVote(-1);
            }
            scoreEl.textContent = score;
        });
    });

    // Theme toggle
    var themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) {
        function updateIcon() {
            var dark = document.documentElement.getAttribute('data-bs-theme') === 'dark';
            themeBtn.querySelector('i').className = dark ? 'bi bi-sun-fill' : 'bi bi-moon-fill';
        }
        updateIcon();
        themeBtn.addEventListener('click', function () {
            var current = document.documentElement.getAttribute('data-bs-theme');
            var next = current === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-bs-theme', next);
            localStorage.setItem('theme', next);
            updateIcon();
        });
    }

    // Tag picker (UI only)
    var tagSearch = document.getElementById('tag-search');
    if (tagSearch) {
        var tagResults = document.getElementById('tag-search-results');
        var tagSelected = document.getElementById('tag-selected');
        var tagHidden = document.getElementById('tags-hidden-input');
        var sampleTags = ['automation', 'analytics', 'integration', 'deployment', 'security', 'monitoring', 'reporting', 'workflow'];

        function getSelectedTags() {
            return tagHidden.value ? tagHidden.value.split(',').map(function (t) { return t.trim(); }).filter(Boolean) : [];
        }

        function addTag(name) {
            name = name.toLowerCase().trim().replace(/[^a-z0-9\-_]/g, '-').replace(/-+/g, '-');
            if (!name) return;
            var tags = getSelectedTags();
            if (tags.includes(name) || tags.length >= 10) return;
            tags.push(name);
            tagHidden.value = tags.join(', ');
            var pill = document.createElement('span');
            pill.className = 'mcp-dep-pill';
            pill.dataset.tag = name;
            pill.innerHTML = name + ' <span class="remove-dep">&times;</span>';
            pill.querySelector('.remove-dep').addEventListener('click', function () {
                pill.remove();
                var cur = getSelectedTags().filter(function (t) { return t !== name; });
                tagHidden.value = cur.join(', ');
            });
            tagSelected.appendChild(pill);
        }

        tagSearch.addEventListener('input', function () {
            var q = tagSearch.value.trim().toLowerCase();
            if (q.length < 1) { tagResults.style.display = 'none'; return; }
            var selected = getSelectedTags();
            var filtered = sampleTags.filter(function (t) { return t.includes(q) && !selected.includes(t); });
            if (filtered.length) {
                tagResults.innerHTML = filtered.map(function (t) {
                    return '<button type="button" class="list-group-item list-group-item-action" data-tag="' + t + '">' + t + '</button>';
                }).join('');
                tagResults.style.display = 'block';
            } else if (q.length >= 2) {
                tagResults.innerHTML = '<button type="button" class="list-group-item list-group-item-action" data-tag="' + q + '"><i class="bi bi-plus-circle me-1"></i> Create "' + q + '"</button>';
                tagResults.style.display = 'block';
            } else {
                tagResults.style.display = 'none';
            }
        });

        tagSearch.addEventListener('keydown', function (e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                var val = tagSearch.value.trim();
                if (val) { addTag(val); tagSearch.value = ''; tagResults.style.display = 'none'; }
            }
        });

        tagResults.addEventListener('click', function (e) {
            var btn = e.target.closest('[data-tag]');
            if (btn) { addTag(btn.dataset.tag); tagSearch.value = ''; tagResults.style.display = 'none'; tagSearch.focus(); }
        });

        document.addEventListener('click', function (e) {
            if (!tagResults.contains(e.target) && e.target !== tagSearch) tagResults.style.display = 'none';
        });
    }

    // Collapsible sections
    document.querySelectorAll('.category-header').forEach(function (header) {
        header.addEventListener('click', function () {
            var target = document.getElementById(header.dataset.target);
            if (target) target.classList.toggle('d-none');
            var icon = header.querySelector('.bi-chevron-down, .bi-chevron-right');
            if (icon) icon.classList.toggle('bi-chevron-down'), icon.classList.toggle('bi-chevron-right');
        });
    });

    // Markdown toolbar preview toggle (UI demo)
    document.querySelectorAll('.md-preview-toggle').forEach(function (btn) {
        var toolbar = btn.closest('.md-toolbar');
        var ta = toolbar.nextElementSibling;
        while (ta && ta.tagName !== 'TEXTAREA') ta = ta.nextElementSibling;
        if (!ta) ta = toolbar.parentElement.querySelector('textarea');
        var previewPane = ta ? document.getElementById(ta.id + '-preview') : null;
        if (!ta || !previewPane) return;
        var previewing = false;

        btn.addEventListener('click', function () {
            previewing = !previewing;
            if (previewing) {
                btn.textContent = 'Edit';
                btn.classList.add('active');
                ta.style.display = 'none';
                previewPane.style.display = 'block';
                previewPane.innerHTML = '<p>' + (ta.value || '<em class="text-muted">Nothing to preview.</em>') + '</p>';
            } else {
                btn.textContent = 'Preview';
                btn.classList.remove('active');
                ta.style.display = '';
                previewPane.style.display = 'none';
            }
        });
    });
});
