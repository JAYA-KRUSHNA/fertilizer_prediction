// ═══════════════════════════════════════════════
// CropWise AI — Premium Home Effects
// Particles, Parallax, Counters, Tilt, Scroll FX
// ═══════════════════════════════════════════════

(function () {
    'use strict';

    // ── Floating Particle System ──
    function initParticleCanvas() {
        const canvas = document.getElementById('particle-canvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        let width, height;
        let particles = [];
        let animId;

        function resize() {
            width = canvas.width = window.innerWidth;
            height = canvas.height = window.innerHeight;
        }

        class Particle {
            constructor() {
                this.reset();
            }
            reset() {
                this.x = Math.random() * width;
                this.y = Math.random() * height;
                this.size = Math.random() * 3 + 1;
                this.speedX = (Math.random() - 0.5) * 0.5;
                this.speedY = (Math.random() - 0.5) * 0.3 - 0.2;
                this.opacity = Math.random() * 0.5 + 0.1;
                this.hue = 120 + Math.random() * 40; // Green spectrum
            }
            update() {
                this.x += this.speedX;
                this.y += this.speedY;
                if (this.y < -10 || this.x < -10 || this.x > width + 10) this.reset();
                if (this.y < 0) { this.y = height; this.x = Math.random() * width; }
            }
            draw() {
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
                ctx.fillStyle = `hsla(${this.hue}, 70%, 55%, ${this.opacity})`;
                ctx.fill();
            }
        }

        function init() {
            resize();
            const count = Math.min(80, Math.floor(width * height / 15000));
            particles = Array.from({ length: count }, () => new Particle());
        }

        function drawConnections() {
            for (let i = 0; i < particles.length; i++) {
                for (let j = i + 1; j < particles.length; j++) {
                    const dx = particles[i].x - particles[j].x;
                    const dy = particles[i].y - particles[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 120) {
                        ctx.beginPath();
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.strokeStyle = `hsla(140, 60%, 50%, ${0.08 * (1 - dist / 120)})`;
                        ctx.lineWidth = 0.5;
                        ctx.stroke();
                    }
                }
            }
        }

        function animate() {
            ctx.clearRect(0, 0, width, height);
            particles.forEach(p => { p.update(); p.draw(); });
            drawConnections();
            animId = requestAnimationFrame(animate);
        }

        init();
        animate();
        window.addEventListener('resize', () => { resize(); init(); });
    }

    // ── Animated Counter ──
    function initCounters() {
        const counters = document.querySelectorAll('[data-count]');
        const observed = new Set();

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting && !observed.has(entry.target)) {
                    observed.add(entry.target);
                    animateCounter(entry.target);
                }
            });
        }, { threshold: 0.5 });

        counters.forEach(el => observer.observe(el));
    }

    function animateCounter(el) {
        const target = parseFloat(el.dataset.count);
        const isFloat = String(target).includes('.');
        const duration = 2000;
        const startTime = performance.now();

        function tick(now) {
            const elapsed = now - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 4); // easeOutQuart
            const current = target * eased;

            el.textContent = isFloat ? current.toFixed(1) : Math.floor(current);

            if (progress < 1) requestAnimationFrame(tick);
            else el.textContent = isFloat ? target.toFixed(1) : target;
        }

        requestAnimationFrame(tick);
    }

    // ── Parallax Scrolling ──
    function initParallax() {
        const parallaxElements = document.querySelectorAll('[data-parallax]');
        if (!parallaxElements.length) return;

        function handleScroll() {
            const scrollY = window.scrollY;
            parallaxElements.forEach(el => {
                const speed = parseFloat(el.dataset.parallax);
                el.style.transform = `translateY(${scrollY * speed}px)`;
            });
        }

        window.addEventListener('scroll', handleScroll, { passive: true });
    }

    // ── Card Tilt Effect ──
    function initTiltCards() {
        const cards = document.querySelectorAll('[data-tilt]');

        cards.forEach(card => {
            card.addEventListener('mousemove', (e) => {
                const rect = card.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                const centerX = rect.width / 2;
                const centerY = rect.height / 2;
                const rotateX = (y - centerY) / centerY * -8;
                const rotateY = (x - centerX) / centerX * 8;

                card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateZ(10px)`;
                card.style.boxShadow = `${-rotateY * 2}px ${rotateX * 2}px 30px rgba(46, 125, 50, 0.15)`;

                // Move glow
                const glowX = (x / rect.width) * 100;
                const glowY = (y / rect.height) * 100;
                card.style.setProperty('--glow-x', `${glowX}%`);
                card.style.setProperty('--glow-y', `${glowY}%`);
            });

            card.addEventListener('mouseleave', () => {
                card.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) translateZ(0)';
                card.style.boxShadow = '';
            });
        });
    }

    // ── Scroll Reveal Animations ──
    function initScrollReveal() {
        const revealElements = document.querySelectorAll(
            '.feature-card, .stat-card, .step, .section-header, .trust-item, .hero-metric, .how-it-works-cta'
        );

        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry, i) => {
                if (entry.isIntersecting) {
                    // Add staggered delay
                    const siblings = entry.target.parentElement?.children;
                    let index = 0;
                    if (siblings) {
                        index = Array.from(siblings).indexOf(entry.target);
                    }
                    setTimeout(() => {
                        entry.target.classList.add('revealed');
                    }, index * 100);
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.15, rootMargin: '0px 0px -40px 0px' });

        revealElements.forEach(el => {
            el.classList.add('reveal-on-scroll');
            observer.observe(el);
        });
    }

    // ── Smooth Scroll for Anchor Links ──
    function initSmoothScroll() {
        document.querySelectorAll('a[href^="#"]').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const target = document.querySelector(link.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            });
        });
    }

    // ── Hero Scroll Indicator Hide ──
    function initScrollIndicator() {
        const indicator = document.querySelector('.hero-scroll-indicator');
        if (!indicator) return;

        window.addEventListener('scroll', () => {
            if (window.scrollY > 100) {
                indicator.style.opacity = '0';
                indicator.style.pointerEvents = 'none';
            } else {
                indicator.style.opacity = '1';
                indicator.style.pointerEvents = 'auto';
            }
        }, { passive: true });
    }

    // ── Header Scroll Shrink ─
    function initHeaderScroll() {
        const header = document.querySelector('header');
        if (!header) return;

        window.addEventListener('scroll', () => {
            if (window.scrollY > 50) {
                header.classList.add('header-scrolled');
            } else {
                header.classList.remove('header-scrolled');
            }
        }, { passive: true });
    }

    // ── Stat Bar Animation ──
    function initStatBars() {
        const bars = document.querySelectorAll('.stat-bar-fill');
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animated');
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.5 });

        bars.forEach(bar => observer.observe(bar));
    }

    // ── Init All ──
    document.addEventListener('DOMContentLoaded', () => {
        initParticleCanvas();
        initCounters();
        initParallax();
        initTiltCards();
        initScrollReveal();
        initSmoothScroll();
        initScrollIndicator();
        initHeaderScroll();
        initStatBars();
    });

})();
