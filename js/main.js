document.addEventListener('DOMContentLoaded', () => {
    /* ---------- Mobile navigation ---------- */
    const toggle = document.querySelector('.nav-toggle');
    const navList = document.querySelector('.nav-links');

    if (toggle && navList) {
        toggle.addEventListener('click', () => {
            const open = navList.classList.toggle('open');
            toggle.setAttribute('aria-expanded', String(open));
        });
        navList.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                navList.classList.remove('open');
                toggle.setAttribute('aria-expanded', 'false');
            });
        });
    }

    /* ---------- Lazy videos: only load/play what is on screen ----------
       The page ships ~50 looping clips; loading them all at once stalls
       the browser, so each <source> keeps its URL in data-src until the
       clip scrolls into view.                                          */
    const lazyVideos = document.querySelectorAll('video[data-lazy]');

    const loadVideo = video => {
        if (!video.dataset.loaded) {
            video.querySelectorAll('source[data-src]').forEach(source => {
                source.src = source.dataset.src;
            });
            video.load();
            video.dataset.loaded = '1';
        }
        const played = video.play();
        if (played) played.catch(() => {});
    };

    if ('IntersectionObserver' in window) {
        const videoObserver = new IntersectionObserver(entries => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    loadVideo(entry.target);
                } else if (entry.target.dataset.loaded) {
                    entry.target.pause();
                }
            });
        }, { rootMargin: '250px 0px' });

        lazyVideos.forEach(video => videoObserver.observe(video));
    } else {
        lazyVideos.forEach(loadVideo);
    }

    /* ---------- Fade sections in as they enter the viewport ---------- */
    const revealables = document.querySelectorAll('.reveal');
    if ('IntersectionObserver' in window) {
        const revealObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('in-view');
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.08 });
        revealables.forEach(el => revealObserver.observe(el));
    } else {
        revealables.forEach(el => el.classList.add('in-view'));
    }

    /* ---------- Copy BibTeX ---------- */
    document.querySelectorAll('.copy-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const code = btn.parentElement.querySelector('code');
            if (!code) return;
            try {
                await navigator.clipboard.writeText(code.innerText);
            } catch (err) {
                const range = document.createRange();
                range.selectNodeContents(code);
                const selection = window.getSelection();
                selection.removeAllRanges();
                selection.addRange(range);
                document.execCommand('copy');
                selection.removeAllRanges();
            }
            btn.textContent = 'Copied!';
            btn.classList.add('copied');
            setTimeout(() => {
                btn.textContent = 'Copy';
                btn.classList.remove('copied');
            }, 1600);
        });
    });

    /* ---------- Scroll spy + back-to-top ---------- */
    const sections = document.querySelectorAll('section[id], header[id]');
    const navLinks = document.querySelectorAll('.nav-links a[href^="#"]');
    const toTop = document.querySelector('.to-top');

    if (toTop) {
        toTop.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));
    }

    let ticking = false;
    const onScroll = () => {
        if (ticking) return;
        ticking = true;
        window.requestAnimationFrame(() => {
            let current = '';
            sections.forEach(section => {
                if (window.scrollY >= section.offsetTop - 120) {
                    current = section.id;
                }
            });
            navLinks.forEach(link => {
                link.classList.toggle('active', link.getAttribute('href') === `#${current}`);
            });
            if (toTop) toTop.classList.toggle('visible', window.scrollY > 600);
            ticking = false;
        });
    };

    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
});
