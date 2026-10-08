(function () {
  var showcase = document.querySelector('[data-home-showcase]');
  if (!showcase) return;

  var slides = Array.prototype.slice.call(showcase.querySelectorAll('[data-slide]'));
  var dots = Array.prototype.slice.call(showcase.querySelectorAll('[data-showcase-dot]'));
  var current = showcase.querySelector('[data-showcase-current]');
  var progress = showcase.querySelector('.yy-showcase__progress span');
  var reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  var activeIndex = 0;
  var lastLoadedIndex = 0;
  var intervalId;
  var scrollFrame;
  var isShowcaseVisible = true;
  var preloadTimer;

  function loadSlideImage(index) {
    var image = slides[index].querySelector('img[data-src]');
    if (!image) return;
    image.src = image.dataset.src;
    image.removeAttribute('data-src');
  }

  function scheduleNextImage() {
    window.clearTimeout(preloadTimer);
    preloadTimer = window.setTimeout(function () {
      if (!isShowcaseVisible || document.hidden || lastLoadedIndex >= slides.length - 1) return;
      lastLoadedIndex += 1;
      loadSlideImage(lastLoadedIndex);
    }, 3000);
  }

  function setSlide(index, shouldRestart) {
    activeIndex = (index + slides.length) % slides.length;
    loadSlideImage(activeIndex);
    lastLoadedIndex = Math.max(lastLoadedIndex, activeIndex);
    showcase.classList.toggle('is-light', slides[activeIndex].classList.contains('yy-showcase__slide--light'));

    slides.forEach(function (slide, slideIndex) {
      slide.classList.toggle('is-active', slideIndex === activeIndex);
      slide.setAttribute('aria-hidden', slideIndex === activeIndex ? 'false' : 'true');
    });

    dots.forEach(function (dot, dotIndex) {
      var active = dotIndex === activeIndex;
      dot.classList.toggle('is-active', active);
      if (active) {
        dot.setAttribute('aria-current', 'true');
      } else {
        dot.removeAttribute('aria-current');
      }
    });

    if (current) current.textContent = String(activeIndex + 1).padStart(2, '0');
    if (progress) progress.style.transform = 'scaleX(' + ((activeIndex + 1) / slides.length) + ')';

    scheduleNextImage();
    if (shouldRestart) restartAutoplay();
  }

  function stopAutoplay() {
    window.clearInterval(intervalId);
  }

  function restartAutoplay() {
    stopAutoplay();
    if (reducedMotion.matches || document.hidden || !isShowcaseVisible) return;
    intervalId = window.setInterval(function () {
      setSlide(activeIndex + 1, false);
    }, 6800);
  }

  function updateParallax() {
    var rect = showcase.getBoundingClientRect();
    var progressValue = Math.min(Math.max(-rect.top / Math.max(window.innerHeight, 1), 0), 1);

    slides.forEach(function (slide) {
      var image = slide.querySelector('img');
      if (image) {
        var offset = (progressValue - 0.5) * 22;
        image.style.setProperty('--yy-parallax-y', offset.toFixed(2) + 'px');
      }
    });
    scrollFrame = null;
  }

  function requestParallax() {
    if (!scrollFrame) scrollFrame = window.requestAnimationFrame(updateParallax);
  }

  showcase.querySelector('[data-showcase-prev]').addEventListener('click', function () {
    setSlide(activeIndex - 1, true);
  });

  showcase.querySelector('[data-showcase-next]').addEventListener('click', function () {
    setSlide(activeIndex + 1, true);
  });

  dots.forEach(function (dot) {
    dot.addEventListener('click', function () {
      setSlide(parseInt(dot.dataset.showcaseDot, 10), true);
    });
  });

  document.addEventListener('visibilitychange', restartAutoplay);
  window.addEventListener('scroll', requestParallax, { passive: true });
  window.addEventListener('resize', requestParallax);
  reducedMotion.addEventListener('change', restartAutoplay);

  if ('IntersectionObserver' in window) {
    new IntersectionObserver(function (entries) {
      isShowcaseVisible = entries[0].isIntersecting;
      restartAutoplay();
    }).observe(showcase);
  }

  setSlide(0, false);
  requestParallax();
  restartAutoplay();
}());
