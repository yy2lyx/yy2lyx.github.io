(function () {
  var showcase = document.querySelector('[data-home-showcase]');
  if (!showcase) return;

  var slides = Array.prototype.slice.call(showcase.querySelectorAll('[data-slide]'));
  var dots = Array.prototype.slice.call(showcase.querySelectorAll('[data-showcase-dot]'));
  var current = showcase.querySelector('[data-showcase-current]');
  var progress = showcase.querySelector('.yy-showcase__progress span');
  var reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  var activeIndex = 0;
  var intervalId;
  var scrollFrame;
  var isPointerDown = false;

  function setSlide(index, shouldRestart) {
    activeIndex = (index + slides.length) % slides.length;
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

    if (shouldRestart) restartAutoplay();
  }

  function stopAutoplay() {
    window.clearInterval(intervalId);
  }

  function restartAutoplay() {
    stopAutoplay();
    if (reducedMotion.matches || document.hidden) return;
    intervalId = window.setInterval(function () {
      setSlide(activeIndex + 1, false);
    }, 6800);
  }

  function updateParallax() {
    var rect = showcase.getBoundingClientRect();
    var trackHeight = Math.max(showcase.offsetHeight - window.innerHeight, 1);
    var progressValue = Math.min(Math.max(-rect.top / trackHeight, 0), 1);
    var activeFromScroll = Math.min(slides.length - 1, Math.floor(progressValue * slides.length + 0.15));

    if (!isPointerDown && rect.top <= 0 && rect.bottom >= window.innerHeight && activeFromScroll !== activeIndex) {
      setSlide(activeFromScroll, true);
    }

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

  showcase.addEventListener('pointerdown', function () { isPointerDown = true; });
  showcase.addEventListener('pointerup', function () { isPointerDown = false; });
  showcase.addEventListener('pointercancel', function () { isPointerDown = false; });
  document.addEventListener('visibilitychange', restartAutoplay);
  window.addEventListener('scroll', requestParallax, { passive: true });
  window.addEventListener('resize', requestParallax);
  reducedMotion.addEventListener('change', restartAutoplay);

  setSlide(0, false);
  requestParallax();
  restartAutoplay();
}());
