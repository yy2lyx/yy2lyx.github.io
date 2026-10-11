---
layout: article
titles:
  # @start locale config
  en      : &EN       About
  en-GB   : *EN
  en-US   : *EN
  en-CA   : *EN
  en-AU   : *EN
  zh-Hans : &ZH_HANS  About Me
  zh      : *ZH_HANS
  zh-CN   : *ZH_HANS
  zh-SG   : *ZH_HANS
  zh-Hant : &ZH_HANT  關於
  zh-TW   : *ZH_HANT
  zh-HK   : *ZH_HANT
  ko      : &KO       소개
  ko-KR   : *KO
  fr      : &FR       À propos
  fr-BE   : *FR
  fr-CA   : *FR
  fr-CH   : *FR
  fr-FR   : *FR
  fr-LU   : *FR
  # @end locale config
key: page-about
---

<style>
  .yy-about-profile {
    display: grid !important;
    grid-template-columns: 220px minmax(0, 1fr) !important;
    gap: 56px !important;
    align-items: center !important;
  }

  .yy-about-profile__portrait {
    width: 220px !important;
    height: 220px !important;
  }

  .yy-about-profile__portrait img {
    display: block !important;
    width: 220px !important;
    height: 220px !important;
    object-fit: cover !important;
  }

  @media (max-width: 680px) {
    .yy-about-profile {
      grid-template-columns: 1fr !important;
      gap: 32px !important;
    }

    .yy-about-profile__portrait,
    .yy-about-profile__portrait img {
      width: 180px !important;
      height: 180px !important;
    }
  }
</style>

<section class="yy-about-profile" aria-labelledby="yy-about-name">
  <div class="yy-about-profile__portrait">
    <img src="{{ '/assets/images/yy-ai-avatar.webp' | relative_url }}" alt="CarlYe 的头像" width="720" height="720">
  </div>

  <div class="yy-about-profile__content">
    <h1 id="yy-about-name">CarlYe</h1>
    <p class="yy-about-profile__role">AI Practitioner · Deep Learning / NLP / CV</p>

    <p>你好，我是 CarlYe，一名专注于机器学习与深度学习的 AI 实践者。我的研究与实践横跨自然语言处理和计算机视觉，并以 Python 与 C++ 为主要工具，兼顾算法创新与系统性能。</p>

    <p>这个博客记录我在模型、工具和工程实践中的思考与经验。希望这些内容能为你带来启发，也期待与你共同学习、彼此成就。</p>

    <p class="yy-about-profile__motto">Python for ideas, C++ for performance.</p>
  </div>
</section>
