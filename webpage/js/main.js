// Ferromode Website - Main JavaScript

document.addEventListener('DOMContentLoaded', () => {
  initializeNav();
  initializeTabs();
  initializeFAQ();
  setActiveNav();
});

// ========== Navigation ========== 

function initializeNav() {
  const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
  const navLinks = document.querySelector('.nav-links');

  if (mobileMenuBtn) {
    mobileMenuBtn.addEventListener('click', () => {
      navLinks.classList.toggle('mobile-open');
    });

    // Close menu when a link is clicked
    document.querySelectorAll('.nav-links a').forEach(link => {
      link.addEventListener('click', () => {
        navLinks.classList.remove('mobile-open');
      });
    });

    // Close menu when clicking outside
    document.addEventListener('click', (e) => {
      if (!e.target.closest('nav')) {
        navLinks.classList.remove('mobile-open');
      }
    });
  }
}

function setActiveNav() {
  const navLinks = document.querySelectorAll('.nav-links a');
  const currentPage = window.location.pathname.split('/').pop() || 'index.html';

  navLinks.forEach(link => {
    const href = link.getAttribute('href');
    if (href === currentPage || (currentPage === '' && href === 'index.html')) {
      link.classList.add('active');
    } else {
      link.classList.remove('active');
    }
  });
}

// ========== Tabs ========== 

function initializeTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const tabGroup = btn.closest('.example-body');
      if (!tabGroup) return;

      const lang = btn.getAttribute('data-lang');

      tabGroup.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      tabGroup.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      // Use .tab-content in selector so we don't accidentally match the button itself
      const content = tabGroup.querySelector(`.tab-content[data-lang="${lang}"]`);
      if (content) content.classList.add('active');
    });
  });
}

// ========== FAQ Accordion ========== 

function initializeFAQ() {
  const faqQuestions = document.querySelectorAll('.faq-question');

  faqQuestions.forEach(question => {
    question.addEventListener('click', () => {
      const faqItem = question.closest('.faq-item');
      const answer = faqItem.querySelector('.faq-answer');
      const toggle = question.querySelector('.faq-toggle');

      // Close other open items
      document.querySelectorAll('.faq-item').forEach(item => {
        if (item !== faqItem) {
          item.querySelector('.faq-answer').classList.remove('open');
          item.querySelector('.faq-toggle').classList.remove('open');
        }
      });

      // Toggle current item
      answer.classList.toggle('open');
      toggle.classList.toggle('open');
    });
  });
}

// ========== Smooth Scroll for Anchor Links ========== 

document.querySelectorAll('a[href^="#"]').forEach(link => {
  link.addEventListener('click', (e) => {
    const target = document.querySelector(link.getAttribute('href'));
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth' });
    }
  });
});

// ========== Copy Code Handler ==========

window.copyCode = function(button) {
  const code = button.closest('.code-block').querySelector('code').innerText;
  navigator.clipboard.writeText(code).then(() => {
    const orig = button.innerText;
    button.innerText = '✓ Copied!';
    setTimeout(() => { button.innerText = orig; }, 2000);
  });
};
