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
  const tabBtns = document.querySelectorAll('.tab-btn');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const tabGroup = btn.closest('.example-body');
      if (!tabGroup) return;

      // Get the data-lang attribute
      const lang = btn.getAttribute('data-lang');

      // Remove active class from all buttons and contents in this group
      tabGroup.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      tabGroup.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

      // Add active class to clicked button and corresponding content
      btn.classList.add('active');
      const content = tabGroup.querySelector(`[data-lang="${lang}"]`);
      if (content) {
        content.classList.add('active');
      }
    });
  });

  // Set first tab active by default
  document.querySelectorAll('.example-body').forEach(body => {
    const firstBtn = body.querySelector('.tab-btn');
    const firstContent = body.querySelector('.tab-content');
    if (firstBtn && firstContent) {
      firstBtn.classList.add('active');
      firstContent.classList.add('active');
    }
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

// ========== Download Handlers ========== 

window.downloadData = function(filename) {
  const sampleData = getSampleData(filename);
  const element = document.createElement('a');
  element.setAttribute('href', 'data:text/csv;charset=utf-8,' + encodeURIComponent(sampleData));
  element.setAttribute('download', filename);
  element.style.display = 'none';
  document.body.appendChild(element);
  element.click();
  document.body.removeChild(element);
};

function getSampleData(filename) {
  const data = {
    'ecg_sample.csv': `time,signal
0.0,0.5
0.01,0.45
0.02,0.42
0.03,0.48
0.04,0.52
0.05,0.58
0.06,0.62
0.07,0.58
0.08,0.52
0.09,0.48
0.1,0.45
0.11,0.42
0.12,0.4
0.13,0.38
0.14,0.35
0.15,0.32
0.16,0.3
0.17,0.28
0.18,0.25
0.19,0.22
0.2,0.2`,
    'seismic_sample.csv': `time,displacement
0.0,0.0
0.1,0.05
0.2,0.12
0.3,0.18
0.4,0.22
0.5,0.25
0.6,0.22
0.7,0.18
0.8,0.12
0.9,0.05
1.0,0.0
1.1,-0.08
1.2,-0.15
1.3,-0.2
1.4,-0.22
1.5,-0.2
1.6,-0.15
1.7,-0.08
1.8,0.0
1.9,0.08
2.0,0.15`,
    'speech_sample.csv': `time,amplitude
0.0,0.01
0.01,0.02
0.02,0.08
0.03,0.15
0.04,0.25
0.05,0.32
0.06,0.38
0.07,0.42
0.08,0.40
0.09,0.35
0.1,0.28
0.11,0.18
0.12,0.08
0.13,0.02
0.14,0.01
0.15,-0.01
0.16,-0.02
0.17,-0.08
0.18,-0.15
0.19,-0.22
0.2,-0.25`,
    'finance_sample.csv': `date,price
2024-01-01,100.0
2024-01-02,102.5
2024-01-03,101.8
2024-01-04,103.2
2024-01-05,104.5
2024-01-06,103.8
2024-01-07,105.2
2024-01-08,106.5
2024-01-09,107.2
2024-01-10,106.8
2024-01-11,108.5
2024-01-12,109.8
2024-01-13,110.5
2024-01-14,109.2
2024-01-15,108.5
2024-01-16,110.0
2024-01-17,111.5
2024-01-18,112.2
2024-01-19,111.8
2024-01-20,113.5
2024-01-21,114.2`
  };

  return data[filename] || '';
}

// ========== Copy Code Handler ========== 

window.copyCode = function(button) {
  const codeBlock = button.closest('.code-block');
  const code = codeBlock.textContent;
  
  navigator.clipboard.writeText(code).then(() => {
    const originalText = button.textContent;
    button.textContent = 'Copied!';
    button.style.backgroundColor = '#2ecc71';
    
    setTimeout(() => {
      button.textContent = originalText;
      button.style.backgroundColor = '';
    }, 2000);
  }).catch(() => {
    alert('Failed to copy code');
  });
};
