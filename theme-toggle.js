// Theme toggle functionality
const themeToggle = document.getElementById('theme-toggle');
const body = document.body;
const themeIcon = document.querySelector('.theme-icon');

// Check for saved theme preference or default to dark mode
const currentTheme = localStorage.getItem('theme') || 'dark';
body.classList.toggle('dark-mode', currentTheme === 'dark');
updateThemeIcon(currentTheme === 'dark');

themeToggle.addEventListener('click', () => {
    body.classList.toggle('dark-mode');
    const isDark = body.classList.contains('dark-mode');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    updateThemeIcon(isDark);
});

function updateThemeIcon(isDark) {
    themeIcon.textContent = isDark ? '☀️' : '🌙';
}
