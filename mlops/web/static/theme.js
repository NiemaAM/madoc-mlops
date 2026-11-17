/* The theme, applied before the page is drawn: "light", "dark", or nothing (the system's). app.js changes it. */
(function () {
  let theme = null;
  try { theme = localStorage.getItem("mlops-theme"); } catch (_) { /* storage blocked: the system's theme */ }
  if (theme === "light" || theme === "dark") document.documentElement.setAttribute("data-theme", theme);
})();
