/* Both prompts remain downloadable without JavaScript. */
document.querySelectorAll('[data-agent-handoff]').forEach(root => {
  const modes = [...root.querySelectorAll('[data-agent-mode]')];
  const panels = [...root.querySelectorAll('[data-agent-panel]')];
  const status = root.querySelector('[data-agent-copy-status]');
  modes.forEach(button => button.addEventListener('click', () => {
    modes.forEach(mode => mode.setAttribute('aria-pressed', String(mode === button)));
    panels.forEach(panel => { panel.hidden = panel.dataset.agentPanel !== button.dataset.agentMode; });
    root.querySelectorAll('[data-agent-copy]').forEach(copy => { copy.textContent = 'Copy agent brief'; });
    status.textContent = '';
  }));
  panels.forEach(panel => panel.querySelector('[data-agent-copy]').addEventListener('click', async event => {
    const button = event.currentTarget;
    const code = panel.querySelector('code');
    try {
      await navigator.clipboard.writeText(code.textContent);
      button.textContent = 'Copied';
      status.textContent = 'Agent brief copied. Replace the bracketed fields before using it.';
    } catch {
      const range = document.createRange();
      range.selectNodeContents(code);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      status.textContent = 'Brief selected. Use your keyboard copy shortcut, or download the plain-text file.';
    }
  }));
});
