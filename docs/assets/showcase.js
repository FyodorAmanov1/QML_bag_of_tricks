/* Progressive enhancement: the evidence and links work without JavaScript. */
(() => {
  const root = document.querySelector('.qml-home');
  if (!root) return;
  const filters = [...root.querySelectorAll('[data-evidence-filter]')];
  const cards = [...root.querySelectorAll('[data-evidence-status]')];
  filters.forEach(button => button.addEventListener('click', () => {
    const filter = button.dataset.evidenceFilter;
    filters.forEach(item => item.setAttribute('aria-pressed', String(item === button)));
    cards.forEach(card => { card.hidden = filter !== 'all' && card.dataset.evidenceStatus !== filter; });
    const visible = cards.filter(card => !card.hidden).length;
    root.querySelector('#qml-case-count').textContent = `${visible} ${visible === 1 ? 'case' : 'cases'}`;
  }));
  const task = root.querySelector('#qml-task');
  task.addEventListener('change', () => root.querySelectorAll('[data-task-panel]').forEach(panel => {
    panel.hidden = panel.dataset.taskPanel !== task.value;
  }));
  const code = root.querySelector('#qml-commands');
  root.querySelectorAll('[data-sdk]').forEach(button => button.addEventListener('click', () => {
    root.querySelectorAll('[data-sdk]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
    const sdk = button.dataset.sdk;
    root.querySelector('#qml-copy').textContent = 'Copy commands';
    root.querySelector('#qml-copy-status').textContent = '';
    code.textContent = `python -m pip install -e ".[${sdk},learning]"\npython -m qml_tricks.learning.case_models \\\n  --case multiclass_kernel \\\n  --framework ${sdk} \\\n  --output results/local/first_run.json`;
  }));
  root.querySelector('#qml-copy').addEventListener('click', async event => {
    const button = event.currentTarget;
    try {
      await navigator.clipboard.writeText(code.textContent);
      button.textContent = 'Copied';
      root.querySelector('#qml-copy-status').textContent = 'Commands copied to clipboard.';
    } catch {
      const range = document.createRange();
      range.selectNodeContents(code);
      const selection = window.getSelection();
      selection.removeAllRanges(); selection.addRange(range);
      root.querySelector('#qml-copy-status').textContent = 'Commands selected. Use your keyboard copy shortcut.';
    }
  });
})();
