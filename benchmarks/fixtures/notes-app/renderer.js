const box = document.getElementById('q');
const list = document.getElementById('results');

box.addEventListener('input', async () => {
  const found = await window.notes.search(box.value);
  list.innerHTML = '';
  for (const note of found) {
    const item = document.createElement('li');
    item.textContent = note.title;
    item.addEventListener('click', () => window.notes.copy(note.body));
    list.appendChild(item);
  }
});
